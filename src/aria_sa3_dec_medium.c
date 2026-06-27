/*
 * aria_sa3_dec_medium.c - taae_v2 decoder for the MEDIUM model (latent -> audio).
 *
 * Differs from small-music (aria_sa3_dec.c): transformer_dim 1536, depth 12,
 * inner 4608, mapping WNConv1d k1 (not k3), and -- the big one -- the resampling
 * runs the whole token sequence through SLIDING-WINDOW (banded) attention instead
 * of chunked attention with a midpoint-shift halo. The later blocks use a
 * sinusoidal FF. The block kernel is taae_med_block_forward (aria_taae.c).
 *
 * Flow (latent[256,T] -> audio[2, T*4096]):
 *   1. softnorm: z = latent * running_std
 *   2. transpose + Linear 256->1536 (decoder.layers.1)
 *   3. per latent token: [token, 16x new_tokens] -> seq[T*17, 1536]
 *   4. 12 blocks over the full seq, band mask window [17,17]; blocks with
 *      (12-i) < 8 use the sinusoidal FF
 *   5. take the last 16 of each 17-group -> feat[1536, T*16]
 *   6. WNConv1d mapping 1536->512 (k1), unpatch (256) -> stereo audio
 */

#include "aria_sa3_dec.h"
#include "aria_taae.h"
#include "aria_ops.h"
#include "aria_arena.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MED_D     1536
#define MED_DEPTH 12
#define MED_H     24
#define MED_HD    64
#define MED_INNER 4608   /* ff_mult 3 */
#define MED_OUT   512
#define MED_SEG   17     /* 1 token + 16 new */
#define MED_WIN   17     /* sliding-window half-width [17,17] */

struct aria_sa3_dec_medium {
    float running_std;
    const float *proj_w, *proj_b;     /* layers.1 [1536,256],[1536] */
    const float *new_tokens;          /* [1536] */
    taae_block_w blocks[MED_DEPTH];
    float *mapping_w;                 /* folded [512,1536] (k1, owned) */
    const float *mapping_b;           /* [512] */
    int failed;
};

static const float *track(aria_sa3_dec_medium *m, safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) { fprintf(stderr, "dec_medium: missing %s\n", name); m->failed = 1; return NULL; }
    const float *p = safetensors_f32_ptr(sf, t);
    if (!p) { fprintf(stderr, "dec_medium: %s not F32\n", name); m->failed = 1; return NULL; }
    return p;
}
static float track_scalar(aria_sa3_dec_medium *m, safetensors_file_t *sf, const char *name) {
    const float *p = track(m, sf, name); return p ? p[0] : 0.0f;
}

aria_sa3_dec_medium *aria_sa3_dec_medium_load(safetensors_file_t *sf) {
    aria_sa3_dec_medium *m = calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->running_std = track_scalar(m, sf, "pretransform.model.bottleneck.running_std");
    m->proj_w = track(m, sf, "pretransform.model.decoder.layers.1.weight");
    m->proj_b = track(m, sf, "pretransform.model.decoder.layers.1.bias");
    m->new_tokens = track(m, sf, "pretransform.model.decoder.layers.3.new_tokens");

    char nm[256];
    #define P "pretransform.model.decoder.layers.3.transformers.%d."
    for (int i = 0; i < MED_DEPTH; i++) {
        taae_block_w *b = &m->blocks[i];
        #define SCA(dst, suf) do { snprintf(nm, sizeof(nm), P suf, i); dst = track_scalar(m, sf, nm); } while (0)
        #define PTR(dst, suf) do { snprintf(nm, sizeof(nm), P suf, i); dst = track(m, sf, nm); } while (0)
        SCA(b->pre_alpha, "pre_norm.alpha"); PTR(b->pre_gamma, "pre_norm.gamma"); PTR(b->pre_beta, "pre_norm.beta");
        PTR(b->to_qkv, "self_attn.to_qkv.weight");
        SCA(b->qn_alpha, "self_attn.q_norm.alpha"); PTR(b->qn_gamma, "self_attn.q_norm.gamma"); PTR(b->qn_beta, "self_attn.q_norm.beta");
        SCA(b->kn_alpha, "self_attn.k_norm.alpha"); PTR(b->kn_gamma, "self_attn.k_norm.gamma"); PTR(b->kn_beta, "self_attn.k_norm.beta");
        PTR(b->to_out, "self_attn.to_out.weight");
        SCA(b->ff_alpha, "ff_norm.alpha"); PTR(b->ff_gamma, "ff_norm.gamma"); PTR(b->ff_beta, "ff_norm.beta");
        PTR(b->ff_in_w, "ff.ff.0.proj.weight"); PTR(b->ff_in_b, "ff.ff.0.proj.bias");
        PTR(b->ff_out_w, "ff.ff.2.weight"); PTR(b->ff_out_b, "ff.ff.2.bias");
        #undef SCA
        #undef PTR
    }
    #undef P

    /* fold the weight-normalized mapping conv (k1): fw[o,i] = v[o,i] * g[o]/||v[o]|| */
    const float *g = track(m, sf, "pretransform.model.decoder.layers.3.mapping.weight_g");   /* [512,1,1] */
    const float *vv = track(m, sf, "pretransform.model.decoder.layers.3.mapping.weight_v");  /* [512,1536,1] */
    m->mapping_b = track(m, sf, "pretransform.model.decoder.layers.3.mapping.bias");
    if (!m->failed) {
        int rowlen = MED_D;
        float *fw = malloc((size_t)MED_OUT * rowlen * sizeof(float));
        for (int o = 0; o < MED_OUT; o++) {
            double ss = 0.0;
            const float *vr = vv + (size_t)o * rowlen;
            for (int idx = 0; idx < rowlen; idx++) ss += (double)vr[idx] * vr[idx];
            float scale = (float)((double)g[o] / sqrt(ss));
            float *fr = fw + (size_t)o * rowlen;
            for (int idx = 0; idx < rowlen; idx++) fr[idx] = vr[idx] * scale;
        }
        m->mapping_w = fw;
    }
    if (m->failed) { aria_sa3_dec_medium_free(m); return NULL; }
    return m;
}

void aria_sa3_dec_medium_free(aria_sa3_dec_medium *m) {
    if (!m) return;
    free(m->mapping_w);
    free(m);
}

void aria_sa3_dec_medium_get_view(const aria_sa3_dec_medium *m, aria_sa3_dec_medium_view *v) {
    v->running_std = m->running_std;
    v->proj_w = m->proj_w; v->proj_b = m->proj_b; v->new_tokens = m->new_tokens;
    v->mapping_w = m->mapping_w; v->mapping_b = m->mapping_b;
    for (int i = 0; i < MED_DEPTH; i++) {
        const taae_block_w *s = &m->blocks[i];
        aria_taae_block_view *d = &v->blocks[i];
        d->pre_alpha = s->pre_alpha; d->qn_alpha = s->qn_alpha;
        d->kn_alpha = s->kn_alpha;   d->ff_alpha = s->ff_alpha;
        d->pre_gamma = s->pre_gamma; d->pre_beta = s->pre_beta;
        d->qn_gamma = s->qn_gamma;   d->qn_beta = s->qn_beta;
        d->kn_gamma = s->kn_gamma;   d->kn_beta = s->kn_beta;
        d->ff_gamma = s->ff_gamma;   d->ff_beta = s->ff_beta;
        d->to_qkv = s->to_qkv; d->to_out = s->to_out;
        d->ff_in_w = s->ff_in_w; d->ff_in_b = s->ff_in_b;
        d->ff_out_w = s->ff_out_w; d->ff_out_b = s->ff_out_b;
    }
}

/* latent[256,T] -> audio[2, T*4096] */
void aria_sa3_dec_medium_forward(const aria_sa3_dec_medium *m, float *audio, const float *latent, int T) {
    const int D = MED_D;
    int N = T * MED_SEG;                  /* full token sequence length */

    /* 1. softnorm + transpose [256,T] -> [T,256], proj -> seq token slots */
    float *ztc = malloc((size_t)T * 256 * sizeof(float));
    if (!ztc) return;
    for (int t = 0; t < T; t++)
        for (int c = 0; c < 256; c++) ztc[(size_t)t * 256 + c] = latent[(size_t)c * T + t] * m->running_std;
    float *proj = malloc((size_t)T * D * sizeof(float));
    aria_linear(proj, ztc, m->proj_w, m->proj_b, T, 256, D);
    free(ztc);

    /* 3. seq[T*17, 1536]: per token [proj_token, 16x new_tokens] */
    float *seq = malloc((size_t)N * D * sizeof(float));
    for (int t = 0; t < T; t++) {
        memcpy(seq + (size_t)(t * MED_SEG) * D, proj + (size_t)t * D, (size_t)D * sizeof(float));
        for (int j = 1; j < MED_SEG; j++)
            memcpy(seq + (size_t)(t * MED_SEG + j) * D, m->new_tokens, (size_t)D * sizeof(float));
    }
    free(proj);

    /* rope + sliding-window band mask [N,N] (window [17,17]) */
    float *rc = malloc((size_t)N * (TAAE_ROT / 2) * sizeof(float));
    float *rs = malloc((size_t)N * (TAAE_ROT / 2) * sizeof(float));
    aria_rope_freqs(rc, rs, N, TAAE_ROT, 10000.0f);
    float *mask = malloc((size_t)N * N * sizeof(float));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            int d = j - i;
            mask[(size_t)i * N + j] = (d >= -MED_WIN && d <= MED_WIN) ? 0.0f : -INFINITY;
        }

    /* 4. 12 blocks over the full sequence; blocks with (depth-i) < 8 are sinusoidal */
    aria_arena ar;
    aria_arena_init(&ar, taae_med_block_floats(N, D, MED_INNER) * sizeof(float));
    for (int b = 0; b < MED_DEPTH; b++) {
        int sinusoidal = (MED_DEPTH - b) < 8;
        taae_med_block_forward(seq, N, D, MED_H, MED_HD, MED_INNER, &m->blocks[b], rc, rs, mask, sinusoidal, &ar);
    }
    aria_arena_free(&ar);
    free(rc); free(rs); free(mask);

    /* 5. last 16 of each 17-group -> feat[1536, T*16] (channel-major) */
    int Lt = T * 16;
    float *feat = malloc((size_t)D * Lt * sizeof(float));
    for (int t = 0; t < T; t++)
        for (int j = 0; j < 16; j++)
            for (int d = 0; d < D; d++)
                feat[(size_t)d * Lt + (size_t)t * 16 + j] = seq[(size_t)(t * MED_SEG + 1 + j) * D + d];
    free(seq);

    /* 6. WNConv1d mapping 1536->512 (k1 p0), then unpatch -> stereo */
    float *mapped = malloc((size_t)MED_OUT * Lt * sizeof(float));
    aria_conv1d(mapped, feat, m->mapping_w, m->mapping_b, D, MED_OUT, 1, 0, Lt);
    free(feat);
    aria_unpatch_stereo(audio, mapped, Lt);   /* [2, Lt*256] = [2, T*4096] */
    free(mapped);
}
