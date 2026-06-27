/*
 * aria_sa3_dec.c - taae_v2 decoder (CPU). See aria_sa3_dec.h for the spec.
 */

#include "aria_sa3_dec.h"
#include "aria_ops.h"
#include "aria_arena.h"
#include "aria_taae.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DEC_D   TAAE_D    /* transformer model dim (768) */
#define DEC_ROT TAAE_ROT  /* rotated dims (32) */
#define DEC_S   TAAE_S    /* effective chunk = 2 * 17 (34) */
#define DEC_OUT 512        /* decoder out channels */

struct aria_sa3_dec {
    float running_std;
    const float *proj_w, *proj_b;        /* layers.1 [768,256],[768] */
    const float *new_tokens;             /* [768] */
    taae_block_w blocks[6];
    float *mapping_w;                    /* folded [512,768,3] (owned) */
    const float *mapping_b;              /* [512] */
    int failed;
};

/* ---- channel<->token transpose (decoder-local) ---- */
static void transpose_ct_tc(float *dst, const float *src, int C, int T) {
    for (int c = 0; c < C; c++) for (int t = 0; t < T; t++) dst[(size_t)t * C + c] = src[(size_t)c * T + t];
}

void aria_sa3_softnorm_decode(const aria_sa3_dec *m, float *z, const float *latent, int T) {
    for (size_t i = 0; i < (size_t)256 * T; i++) z[i] = latent[i] * m->running_std;
}

void aria_unpatch_stereo(float *audio, const float *dec, int L) {
    for (int c = 0; c < 2; c++)
        for (int l = 0; l < L; l++)
            for (int hh = 0; hh < 256; hh++)
                audio[(size_t)c * L * 256 + (size_t)l * 256 + hh] = dec[(size_t)(c * 256 + hh) * L + l];
}

void aria_sa3_same_decode(const aria_sa3_dec *m, float *dec, const float *z, int T) {
    const int D = DEC_D;
    /* transpose [256,T] -> [T,256], proj -> [T,768] */
    float *ztc = calloc((size_t)T * 256, sizeof(float));
    transpose_ct_tc(ztc, z, 256, T);
    float *x = malloc((size_t)T * D * sizeof(float));
    aria_linear(x, ztc, m->proj_w, m->proj_b, T, 256, D);
    free(ztc);

    /* zero-pad T to even Tp, expand with new_tokens -> seq[L,768], L = Tp*17 */
    int Tp = (T % 2 == 0) ? T : T + 1;
    int L = Tp * 17;
    float *seq = malloc((size_t)L * D * sizeof(float));
    for (int t = 0; t < Tp; t++) {
        const float *xt = (t < T) ? x + (size_t)t * D : NULL;  /* padded token = zeros */
        if (xt) memcpy(seq + (size_t)(t * 17) * D, xt, (size_t)D * sizeof(float));
        else    memset(seq + (size_t)(t * 17) * D, 0, (size_t)D * sizeof(float));
        for (int j = 1; j <= 16; j++)
            memcpy(seq + (size_t)(t * 17 + j) * D, m->new_tokens, (size_t)D * sizeof(float));
    }
    free(x);

    /* rope tables for S=34 */
    float rcos[DEC_S * (DEC_ROT / 2)], rsin[DEC_S * (DEC_ROT / 2)];
    aria_rope_freqs(rcos, rsin, DEC_S, DEC_ROT, 10000.0f);

    int nth = 1;
    #ifdef _OPENMP
    nth = omp_get_max_threads();
    #endif
    aria_arena *arenas = malloc((size_t)nth * sizeof(aria_arena));
    for (int i = 0; i < nth; i++) aria_arena_init(&arenas[i], taae_block_arena_bytes());
    taae_chunk_pass(seq, L, &m->blocks[0], rcos, rsin, 0, arenas);  /* blocks 0,1,2 */
    taae_chunk_pass(seq, L, &m->blocks[3], rcos, rsin, 1, arenas);  /* blocks 3,4,5 midpoint-shift */
    for (int i = 0; i < nth; i++) aria_arena_free(&arenas[i]);
    free(arenas);

    /* extract last 16 of each 17-group -> feat[768, Tp*16] */
    int Lout = Tp * 16;
    float *feat = malloc((size_t)D * Lout * sizeof(float));
    for (int t = 0; t < Tp; t++)
        for (int j = 0; j < 16; j++)
            for (int d = 0; d < D; d++)
                feat[(size_t)d * Lout + (size_t)t * 16 + j] = seq[(size_t)(t * 17 + 1 + j) * D + d];
    free(seq);

    /* WNConv1d mapping 768 -> 512, k3 p1 */
    float *mapped = malloc((size_t)DEC_OUT * Lout * sizeof(float));
    aria_conv1d(mapped, feat, m->mapping_w, m->mapping_b, D, DEC_OUT, 3, 1, Lout);
    free(feat);

    /* trim to T*16 */
    int Lt = T * 16;
    for (int c = 0; c < DEC_OUT; c++)
        memcpy(dec + (size_t)c * Lt, mapped + (size_t)c * Lout, (size_t)Lt * sizeof(float));
    free(mapped);
}

void aria_sa3_dec_forward(const aria_sa3_dec *m, float *out_audio, const float *latent, int T) {
    float *z = malloc((size_t)256 * T * sizeof(float));
    aria_sa3_softnorm_decode(m, z, latent, T);
    int Lt = T * 16;
    float *dec = malloc((size_t)DEC_OUT * Lt * sizeof(float));
    aria_sa3_same_decode(m, dec, z, T);
    free(z);
    aria_unpatch_stereo(out_audio, dec, Lt);  /* [2, Lt*256] = [2, T*4096] */
    free(dec);
}

void aria_sa3_dec_block_test(const aria_sa3_dec *m, int idx, float *xc, int N) {
    float *rcos = malloc((size_t)N * (DEC_ROT / 2) * sizeof(float));
    float *rsin = malloc((size_t)N * (DEC_ROT / 2) * sizeof(float));
    aria_rope_freqs(rcos, rsin, N, DEC_ROT, 10000.0f);
    aria_arena ar;
    size_t fl = 12 * (size_t)N * DEC_D + (size_t)N * TAAE_QKV + (size_t)N * N + 3 * (size_t)N * TAAE_INNER + 4096;
    aria_arena_init(&ar, fl * sizeof(float));
    taae_block_forward(xc, N, &m->blocks[idx], rcos, rsin, &ar);
    aria_arena_free(&ar);
    free(rcos); free(rsin);
}

/* ---- loader ---- */
/* zero-copy: borrow the F32 weight directly from the mmap (valid while sf open) */
static const float *track(aria_sa3_dec *m, safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) { fprintf(stderr, "aria_sa3_dec_load: missing %s\n", name); m->failed = 1; return NULL; }
    const float *p = safetensors_f32_ptr(sf, t);
    if (!p) { fprintf(stderr, "aria_sa3_dec_load: %s is not F32\n", name); m->failed = 1; return NULL; }
    return p;
}
static float track_scalar(aria_sa3_dec *m, safetensors_file_t *sf, const char *name) {
    const float *p = track(m, sf, name);
    return p ? p[0] : 0.0f;
}

aria_sa3_dec *aria_sa3_dec_load(safetensors_file_t *sf) {
    aria_sa3_dec *m = calloc(1, sizeof(*m));
    if (!m) return NULL;

    m->running_std = track_scalar(m, sf, "pretransform.model.bottleneck.running_std");
    m->proj_w = track(m, sf, "pretransform.model.decoder.layers.1.weight");
    m->proj_b = track(m, sf, "pretransform.model.decoder.layers.1.bias");
    m->new_tokens = track(m, sf, "pretransform.model.decoder.layers.3.new_tokens");

    char nm[256];
    #define P "pretransform.model.decoder.layers.3.transformers.%d."
    for (int i = 0; i < 6; i++) {
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

    /* fold the weight-normalized mapping conv */
    const float *g = track(m, sf, "pretransform.model.decoder.layers.3.mapping.weight_g");   /* [512,1,1] */
    const float *vv = track(m, sf, "pretransform.model.decoder.layers.3.mapping.weight_v");  /* [512,768,3] */
    m->mapping_b = track(m, sf, "pretransform.model.decoder.layers.3.mapping.bias");
    if (!m->failed) {
        int rowlen = 768 * 3;
        float *fw = malloc((size_t)512 * rowlen * sizeof(float));
        for (int o = 0; o < 512; o++) {
            double ss = 0.0;
            const float *vr = vv + (size_t)o * rowlen;
            for (int idx = 0; idx < rowlen; idx++) ss += (double)vr[idx] * vr[idx];
            float scale = (float)((double)g[o] / sqrt(ss));
            float *fr = fw + (size_t)o * rowlen;
            for (int idx = 0; idx < rowlen; idx++) fr[idx] = vr[idx] * scale;
        }
        m->mapping_w = fw;  /* owned via dec_free's mapping_w branch */
    }

    if (m->failed) { aria_sa3_dec_free(m); return NULL; }
    return m;
}

void aria_sa3_dec_free(aria_sa3_dec *m) {
    if (!m) return;
    free(m->mapping_w);   /* the folded conv weight is the only owned buffer */
    free(m);
}

void aria_sa3_dec_get_view(const aria_sa3_dec *m, aria_sa3_dec_view *v) {
    v->running_std = m->running_std;
    v->proj_w = m->proj_w; v->proj_b = m->proj_b; v->new_tokens = m->new_tokens;
    v->mapping_w = m->mapping_w; v->mapping_b = m->mapping_b;
    for (int i = 0; i < 6; i++) {
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
