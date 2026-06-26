/*
 * aria_sa3_dit.c - SA3 conditioning embedders + DiT transformer block (CPU).
 */

#include "aria_sa3_dit.h"
#include "aria_ops.h"
#include "aria_cond.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

void aria_sa3_mlp2(float *out, const float *in, int N, int in_dim, int mid, int out_dim,
                   const float *W0, const float *b0, const float *W2, const float *b2) {
    float *h = malloc((size_t)N * mid * sizeof(float));
    aria_linear(h, in, W0, b0, N, in_dim, mid);
    aria_silu(h, N * mid);
    aria_linear(out, h, W2, b2, N, mid, out_dim);
    free(h);
}

void aria_sa3_timestep_embed(float *out, float t, int feat_dim, int ed,
                             const float *W0, const float *b0, const float *W2, const float *b2) {
    float *fourier = malloc((size_t)feat_dim * sizeof(float));
    aria_expo_fourier(fourier, t, feat_dim, 0.5f, 10000.0f);
    aria_sa3_mlp2(out, fourier, 1, feat_dim, ed, ed, W0, b0, W2, b2);
    free(fourier);
}

/* dst[H,S,hd] from src[S, stride] reading hd-blocks at `offset` per head:
 * dst[h,s,d] = src[s*stride + offset + h*hd + d] */
static void extract_heads(float *dst, const float *src, int S, int H, int hd,
                          int stride, int offset) {
    for (int h = 0; h < H; h++)
        for (int s = 0; s < S; s++) {
            const float *sp = src + (size_t)s * stride + offset + (size_t)h * hd;
            float *dp = dst + ((size_t)h * S + s) * hd;
            memcpy(dp, sp, (size_t)hd * sizeof(float));
        }
}

/* dst[S, H*hd] from src[H,S,hd]: dst[s, h*hd+d] = src[h,s,d] */
static void merge_heads(float *dst, const float *src, int S, int H, int hd) {
    for (int h = 0; h < H; h++)
        for (int s = 0; s < S; s++) {
            const float *sp = src + ((size_t)h * S + s) * hd;
            float *dp = dst + (size_t)s * (H * hd) + (size_t)h * hd;
            memcpy(dp, sp, (size_t)hd * sizeof(float));
        }
}

/* y = y*(1+scale)+shift over rows; scale/shift are [dim] broadcast over S rows. */
static void adaln_modulate(float *y, const float *scale, const float *shift, int S, int dim) {
    for (int s = 0; s < S; s++) {
        float *yr = y + (size_t)s * dim;
        for (int i = 0; i < dim; i++) yr[i] = yr[i] * (1.0f + scale[i]) + shift[i];
    }
}

/* y *= sigmoid(1 - gate), gate [dim] broadcast over S rows. */
static void gate_sigmoid(float *y, const float *gate, int S, int dim) {
    for (int s = 0; s < S; s++) {
        float *yr = y + (size_t)s * dim;
        for (int i = 0; i < dim; i++) yr[i] *= sigmoidf(1.0f - gate[i]);
    }
}

void aria_dit_block_forward(float *x, int S, int dim, int num_heads, int head_dim, int inner,
                            const float *context, int Sc, int dim_ctx,
                            const float *global_cond,
                            const float *rope_cos, const float *rope_sin, int rot_dim,
                            const aria_dit_block_w *w) {
    int H = num_heads, hd = head_dim;
    const float eps_norm = 1e-5f, eps_qk = 1e-6f;

    /* modulation = to_scale_shift_gate + global_cond, chunked into 6 [dim] slices */
    float *mod = malloc((size_t)6 * dim * sizeof(float));
    for (int i = 0; i < 6 * dim; i++) mod[i] = w->to_scale_shift_gate[i] + global_cond[i];
    const float *scale_self = mod, *shift_self = mod + dim, *gate_self = mod + 2 * dim;
    const float *scale_ff = mod + 3 * dim, *shift_ff = mod + 4 * dim, *gate_ff = mod + 5 * dim;

    float *residual = malloc((size_t)S * dim * sizeof(float));
    float *h = malloc((size_t)S * dim * sizeof(float));
    float *o = malloc((size_t)S * dim * sizeof(float));
    float *merged = malloc((size_t)S * dim * sizeof(float));
    float *qh = malloc((size_t)H * S * hd * sizeof(float));
    float *kh = malloc((size_t)H * S * hd * sizeof(float));
    float *vh = malloc((size_t)H * S * hd * sizeof(float));
    float *ao = malloc((size_t)H * S * hd * sizeof(float));

    /* ---------- self-attention ---------- */
    memcpy(residual, x, (size_t)S * dim * sizeof(float));
    aria_rmsnorm(h, x, w->pre_norm, S, dim, eps_norm);
    adaln_modulate(h, scale_self, shift_self, S, dim);
    {
        float *qkv = malloc((size_t)S * 3 * dim * sizeof(float));
        aria_linear(qkv, h, w->sa_to_qkv, NULL, S, dim, 3 * dim);
        extract_heads(qh, qkv, S, H, hd, 3 * dim, 0);
        extract_heads(kh, qkv, S, H, hd, 3 * dim, dim);
        extract_heads(vh, qkv, S, H, hd, 3 * dim, 2 * dim);
        free(qkv);
    }
    aria_rmsnorm(qh, qh, w->sa_q_norm, H * S, hd, eps_qk);
    aria_rmsnorm(kh, kh, w->sa_k_norm, H * S, hd, eps_qk);
    aria_rope_apply(qh, rope_cos, rope_sin, H, S, hd, rot_dim);
    aria_rope_apply(kh, rope_cos, rope_sin, H, S, hd, rot_dim);
    aria_attention(ao, qh, kh, vh, H, S, S, hd, NULL);
    merge_heads(merged, ao, S, H, hd);
    aria_linear(o, merged, w->sa_to_out, NULL, S, dim, dim);
    gate_sigmoid(o, gate_self, S, dim);
    for (size_t i = 0; i < (size_t)S * dim; i++) x[i] = residual[i] + o[i];

    /* ---------- cross-attention (no rope, no gate) ---------- */
    memcpy(residual, x, (size_t)S * dim * sizeof(float));
    aria_rmsnorm(h, x, w->cross_norm, S, dim, eps_norm);
    {
        float *q = malloc((size_t)S * dim * sizeof(float));
        aria_linear(q, h, w->ca_to_q, NULL, S, dim, dim);
        extract_heads(qh, q, S, H, hd, dim, 0);
        free(q);
        float *kv = malloc((size_t)Sc * 2 * dim * sizeof(float));
        aria_linear(kv, context, w->ca_to_kv, NULL, Sc, dim_ctx, 2 * dim);
        float *kc = malloc((size_t)H * Sc * hd * sizeof(float));
        float *vc = malloc((size_t)H * Sc * hd * sizeof(float));
        float *aoc = malloc((size_t)H * S * hd * sizeof(float));
        extract_heads(kc, kv, Sc, H, hd, 2 * dim, 0);
        extract_heads(vc, kv, Sc, H, hd, 2 * dim, dim);
        free(kv);
        aria_rmsnorm(qh, qh, w->ca_q_norm, H * S, hd, eps_qk);
        aria_rmsnorm(kc, kc, w->ca_k_norm, H * Sc, hd, eps_qk);
        aria_attention(aoc, qh, kc, vc, H, S, Sc, hd, NULL);
        merge_heads(merged, aoc, S, H, hd);
        free(kc); free(vc); free(aoc);
    }
    aria_linear(o, merged, w->ca_to_out, NULL, S, dim, dim);
    for (size_t i = 0; i < (size_t)S * dim; i++) x[i] = residual[i] + o[i];

    /* ---------- feed-forward (GLU) ---------- */
    memcpy(residual, x, (size_t)S * dim * sizeof(float));
    aria_rmsnorm(h, x, w->ff_norm, S, dim, eps_norm);
    adaln_modulate(h, scale_ff, shift_ff, S, dim);
    aria_ff_glu(o, h, S, dim, inner, dim, w->ff_in_w, w->ff_in_b, w->ff_out_w, w->ff_out_b);
    gate_sigmoid(o, gate_ff, S, dim);
    for (size_t i = 0; i < (size_t)S * dim; i++) x[i] = residual[i] + o[i];

    free(mod); free(residual); free(h); free(o); free(merged);
    free(qh); free(kh); free(vh); free(ao);
}

/* ---------------- full DiT model ---------------- */

struct aria_sa3_dit {
    int depth, ed, num_heads, head_dim, inner, io_ch, n_mem, rot_dim, cond_dim, ts_feat_dim;
    const float *preprocess, *postprocess, *project_in, *project_out, *memory_tokens;
    const float *to_cond0, *to_cond2, *to_global0, *to_global2;
    const float *to_ts0_w, *to_ts0_b, *to_ts2_w, *to_ts2_b;
    const float *gce0_w, *gce0_b, *gce2_w, *gce2_b;
    aria_dit_block_w *blocks;
    int failed;
};

/* zero-copy: borrow the F32 weight directly from the mmap (valid while sf open) */
static const float *track(aria_sa3_dit *m, safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) { fprintf(stderr, "aria_sa3_dit_load: missing tensor %s\n", name); m->failed = 1; return NULL; }
    const float *p = safetensors_f32_ptr(sf, t);
    if (!p) { fprintf(stderr, "aria_sa3_dit_load: %s is not F32\n", name); m->failed = 1; return NULL; }
    return p;
}

aria_sa3_dit *aria_sa3_dit_load(safetensors_file_t *sf, const aria_sa3_config *cfg) {
    aria_sa3_dit *m = calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->depth = cfg->depth;
    m->ed = cfg->embed_dim;
    m->num_heads = cfg->num_heads;
    m->head_dim = cfg->head_dim;
    m->inner = (int)(cfg->embed_dim * cfg->ff_mult);
    m->io_ch = cfg->io_channels;
    m->n_mem = cfg->num_memory_tokens;
    m->rot_dim = cfg->head_dim / 2 < 32 ? 32 : cfg->head_dim / 2;  /* max(head_dim//2, 32) */
    m->cond_dim = cfg->cond_token_dim;
    m->ts_feat_dim = 256;
    m->blocks = calloc((size_t)m->depth, sizeof(aria_dit_block_w));

    m->preprocess    = track(m, sf, "model.model.preprocess_conv.weight");
    m->postprocess   = track(m, sf, "model.model.postprocess_conv.weight");
    m->project_in    = track(m, sf, "model.model.transformer.project_in.weight");
    m->project_out   = track(m, sf, "model.model.transformer.project_out.weight");
    m->memory_tokens = track(m, sf, "model.model.transformer.memory_tokens");
    m->to_cond0      = track(m, sf, "model.model.to_cond_embed.0.weight");
    m->to_cond2      = track(m, sf, "model.model.to_cond_embed.2.weight");
    m->to_global0    = track(m, sf, "model.model.to_global_embed.0.weight");
    m->to_global2    = track(m, sf, "model.model.to_global_embed.2.weight");
    m->to_ts0_w      = track(m, sf, "model.model.to_timestep_embed.0.weight");
    m->to_ts0_b      = track(m, sf, "model.model.to_timestep_embed.0.bias");
    m->to_ts2_w      = track(m, sf, "model.model.to_timestep_embed.2.weight");
    m->to_ts2_b      = track(m, sf, "model.model.to_timestep_embed.2.bias");
    m->gce0_w        = track(m, sf, "model.model.transformer.global_cond_embedder.0.weight");
    m->gce0_b        = track(m, sf, "model.model.transformer.global_cond_embedder.0.bias");
    m->gce2_w        = track(m, sf, "model.model.transformer.global_cond_embedder.2.weight");
    m->gce2_b        = track(m, sf, "model.model.transformer.global_cond_embedder.2.bias");

    char nm[256];
    for (int i = 0; i < m->depth; i++) {
        aria_dit_block_w *b = &m->blocks[i];
        #define BW(field, suffix) do { \
            snprintf(nm, sizeof(nm), "model.model.transformer.layers.%d." suffix, i); \
            b->field = track(m, sf, nm); } while (0)
        BW(pre_norm, "pre_norm.gamma");
        BW(cross_norm, "cross_attend_norm.gamma");
        BW(ff_norm, "ff_norm.gamma");
        BW(sa_to_qkv, "self_attn.to_qkv.weight");
        BW(sa_q_norm, "self_attn.q_norm.gamma");
        BW(sa_k_norm, "self_attn.k_norm.gamma");
        BW(sa_to_out, "self_attn.to_out.weight");
        BW(ca_to_q, "cross_attn.to_q.weight");
        BW(ca_to_kv, "cross_attn.to_kv.weight");
        BW(ca_q_norm, "cross_attn.q_norm.gamma");
        BW(ca_k_norm, "cross_attn.k_norm.gamma");
        BW(ca_to_out, "cross_attn.to_out.weight");
        BW(ff_in_w, "ff.ff.0.proj.weight");
        BW(ff_in_b, "ff.ff.0.proj.bias");
        BW(ff_out_w, "ff.ff.2.weight");
        BW(ff_out_b, "ff.ff.2.bias");
        BW(to_scale_shift_gate, "to_scale_shift_gate");
        #undef BW
    }

    if (m->failed) { aria_sa3_dit_free(m); return NULL; }
    return m;
}

void aria_sa3_dit_free(aria_sa3_dit *m) {
    if (!m) return;
    free(m->blocks);   /* weights are borrowed from the mmap; nothing else to free */
    free(m);
}

/* transpose [C,T] -> [T,C] */
static void transpose_ct_tc(float *dst, const float *src, int C, int T) {
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++)
            dst[(size_t)t * C + c] = src[(size_t)c * T + t];
}
/* transpose [T,C] -> [C,T] */
static void transpose_tc_ct(float *dst, const float *src, int T, int C) {
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++)
            dst[(size_t)c * T + t] = src[(size_t)t * C + c];
}

void aria_sa3_dit_forward(const aria_sa3_dit *m, float *out_CT, const float *x_CT, int T,
                          float t, const float *cross_768, int n_cond, const float *global_768) {
    int C = m->io_ch, ed = m->ed, Mt = m->n_mem, S = Mt + T, rot = m->rot_dim;

    /* conditioning embeddings */
    float *cross_ed = malloc((size_t)n_cond * ed * sizeof(float));
    aria_sa3_mlp2(cross_ed, cross_768, n_cond, m->cond_dim, ed, ed, m->to_cond0, NULL, m->to_cond2, NULL);

    float *global_ed = malloc((size_t)ed * sizeof(float));
    aria_sa3_mlp2(global_ed, global_768, 1, m->cond_dim, ed, ed, m->to_global0, NULL, m->to_global2, NULL);
    float *ts = malloc((size_t)ed * sizeof(float));
    aria_sa3_timestep_embed(ts, t, m->ts_feat_dim, ed, m->to_ts0_w, m->to_ts0_b, m->to_ts2_w, m->to_ts2_b);
    for (int i = 0; i < ed; i++) global_ed[i] += ts[i];

    float *gcond = malloc((size_t)6 * ed * sizeof(float));
    aria_sa3_mlp2(gcond, global_ed, 1, ed, ed, 6 * ed, m->gce0_w, m->gce0_b, m->gce2_w, m->gce2_b);

    /* preprocess_conv(x) + x  (1x1 conv over channels), x: [C,T] -> [T,C] */
    float *xtc = malloc((size_t)T * C * sizeof(float));
    transpose_ct_tc(xtc, x_CT, C, T);
    float *pre = malloc((size_t)T * C * sizeof(float));
    aria_linear(pre, xtc, m->preprocess, NULL, T, C, C);
    for (size_t i = 0; i < (size_t)T * C; i++) xtc[i] += pre[i];
    free(pre);

    /* project_in [T,C] -> [T,ed], prepend memory tokens -> seq [S,ed] */
    float *seq = malloc((size_t)S * ed * sizeof(float));
    memcpy(seq, m->memory_tokens, (size_t)Mt * ed * sizeof(float));
    aria_linear(seq + (size_t)Mt * ed, xtc, m->project_in, NULL, T, C, ed);
    free(xtc);

    /* rope over the full sequence */
    float *cosb = malloc((size_t)S * (rot / 2) * sizeof(float));
    float *sinb = malloc((size_t)S * (rot / 2) * sizeof(float));
    aria_rope_freqs(cosb, sinb, S, rot, 10000.0f);

    for (int b = 0; b < m->depth; b++)
        aria_dit_block_forward(seq, S, ed, m->num_heads, m->head_dim, m->inner,
                               cross_ed, n_cond, ed, gcond, cosb, sinb, rot, &m->blocks[b]);

    /* strip memory tokens, project_out [T,ed] -> [T,C] */
    float *outtc = malloc((size_t)T * C * sizeof(float));
    aria_linear(outtc, seq + (size_t)Mt * ed, m->project_out, NULL, T, ed, C);

    /* postprocess_conv(out) + out */
    float *postc = malloc((size_t)T * C * sizeof(float));
    aria_linear(postc, outtc, m->postprocess, NULL, T, C, C);
    for (size_t i = 0; i < (size_t)T * C; i++) outtc[i] += postc[i];
    free(postc);

    transpose_tc_ct(out_CT, outtc, T, C);

    free(cross_ed); free(global_ed); free(ts); free(gcond);
    free(seq); free(cosb); free(sinb); free(outtc);
}
