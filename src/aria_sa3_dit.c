/*
 * aria_sa3_dit.c - SA3 conditioning embedders + DiT transformer block (CPU).
 */

#include "aria_sa3_dit.h"
#include "aria_ops.h"
#include "aria_cond.h"
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
