/*
 * aria_cpu.c - CPU implementations of the aria_ops primitive surface.
 *
 * Scalar + compiler-autovectorized (-O3 -march=native), OpenMP across rows.
 * SIMD-intrinsic hot paths land in Phase 3; correctness comes first.
 */

#include "aria_ops.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void aria_linear(float *y, const float *x, const float *W, const float *b,
                 int M, int K, int N) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int m = 0; m < M; m++) {
        const float *xr = x + (size_t)m * K;
        float *yr = y + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            const float *wr = W + (size_t)n * K;
            float acc = b ? b[n] : 0.0f;
            for (int k = 0; k < K; k++) acc += xr[k] * wr[k];
            yr[n] = acc;
        }
    }
}

void aria_matmul(float *C, const float *A, const float *B, int M, int K, int N) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int m = 0; m < M; m++) {
        const float *ar = A + (size_t)m * K;
        float *cr = C + (size_t)m * N;
        for (int n = 0; n < N; n++) cr[n] = 0.0f;
        for (int k = 0; k < K; k++) {
            float a = ar[k];
            const float *br = B + (size_t)k * N;
            for (int n = 0; n < N; n++) cr[n] += a * br[n];
        }
    }
}

void aria_rmsnorm(float *y, const float *x, const float *weight,
                  int rows, int dim, float eps) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * dim;
        float *yr = y + (size_t)r * dim;
        double ss = 0.0;
        for (int i = 0; i < dim; i++) ss += (double)xr[i] * xr[i];
        float inv = (float)(1.0 / sqrt(ss / dim + eps));
        if (weight) for (int i = 0; i < dim; i++) yr[i] = xr[i] * inv * weight[i];
        else        for (int i = 0; i < dim; i++) yr[i] = xr[i] * inv;
    }
}

void aria_dynamic_tanh(float *y, const float *x, float alpha,
                       const float *weight, const float *bias,
                       int rows, int dim) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * dim;
        float *yr = y + (size_t)r * dim;
        for (int i = 0; i < dim; i++) {
            float t = tanhf(alpha * xr[i]);
            yr[i] = t * (weight ? weight[i] : 1.0f) + (bias ? bias[i] : 0.0f);
        }
    }
}

void aria_silu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

void aria_gelu_tanh(float *x, int n) {
    const float c = 0.7978845608028654f; /* sqrt(2/pi) */
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float inner = c * (v + 0.044715f * v * v * v);
        x[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

void aria_silu_gate(float *out, const float *gate, const float *up, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

void aria_softmax_inplace(float *x, int rows, int cols, const float *mask) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int r = 0; r < rows; r++) {
        float *xr = x + (size_t)r * cols;
        const float *mr = mask ? mask + (size_t)r * cols : NULL;
        float maxv = -INFINITY;
        for (int i = 0; i < cols; i++) {
            float v = xr[i] + (mr ? mr[i] : 0.0f);
            xr[i] = v;
            if (v > maxv) maxv = v;
        }
        float sum = 0.0f;
        for (int i = 0; i < cols; i++) { float e = expf(xr[i] - maxv); xr[i] = e; sum += e; }
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int i = 0; i < cols; i++) xr[i] *= inv;
    }
}

void aria_rope_freqs(float *cos_t, float *sin_t, int n, int rot_dim, float base) {
    int rh = rot_dim / 2;
    for (int i = 0; i < rh; i++) {
        double inv = 1.0 / pow((double)base, (double)(2 * i) / (double)rot_dim);
        for (int p = 0; p < n; p++) {
            double ang = (double)p * inv;
            cos_t[(size_t)p * rh + i] = (float)cos(ang);
            sin_t[(size_t)p * rh + i] = (float)sin(ang);
        }
    }
}

void aria_rope_apply(float *x, const float *cos_t, const float *sin_t,
                     int H, int N, int D, int rot_dim) {
    int rh = rot_dim / 2;
    for (int h = 0; h < H; h++) {
        for (int p = 0; p < N; p++) {
            float *row = x + ((size_t)h * N + p) * D;
            const float *c = cos_t + (size_t)p * rh;
            const float *s = sin_t + (size_t)p * rh;
            for (int i = 0; i < rh; i++) {
                float a = row[i], b = row[i + rh];
                row[i]      = a * c[i] - b * s[i];
                row[i + rh] = b * c[i] + a * s[i];
            }
            /* dims >= rot_dim left unchanged */
        }
    }
}

void aria_attention(float *out, const float *q, const float *k, const float *v,
                    int H, int Nq, int Nk, int D, const float *mask) {
    float scale = 1.0f / sqrtf((float)D);
    float *scores = malloc((size_t)Nq * Nk * sizeof(float));
    if (!scores) return;
    for (int h = 0; h < H; h++) {
        const float *qh = q + (size_t)h * Nq * D;
        const float *kh = k + (size_t)h * Nk * D;
        const float *vh = v + (size_t)h * Nk * D;
        float *oh = out + (size_t)h * Nq * D;
        /* scores[i,j] = qh[i] . kh[j]  (aria_linear computes x @ W^T) */
        aria_linear(scores, qh, kh, NULL, Nq, D, Nk);
        for (size_t i = 0; i < (size_t)Nq * Nk; i++) scores[i] *= scale;
        aria_softmax_inplace(scores, Nq, Nk, mask);
        aria_matmul(oh, scores, vh, Nq, Nk, D);
    }
    free(scores);
}

void aria_conv1d(float *out, const float *in, const float *w, const float *bias,
                 int Cin, int Cout, int K, int pad, int L) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int o = 0; o < Cout; o++) {
        const float *wo = w + (size_t)o * Cin * K;
        float *outo = out + (size_t)o * L;
        for (int t = 0; t < L; t++) {
            float acc = bias ? bias[o] : 0.0f;
            for (int i = 0; i < Cin; i++) {
                const float *wi = wo + (size_t)i * K;
                const float *ini = in + (size_t)i * L;
                for (int k = 0; k < K; k++) {
                    int ti = t + k - pad;
                    if (ti >= 0 && ti < L) acc += wi[k] * ini[ti];
                }
            }
            outo[t] = acc;
        }
    }
}

void aria_ff_glu(float *out, const float *x, int N, int dim, int inner, int dim_out,
                 const float *W_in, const float *b_in,
                 const float *W_out, const float *b_out) {
    float *proj = malloc((size_t)N * 2 * inner * sizeof(float));
    float *gated = malloc((size_t)N * inner * sizeof(float));
    if (!proj || !gated) { free(proj); free(gated); return; }
    aria_linear(proj, x, W_in, b_in, N, dim, 2 * inner);
    for (int n = 0; n < N; n++) {
        const float *pr = proj + (size_t)n * 2 * inner;
        /* GLU: value = first half, gate = second half; out = value * silu(gate) */
        aria_silu_gate(gated + (size_t)n * inner, pr + inner, pr, inner);
    }
    aria_linear(out, gated, W_out, b_out, N, inner, dim_out);
    free(proj);
    free(gated);
}

void aria_add(float *y, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) y[i] = a[i] + b[i];
}
void aria_addto(float *y, const float *a, int n) {
    for (int i = 0; i < n; i++) y[i] += a[i];
}
void aria_scale(float *y, const float *x, float s, int n) {
    for (int i = 0; i < n; i++) y[i] = x[i] * s;
}
void aria_axpy(float *y, const float *x, float s, int n) {
    for (int i = 0; i < n; i++) y[i] += s * x[i];
}
