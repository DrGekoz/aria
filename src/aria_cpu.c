/*
 * aria_cpu.c - CPU implementations of the aria_ops primitive surface.
 *
 * Scalar + compiler-autovectorized (-O3 -march=native), OpenMP across rows.
 * SIMD-intrinsic hot paths land in Phase 3; correctness comes first.
 */

#include "aria_ops.h"
#include <math.h>
#include <string.h>

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
