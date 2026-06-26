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

#ifdef ARIA_BLAS
#include <cblas.h>   /* optional: route GEMM to OpenBLAS/Accelerate/MKL */
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define ARIA_AVX2 1
/* register-tile dimensions (output rows x cols held in ymm accumulators):
 * regs used = NR (weight vecs) + 1 (x vec) + MR*NR (accumulators) <= 16 */
#ifndef ARIA_MR
#define ARIA_MR 4
#endif
#ifndef ARIA_NR
#define ARIA_NR 3
#endif
static inline float aria_hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x1));
    return _mm_cvtss_f32(lo);
}
#endif

/* y[M,N] = x[M,K] @ W^T + b, W=[N,K] (PyTorch Linear).
 * Blocked over output columns (NB) so each weight panel (NB*K) stays in L2 and is
 * reused across all M rows; the inner dot is vectorized via omp-simd reduction
 * (the float reduction won't auto-vectorize under strict FP otherwise). The
 * vectorized summation order differs slightly from a scalar sweep -- within the
 * parity tolerances. */
/* Route only the big projection/FFN GEMMs to BLAS: multithreaded sgemm has
 * per-call thread-pool overhead that dwarfs the work for the per-head attention
 * matmuls (one of their dims is head_dim=64). Gate on the contracted/output dims
 * being large, not on total flops (attention grows with sequence length too). */
#define ARIA_BLAS_BIG(K, N) ((K) >= 128 && (N) >= 128)

/* simd dot for edge tiles */
static inline float aria_dot(const float *a, const float *w, int K) {
    float acc = 0.0f;
    #ifdef _OPENMP
    #pragma omp simd reduction(+:acc)
    #endif
    for (int k = 0; k < K; k++) acc += a[k] * w[k];
    return acc;
}

#ifdef ARIA_AVX2
/* MR x NR output tile, vectorized over K (8 lanes), MR*NR ymm accumulators for ILP.
 * y[(m+i),(n+j)] = dot(x[m+i], W[n+j]) + b[n+j], for i<MR, j<NR. */
static inline void aria_microkernel(float *y, const float *x, const float *W, const float *b,
                                    int m, int n, int K, int N) {
    __m256 acc[ARIA_MR][ARIA_NR];
    const float *xp[ARIA_MR], *wp[ARIA_NR];
    for (int i = 0; i < ARIA_MR; i++) { xp[i] = x + (size_t)(m + i) * K;
        for (int j = 0; j < ARIA_NR; j++) acc[i][j] = _mm256_setzero_ps(); }
    for (int j = 0; j < ARIA_NR; j++) wp[j] = W + (size_t)(n + j) * K;
    int k = 0;
    for (; k + 8 <= K; k += 8) {
        __m256 wv[ARIA_NR];
        for (int j = 0; j < ARIA_NR; j++) wv[j] = _mm256_loadu_ps(wp[j] + k);
        for (int i = 0; i < ARIA_MR; i++) {
            __m256 xv = _mm256_loadu_ps(xp[i] + k);
            for (int j = 0; j < ARIA_NR; j++) acc[i][j] = _mm256_fmadd_ps(xv, wv[j], acc[i][j]);
        }
    }
    for (int i = 0; i < ARIA_MR; i++)
        for (int j = 0; j < ARIA_NR; j++) {
            float s = aria_hsum256(acc[i][j]);
            for (int kk = k; kk < K; kk++) s += xp[i][kk] * wp[j][kk];  /* K%8 tail */
            if (b) s += b[n + j];
            y[(size_t)(m + i) * N + (n + j)] = s;
        }
}
#endif

void aria_linear(float *y, const float *x, const float *W, const float *b,
                 int M, int K, int N) {
#ifdef ARIA_BLAS
    if (ARIA_BLAS_BIG(K, N)) {
        /* y = x[M,K] @ W[N,K]^T  (W is PyTorch Linear weight [out,in]) */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K,
                    1.0f, x, K, W, K, 0.0f, y, N);
        if (b) {
            for (int m = 0; m < M; m++) {
                float *yr = y + (size_t)m * N;
                for (int n = 0; n < N; n++) yr[n] += b[n];
            }
        }
        return;
    }
#endif
#ifdef ARIA_AVX2
    {
        const int MR = ARIA_MR, NR = ARIA_NR;
        int Mfull = M - (M % MR);
        int Nfull = N - (N % NR);
        int NBN = ((48 + NR - 1) / NR) * NR;       /* ~L2 weight panel, multiple of NR */
        int n_chunks = (Nfull > 0) ? (Nfull + NBN - 1) / NBN : 0;
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (int c = 0; c < n_chunks; c++) {
            int cn0 = c * NBN;
            int cn1 = cn0 + NBN < Nfull ? cn0 + NBN : Nfull;
            for (int mm = 0; mm < Mfull; mm += MR)
                for (int nn = cn0; nn < cn1; nn += NR)
                    aria_microkernel(y, x, W, b, mm, nn, K, N);
        }
        /* N-edge: cols [Nfull,N) for rows [0,Mfull) */
        for (int m = 0; m < Mfull; m++)
            for (int n = Nfull; n < N; n++)
                y[(size_t)m * N + n] = aria_dot(x + (size_t)m * K, W + (size_t)n * K, K) + (b ? b[n] : 0.0f);
        /* M-edge: rows [Mfull,M) for all cols */
        for (int m = Mfull; m < M; m++)
            for (int n = 0; n < N; n++)
                y[(size_t)m * N + n] = aria_dot(x + (size_t)m * K, W + (size_t)n * K, K) + (b ? b[n] : 0.0f);
        return;
    }
#else
    const int NB = 32;                 /* NB*K*4 bytes ~ L2-resident weight panel */
    int n_tiles = (N + NB - 1) / NB;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int tile = 0; tile < n_tiles; tile++) {
        int n0 = tile * NB;
        int n1 = (n0 + NB < N) ? n0 + NB : N;
        for (int m = 0; m < M; m++) {
            const float *xr = x + (size_t)m * K;
            float *yr = y + (size_t)m * N;
            for (int n = n0; n < n1; n++) {
                float acc = b ? b[n] : 0.0f;
                acc += aria_dot(xr, W + (size_t)n * K, K);
                yr[n] = acc;
            }
        }
    }
#endif
}

void aria_matmul(float *C, const float *A, const float *B, int M, int K, int N) {
#ifdef ARIA_BLAS
    if (ARIA_BLAS_BIG(K, N)) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                    1.0f, A, K, B, N, 0.0f, C, N);
        return;
    }
#endif
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

void aria_gemma_rmsnorm(float *y, const float *x, const float *weight,
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
        for (int i = 0; i < dim; i++) yr[i] = xr[i] * inv * (1.0f + weight[i]);
    }
}

void aria_softcap(float *s, int n, float cap) {
    for (int i = 0; i < n; i++) s[i] = cap * tanhf(s[i] / cap);
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
