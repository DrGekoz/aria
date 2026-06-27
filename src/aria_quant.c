/*
 * aria_quant.c - weight quantization + dequant-on-use GEMM (CPU). See aria_quant.h.
 * Correctness-first scalar kernels (SIMD is a later optimization); the GEMM shape
 * mirrors aria_linear so the model can dispatch on dtype with no other change.
 */

#include "aria_quant.h"
#include "aria_ops.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

const char *aria_dtype_name(aria_dtype dt) {
    switch (dt) {
        case ARIA_F32:  return "fp32";
        case ARIA_F16:  return "fp16";
        case ARIA_BF16: return "bf16";
        case ARIA_Q8:   return "q8";
        case ARIA_Q4:   return "q4";
        default:        return "?";
    }
}

int aria_dtype_parse(const char *s, aria_dtype *out) {
    if (!s) return -1;
    if (!strcmp(s, "fp32") || !strcmp(s, "f32") || !strcmp(s, "float32")) *out = ARIA_F32;
    else if (!strcmp(s, "fp16") || !strcmp(s, "f16") || !strcmp(s, "half")) *out = ARIA_F16;
    else if (!strcmp(s, "bf16") || !strcmp(s, "bfloat16")) *out = ARIA_BF16;
    else if (!strcmp(s, "q8") || !strcmp(s, "int8")) *out = ARIA_Q8;
    else if (!strcmp(s, "q4") || !strcmp(s, "int4")) *out = ARIA_Q4;
    else return -1;
    return 0;
}

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* ---- Q8 ---- */

void aria_q8_quant(int8_t *q, float *scale, const float *W, int N, int K) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int n = 0; n < N; n++) {
        const float *wr = W + (size_t)n * K;
        float amax = 0.0f;
        for (int k = 0; k < K; k++) { float a = fabsf(wr[k]); if (a > amax) amax = a; }
        float s = amax / 127.0f;
        scale[n] = s;
        int8_t *qr = q + (size_t)n * K;
        if (s > 0.0f) {
            float inv = 1.0f / s;
            for (int k = 0; k < K; k++) qr[k] = (int8_t)clampi((int)lrintf(wr[k] * inv), -127, 127);
        } else {
            memset(qr, 0, (size_t)K);
        }
    }
}

void aria_linear_q8(float *y, const float *x, const int8_t *q, const float *scale,
                    const float *b, int M, int K, int N) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int m = 0; m < M; m++) {
        const float *xr = x + (size_t)m * K;
        float *yr = y + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            const int8_t *qn = q + (size_t)n * K;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += xr[k] * (float)qn[k];
            yr[n] = acc * scale[n] + (b ? b[n] : 0.0f);
        }
    }
}

/* ---- Q4 (block symmetric, 2 nibbles/byte, stored as v+8 in [1,15]) ---- */

void aria_q4_quant(uint8_t *q, float *scale, const float *W, int N, int K) {
    int nblk = aria_q4_nblocks(K);
    size_t rb = aria_q4_rowbytes(K);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int n = 0; n < N; n++) {
        const float *wr = W + (size_t)n * K;
        uint8_t *qr = q + (size_t)n * rb;
        float *sr = scale + (size_t)n * nblk;
        memset(qr, 0, rb);
        for (int bi = 0; bi < nblk; bi++) {
            int k0 = bi * ARIA_Q4_BLOCK;
            int k1 = k0 + ARIA_Q4_BLOCK; if (k1 > K) k1 = K;
            float amax = 0.0f;
            for (int k = k0; k < k1; k++) { float a = fabsf(wr[k]); if (a > amax) amax = a; }
            float s = amax / 7.0f;
            sr[bi] = s;
            float inv = (s > 0.0f) ? 1.0f / s : 0.0f;
            for (int k = k0; k < k1; k++) {
                int v = (s > 0.0f) ? clampi((int)lrintf(wr[k] * inv), -7, 7) : 0;
                uint8_t nib = (uint8_t)(v + 8);                 /* [1,15] */
                if (k & 1) qr[k >> 1] |= (uint8_t)(nib << 4);
                else       qr[k >> 1] |= nib;
            }
        }
    }
}

/* ---- aria_qweight dispatch ---- */

void aria_qweight_set(aria_qweight *w, const float *W, int N, int K, aria_dtype dt) {
    w->dt = dt; w->N = N; w->K = K; w->f32 = NULL; w->q = NULL; w->scale = NULL;
    if (dt == ARIA_Q8) {
        w->q = malloc(aria_q8_qbytes(N, K));
        w->scale = malloc(aria_q8_nscale(N, K) * sizeof(float));
        aria_q8_quant((int8_t *)w->q, w->scale, W, N, K);
    } else if (dt == ARIA_Q4) {
        w->q = malloc(aria_q4_qbytes(N, K));
        w->scale = malloc(aria_q4_nscale(N, K) * sizeof(float));
        aria_q4_quant((uint8_t *)w->q, w->scale, W, N, K);
    } else {
        w->dt = ARIA_F32;   /* fp16/bf16 not yet a CPU storage format -> borrow f32 */
        w->f32 = W;
    }
}

void aria_qweight_free(aria_qweight *w) {
    if (!w) return;
    free(w->q); free(w->scale);
    w->q = NULL; w->scale = NULL; w->f32 = NULL;
}

void aria_linear_qw(float *y, const float *x, const aria_qweight *w, const float *b, int M) {
    if (w->dt == ARIA_Q8)      aria_linear_q8(y, x, (const int8_t *)w->q, w->scale, b, M, w->K, w->N);
    else if (w->dt == ARIA_Q4) aria_linear_q4(y, x, (const uint8_t *)w->q, w->scale, b, M, w->K, w->N);
    else                       aria_linear(y, x, w->f32, b, M, w->K, w->N);
}

size_t aria_qweight_bytes(const aria_qweight *w) {
    if (w->dt == ARIA_Q8) return aria_q8_qbytes(w->N, w->K) + aria_q8_nscale(w->N, w->K) * sizeof(float);
    if (w->dt == ARIA_Q4) return aria_q4_qbytes(w->N, w->K) + aria_q4_nscale(w->N, w->K) * sizeof(float);
    return (size_t)w->N * w->K * sizeof(float);
}

void aria_linear_q4(float *y, const float *x, const uint8_t *q, const float *scale,
                    const float *b, int M, int K, int N) {
    int nblk = aria_q4_nblocks(K);
    size_t rb = aria_q4_rowbytes(K);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int m = 0; m < M; m++) {
        const float *xr = x + (size_t)m * K;
        float *yr = y + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            const uint8_t *qn = q + (size_t)n * rb;
            const float *sn = scale + (size_t)n * nblk;
            float acc = 0.0f;
            for (int bi = 0; bi < nblk; bi++) {
                int k0 = bi * ARIA_Q4_BLOCK;
                int k1 = k0 + ARIA_Q4_BLOCK; if (k1 > K) k1 = K;
                float bacc = 0.0f;
                for (int k = k0; k < k1; k++) {
                    uint8_t byte = qn[k >> 1];
                    int nib = (k & 1) ? (byte >> 4) : (byte & 0x0F);
                    bacc += xr[k] * (float)(nib - 8);
                }
                acc += bacc * sn[bi];
            }
            yr[n] = acc + (b ? b[n] : 0.0f);
        }
    }
}
