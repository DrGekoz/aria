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

/* ---- Q4 (block asymmetric / zero-point; 2 nibbles/byte; per block [min,scale]) ----
 * nibble in [0,15]; W ~= min + nib*scale, scale = (max-min)/15. Affine quant fits
 * skewed weight blocks far better than symmetric int4. */

void aria_q4_quant(uint8_t *q, float *scale, const float *W, int N, int K) {
    int nblk = aria_q4_nblocks(K);
    size_t rb = aria_q4_rowbytes(K);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int n = 0; n < N; n++) {
        const float *wr = W + (size_t)n * K;
        uint8_t *qr = q + (size_t)n * rb;
        float *sr = scale + (size_t)n * 2 * nblk;   /* [min,scale] per block */
        memset(qr, 0, rb);
        for (int bi = 0; bi < nblk; bi++) {
            int k0 = bi * ARIA_Q4_BLOCK;
            int k1 = k0 + ARIA_Q4_BLOCK; if (k1 > K) k1 = K;
            float mn = wr[k0], mx = wr[k0];
            for (int k = k0 + 1; k < k1; k++) { float v = wr[k]; if (v < mn) mn = v; if (v > mx) mx = v; }
            float s = (mx - mn) / 15.0f;
            sr[2 * bi] = mn; sr[2 * bi + 1] = s;
            float inv = (s > 0.0f) ? 1.0f / s : 0.0f;
            for (int k = k0; k < k1; k++) {
                uint8_t nib = (s > 0.0f) ? (uint8_t)clampi((int)lrintf((wr[k] - mn) * inv), 0, 15) : 0;
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

int aria_qweight_write(FILE *f, const aria_qweight *w) {
    if (w->dt != ARIA_Q8 && w->dt != ARIA_Q4) return -1;
    int32_t hdr[3] = { (int32_t)w->dt, w->N, w->K };
    size_t qb = (w->dt == ARIA_Q8) ? aria_q8_qbytes(w->N, w->K) : aria_q4_qbytes(w->N, w->K);
    size_t ns = (w->dt == ARIA_Q8) ? aria_q8_nscale(w->N, w->K) : aria_q4_nscale(w->N, w->K);
    uint64_t qb64 = qb, ns64 = ns;
    if (fwrite(hdr, sizeof(int32_t), 3, f) != 3) return -1;
    if (fwrite(&qb64, sizeof(uint64_t), 1, f) != 1) return -1;
    if (fwrite(w->q, 1, qb, f) != qb) return -1;
    if (fwrite(&ns64, sizeof(uint64_t), 1, f) != 1) return -1;
    if (fwrite(w->scale, sizeof(float), ns, f) != ns) return -1;
    return 0;
}

int aria_qweight_read(FILE *f, aria_qweight *w) {
    int32_t hdr[3];
    if (fread(hdr, sizeof(int32_t), 3, f) != 3) return -1;
    w->dt = (aria_dtype)hdr[0]; w->N = hdr[1]; w->K = hdr[2]; w->f32 = NULL; w->q = NULL; w->scale = NULL;
    if (w->dt != ARIA_Q8 && w->dt != ARIA_Q4) return -1;
    uint64_t qb64, ns64;
    if (fread(&qb64, sizeof(uint64_t), 1, f) != 1) return -1;
    w->q = malloc(qb64);
    if (!w->q || fread(w->q, 1, qb64, f) != qb64) return -1;
    if (fread(&ns64, sizeof(uint64_t), 1, f) != 1) return -1;
    w->scale = (float *)malloc(ns64 * sizeof(float));
    if (!w->scale || fread(w->scale, sizeof(float), ns64, f) != ns64) return -1;
    return 0;
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
            const float *sn = scale + (size_t)n * 2 * nblk;   /* [min,scale] per block */
            float acc = 0.0f;
            for (int bi = 0; bi < nblk; bi++) {
                int k0 = bi * ARIA_Q4_BLOCK;
                int k1 = k0 + ARIA_Q4_BLOCK; if (k1 > K) k1 = K;
                float mn = sn[2 * bi], sc = sn[2 * bi + 1];
                /* sum_k x*(min + nib*scale) = min*sum(x) + scale*sum(nib*x) */
                float xsum = 0.0f, nxsum = 0.0f;
                for (int k = k0; k < k1; k++) {
                    uint8_t byte = qn[k >> 1];
                    int nib = (k & 1) ? (byte >> 4) : (byte & 0x0F);
                    xsum += xr[k];
                    nxsum += xr[k] * (float)nib;
                }
                acc += mn * xsum + sc * nxsum;
            }
            yr[n] = acc + (b ? b[n] : 0.0f);
        }
    }
}
