/*
 * aria_quant.h - weight precision/quantization: dtype tags + pack/unpack +
 * dequant-on-use GEMM. The foundation of E9 (precision & quantization).
 *
 * Weights follow PyTorch's nn.Linear layout [N,K] = [out,in]; aria_linear and
 * its quantized variants all compute y[M,N] = x[M,K] @ W^T + b. Quantization is
 * symmetric (zero-point 0) and dequant happens inside the GEMM (ds4 style), so
 * the packed weights stay small in cache.
 *
 *   Q8  per-row int8:  scale[n] = max|W[n,:]|/127;  W ~= q*scale[n]
 *   Q4  per-block int4: scale[n,blk] = max|blk|/7;  W ~= (nib-8)*scale  (block 32)
 */

#ifndef ARIA_QUANT_H
#define ARIA_QUANT_H

#include <stdint.h>
#include <stddef.h>

typedef enum {
    ARIA_F32  = 0,   /* float32 (zero-copy from the mmap; the default) */
    ARIA_F16  = 1,   /* float16 storage, fp32 compute (E9.0 wiring) */
    ARIA_BF16 = 2,   /* bfloat16 storage, fp32 compute (E9.0 wiring) */
    ARIA_Q8   = 3,   /* int8, per-row symmetric */
    ARIA_Q4   = 4,   /* int4, per-block symmetric */
} aria_dtype;

const char *aria_dtype_name(aria_dtype dt);
/* parse "fp32"/"f32"/"fp16"/"f16"/"bf16"/"q8"/"q4" -> *out; returns 0 on success. */
int aria_dtype_parse(const char *s, aria_dtype *out);

/* ---- Q8: per-row symmetric int8 ---- */
/* quantize W[N,K] (row-major) -> q[N*K] int8 + scale[N] (caller-allocated). */
void aria_q8_quant(int8_t *q, float *scale, const float *W, int N, int K);
/* dequant-on-use GEMM: y[M,N] = x[M,K] @ dequant(q,scale)^T + b. b may be NULL. */
void aria_linear_q8(float *y, const float *x, const int8_t *q, const float *scale,
                    const float *b, int M, int K, int N);

/* ---- Q4: per-block symmetric int4 (block = ARIA_Q4_BLOCK), 2 nibbles/byte ---- */
#define ARIA_Q4_BLOCK 32
static inline int    aria_q4_nblocks(int K) { return (K + ARIA_Q4_BLOCK - 1) / ARIA_Q4_BLOCK; }
static inline size_t aria_q4_rowbytes(int K) { return (size_t)((K + 1) / 2); }
/* quantize W[N,K] -> packed q[N*ceil(K/2)] + scale[N*nblocks] (caller-allocated). */
void aria_q4_quant(uint8_t *q, float *scale, const float *W, int N, int K);
void aria_linear_q4(float *y, const float *x, const uint8_t *q, const float *scale,
                    const float *b, int M, int K, int N);

/* packed-size helpers (bytes / floats) for allocating a quantized [N,K] weight. */
static inline size_t aria_q8_qbytes(int N, int K)   { return (size_t)N * K; }
static inline size_t aria_q8_nscale(int N, int K)   { (void)K; return (size_t)N; }
static inline size_t aria_q4_qbytes(int N, int K)   { return (size_t)N * aria_q4_rowbytes(K); }
static inline size_t aria_q4_nscale(int N, int K)   { return (size_t)N * aria_q4_nblocks(K); }

#endif /* ARIA_QUANT_H */
