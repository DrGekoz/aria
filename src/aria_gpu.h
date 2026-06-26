/*
 * aria_gpu.h - CUDA backend surface (E8).
 *
 * Built only under -DARIA_CUDA (aria_cuda.cu). Device-pointer kernels are the
 * building blocks for an eventual device-resident forward (E8.5); the host-pointer
 * wrappers (allocate + copy + compute + copy back) are for op-level parity tests
 * and standalone use. All symbols are C-callable.
 */

#ifndef ARIA_GPU_H
#define ARIA_GPU_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 1 if a usable CUDA device is present (0 in a CPU-only build). */
int aria_cuda_available(void);
/* short device description into buf (name, compute capability, VRAM). */
void aria_cuda_device_info(char *buf, size_t buflen);

/* raw device memory */
void *aria_cuda_malloc(size_t bytes);
void  aria_cuda_free(void *dptr);
void  aria_cuda_memcpy_h2d(void *ddst, const void *hsrc, size_t bytes);
void  aria_cuda_memcpy_d2h(void *hdst, const void *dsrc, size_t bytes);
void  aria_cuda_sync(void);

/* y[M,N] = x[M,K] @ W[N,K]^T + b   (W is PyTorch Linear layout; b may be NULL).
 * _dev: all pointers are device memory. host wrapper: all pointers are host. */
void aria_cuda_linear_dev(float *dy, const float *dx, const float *dW, const float *db,
                          int M, int K, int N);
void aria_cuda_linear(float *y, const float *x, const float *W, const float *b,
                      int M, int K, int N);

#ifdef __cplusplus
}
#endif

#endif /* ARIA_GPU_H */
