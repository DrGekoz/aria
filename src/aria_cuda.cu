/*
 * aria_cuda.cu - CUDA backend (E8). Built with nvcc under -DARIA_CUDA.
 *
 * Correctness-first: a shared-memory tiled GEMM for the linear op, plus device
 * memory helpers. fp32 throughout (sm_61/Pascal has no fast fp16 compute; fp16
 * compute for sm_86 lands later). The kernel operates on device pointers so the
 * eventual device-resident forward (E8.5) can drop the host<->device copies.
 */

#include "aria_gpu.h"
#include <cuda_runtime.h>
#include <cstdio>

#define CK(call) do {                                                        \
    cudaError_t err_ = (call);                                               \
    if (err_ != cudaSuccess)                                                 \
        fprintf(stderr, "aria_cuda: %s -> %s\n", #call, cudaGetErrorString(err_)); \
} while (0)

extern "C" int aria_cuda_available(void) {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    return (e == cudaSuccess && n > 0) ? 1 : 0;
}

extern "C" void aria_cuda_device_info(char *buf, size_t buflen) {
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) {
        snprintf(buf, buflen, "no CUDA device");
        return;
    }
    snprintf(buf, buflen, "%s (sm_%d%d, %zu MiB)", p.name, p.major, p.minor,
             (size_t)(p.totalGlobalMem >> 20));
}

extern "C" void *aria_cuda_malloc(size_t bytes) {
    void *p = NULL;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return NULL;
    return p;
}
extern "C" void aria_cuda_free(void *dptr) { if (dptr) cudaFree(dptr); }
extern "C" void aria_cuda_memcpy_h2d(void *d, const void *h, size_t n) {
    CK(cudaMemcpy(d, h, n, cudaMemcpyHostToDevice));
}
extern "C" void aria_cuda_memcpy_d2h(void *h, const void *d, size_t n) {
    CK(cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost));
}
extern "C" void aria_cuda_sync(void) { CK(cudaDeviceSynchronize()); }

/* ---- tiled GEMM: y[M,N] = x[M,K] @ W[N,K]^T + b ----
 * x and W are both row-major with K contiguous, so y[m,n] = dot(x[m,:], W[n,:]).
 * The W tile is loaded transposed into shared memory so the inner product reads
 * contiguous shared rows. TILE x TILE threads per block, one output each. */
#define TILE 16

__global__ void aria_gemm_nt(float *__restrict__ y, const float *__restrict__ x,
                             const float *__restrict__ W, const float *__restrict__ b,
                             int M, int K, int N) {
    __shared__ float xs[TILE][TILE];
    __shared__ float ws[TILE][TILE];
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty;   /* m */
    int col = blockIdx.x * TILE + tx;   /* n */
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        int kx = k0 + tx, ky = k0 + ty;
        xs[ty][tx] = (row < M && kx < K) ? x[(size_t)row * K + kx] : 0.0f;
        /* ws[ty][tx] = W[col, k0+ty]  (col = blockIdx.x*TILE + tx) */
        ws[ty][tx] = (col < N && ky < K) ? W[(size_t)col * K + ky] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE; i++) acc += xs[ty][i] * ws[i][tx];
        __syncthreads();
    }
    if (row < M && col < N) y[(size_t)row * N + col] = acc + (b ? b[col] : 0.0f);
}

extern "C" void aria_cuda_linear_dev(float *dy, const float *dx, const float *dW,
                                     const float *db, int M, int K, int N) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    aria_gemm_nt<<<grid, block>>>(dy, dx, dW, db, M, K, N);
}

extern "C" void aria_cuda_linear(float *y, const float *x, const float *W,
                                 const float *b, int M, int K, int N) {
    float *dx = NULL, *dW = NULL, *dy = NULL, *db = NULL;
    CK(cudaMalloc(&dx, (size_t)M * K * sizeof(float)));
    CK(cudaMalloc(&dW, (size_t)N * K * sizeof(float)));
    CK(cudaMalloc(&dy, (size_t)M * N * sizeof(float)));
    CK(cudaMemcpy(dx, x, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dW, W, (size_t)N * K * sizeof(float), cudaMemcpyHostToDevice));
    if (b) {
        CK(cudaMalloc(&db, (size_t)N * sizeof(float)));
        CK(cudaMemcpy(db, b, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
    }
    aria_cuda_linear_dev(dy, dx, dW, db, M, K, N);
    CK(cudaGetLastError());
    CK(cudaMemcpy(y, dy, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dx); cudaFree(dW); cudaFree(dy); if (db) cudaFree(db);
}
