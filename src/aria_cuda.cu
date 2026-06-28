/*
 * aria_cuda.cu - CUDA backend (E8). Built with nvcc under -DARIA_CUDA.
 *
 * Correctness-first: a shared-memory tiled GEMM for the linear op, plus device
 * memory helpers. fp32 throughout (sm_61/Pascal has no fast fp16 compute; fp16
 * compute for sm_86 lands later). The kernel operates on device pointers so the
 * eventual device-resident forward (E8.5) can drop the host<->device copies.
 */

#include "aria_gpu.h"
#include "aria_sa3_dit.h"
#include "aria_sa3_dec.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

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

extern "C" int aria_cuda_recommended(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) return 0;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return 0;
    return p.major >= 7 ? 1 : 0;   /* Volta+ (tensor cores) — faster than the AVX2 CPU */
}

extern "C" void aria_cuda_meminfo(size_t *used, size_t *total) {
    size_t freeb = 0, totb = 0;
    if (cudaMemGetInfo(&freeb, &totb) != cudaSuccess) { *used = 0; *total = 0; return; }
    *used = totb - freeb; *total = totb;
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

/* ============================ remaining hot ops ============================
 * Correctness-first kernels (thread-per-output, reductions in-thread; norms use
 * fp64 accumulation to match the CPU path). Host wrappers copy-compute-copy. */

#define THREADS 256
static int nblocks(size_t n) { return (int)((n + THREADS - 1) / THREADS); }

/* device-memory helpers for the host wrappers */
static float *d_in(const float *h, size_t n) {
    float *d = NULL; CK(cudaMalloc(&d, n * sizeof(float)));
    CK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice)); return d;
}
static float *d_new(size_t n) { float *d = NULL; CK(cudaMalloc(&d, n * sizeof(float))); return d; }
static void d_out(float *h, float *d, size_t n) {
    CK(cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost)); cudaFree(d);
}

/* ---- matmul C[M,N] = A[M,K] @ B[K,N] (B not transposed) ---- */
__global__ void k_matmul(float *C, const float *A, const float *B, int M, int K, int N) {
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        As[ty][tx] = (row < M && k0 + tx < K) ? A[(size_t)row * K + k0 + tx] : 0.0f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(size_t)(k0 + ty) * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE; i++) acc += As[ty][i] * Bs[i][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}
static void matmul_dev(float *dC, const float *dA, const float *dB, int M, int K, int N) {
    dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    k_matmul<<<grid, block>>>(dC, dA, dB, M, K, N);
}
extern "C" void aria_cuda_matmul(float *C, const float *A, const float *B, int M, int K, int N) {
    float *dA = d_in(A, (size_t)M * K), *dB = d_in(B, (size_t)K * N), *dC = d_new((size_t)M * N);
    matmul_dev(dC, dA, dB, M, K, N); CK(cudaGetLastError());
    cudaFree(dA); cudaFree(dB); d_out(C, dC, (size_t)M * N);
}

/* ---- rmsnorm / gemma_rmsnorm: one block per row, fp32 shared-mem reduction ----
 * (the old thread-per-row + fp64 accum was 51% of GPU time; fp64 is 1/64 rate on
 * Ampere and 1 thread/row starves the device. fp32 is within the fp16 tolerance.) */
#define RED_TH 256
__global__ void k_rmsnorm(float *y, const float *x, const float *w, int rows, int dim, float eps, int gemma) {
    int r = blockIdx.x; if (r >= rows) return;
    const float *xr = x + (size_t)r * dim; float *yr = y + (size_t)r * dim;
    __shared__ float red[RED_TH];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += xr[i] * xr[i];
    red[threadIdx.x] = ss; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float g = w ? (gemma ? 1.0f + w[i] : w[i]) : 1.0f;
        yr[i] = xr[i] * inv * g;
    }
}
/* fused rmsnorm (weight w) + adaLN: y = (x * rsqrt(meansq+eps) * w) * (1+scale) + shift.
 * Replaces a k_rmsnorm + k_adaln pair (saves one read+write pass of [S,dim] + a launch). */
__global__ void k_rmsnorm_adaln(float *y, const float *x, const float *w,
                                const float *scale, const float *shift, int S, int dim, float eps) {
    int r = blockIdx.x; if (r >= S) return;
    const float *xr = x + (size_t)r * dim; float *yr = y + (size_t)r * dim;
    __shared__ float red[RED_TH];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += xr[i] * xr[i];
    red[threadIdx.x] = ss; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        yr[i] = (xr[i] * inv * w[i]) * (1.0f + scale[i]) + shift[i];
}

extern "C" void aria_cuda_rmsnorm(float *y, const float *x, const float *w, int rows, int dim, float eps) {
    float *dx = d_in(x, (size_t)rows * dim), *dw = w ? d_in(w, dim) : NULL, *dy = d_new((size_t)rows * dim);
    k_rmsnorm<<<rows, RED_TH>>>(dy, dx, dw, rows, dim, eps, 0); CK(cudaGetLastError());
    cudaFree(dx); if (dw) cudaFree(dw); d_out(y, dy, (size_t)rows * dim);
}
extern "C" void aria_cuda_gemma_rmsnorm(float *y, const float *x, const float *w, int rows, int dim, float eps) {
    float *dx = d_in(x, (size_t)rows * dim), *dw = d_in(w, dim), *dy = d_new((size_t)rows * dim);
    k_rmsnorm<<<rows, RED_TH>>>(dy, dx, dw, rows, dim, eps, 1); CK(cudaGetLastError());
    cudaFree(dx); cudaFree(dw); d_out(y, dy, (size_t)rows * dim);
}

/* ---- dynamic tanh: y = tanh(alpha*x)*w[c] + b[c] ---- */
__global__ void k_dyt(float *y, const float *x, float alpha, const float *w, const float *b, size_t tot, int dim) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tot) return;
    int c = (int)(i % dim);
    y[i] = tanhf(alpha * x[i]) * w[c] + b[c];
}
extern "C" void aria_cuda_dynamic_tanh(float *y, const float *x, float alpha,
                                       const float *w, const float *b, int rows, int dim) {
    size_t tot = (size_t)rows * dim;
    float *dx = d_in(x, tot), *dw = d_in(w, dim), *db = d_in(b, dim), *dy = d_new(tot);
    k_dyt<<<nblocks(tot), THREADS>>>(dy, dx, alpha, dw, db, tot, dim); CK(cudaGetLastError());
    cudaFree(dx); cudaFree(dw); cudaFree(db); d_out(y, dy, tot);
}

/* ---- elementwise activations (in place) ---- */
__global__ void k_softcap(float *s, int n, float cap) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < (size_t)n) s[i] = cap * tanhf(s[i] / cap);
}
__global__ void k_silu(float *x, int n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < (size_t)n) { float v = x[i]; x[i] = v / (1.0f + expf(-v)); }
}
__global__ void k_gelu(float *x, int n) {
    const float c = 0.7978845608028654f;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)n) { float v = x[i]; x[i] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v))); }
}
__global__ void k_silugate(float *o, const float *g, const float *u, int n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)n) { float gg = g[i]; o[i] = (gg / (1.0f + expf(-gg))) * u[i]; }
}
extern "C" void aria_cuda_softcap(float *s, int n, float cap) {
    float *d = d_in(s, n); k_softcap<<<nblocks(n), THREADS>>>(d, n, cap); CK(cudaGetLastError()); d_out(s, d, n);
}
extern "C" void aria_cuda_silu(float *x, int n) {
    float *d = d_in(x, n); k_silu<<<nblocks(n), THREADS>>>(d, n); CK(cudaGetLastError()); d_out(x, d, n);
}
extern "C" void aria_cuda_gelu_tanh(float *x, int n) {
    float *d = d_in(x, n); k_gelu<<<nblocks(n), THREADS>>>(d, n); CK(cudaGetLastError()); d_out(x, d, n);
}
extern "C" void aria_cuda_silu_gate(float *out, const float *gate, const float *up, int n) {
    float *dg = d_in(gate, n), *du = d_in(up, n), *don = d_new(n);
    k_silugate<<<nblocks(n), THREADS>>>(don, dg, du, n); CK(cudaGetLastError());
    cudaFree(dg); cudaFree(du); d_out(out, don, n);
}

/* ---- row softmax: one block per row (shared-mem max + sum), optional mask ---- */
__global__ void k_softmax(float *x, int rows, int cols, const float *mask) {
    int r = blockIdx.x; if (r >= rows) return;
    float *xr = x + (size_t)r * cols; const float *mr = mask ? mask + (size_t)r * cols : NULL;
    __shared__ float red[RED_TH];
    float mx = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) { float v = xr[i] + (mr ? mr[i] : 0.0f); xr[i] = v; mx = fmaxf(mx, v); }
    red[threadIdx.x] = mx; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]); __syncthreads(); }
    mx = red[0]; __syncthreads();
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) { float e = expf(xr[i] - mx); xr[i] = e; sum += e; }
    red[threadIdx.x] = sum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s]; __syncthreads(); }
    float inv = red[0] > 0.0f ? 1.0f / red[0] : 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) xr[i] *= inv;
}
__global__ void k_scale(float *y, const float *x, float a, int n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < (size_t)n) y[i] = x[i] * a;
}
extern "C" void aria_cuda_softmax(float *x, int rows, int cols, const float *mask) {
    float *dx = d_in(x, (size_t)rows * cols), *dm = mask ? d_in(mask, (size_t)rows * cols) : NULL;
    k_softmax<<<rows, RED_TH>>>(dx, rows, cols, dm); CK(cudaGetLastError());
    if (dm) cudaFree(dm); d_out(x, dx, (size_t)rows * cols);
}

/* ---- RoPE apply on [H,N,D] (rotate first rot_dim with pairing i,i+rot/2) ---- */
__global__ void k_rope(float *x, const float *c, const float *s, int H, int N, int D, int rot) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   /* h*N + p */
    if (idx >= H * N) return;
    int p = idx % N, rh = rot / 2;
    float *row = x + (size_t)idx * D;
    const float *cc = c + (size_t)p * rh, *ss = s + (size_t)p * rh;
    for (int i = 0; i < rh; i++) {
        float a = row[i], b = row[i + rh];
        row[i] = a * cc[i] - b * ss[i];
        row[i + rh] = b * cc[i] + a * ss[i];
    }
}

/* fused extract-head + per-head rmsnorm(weight w) + RoPE -> head-major out[h,s,:hd].
 * Replaces a k_extract_heads + k_rmsnorm + k_rope chain (saves two [H,S,hd] HBM
 * round-trips + two launches per q/k). One block per (h,s) row, hd threads (hd<=64).
 * rcos==NULL skips RoPE (cross-attn q). */
__global__ void k_extract_normrope(float *out, const float *src, const float *w,
                                   const float *rcos, const float *rsin,
                                   int S, int H, int hd, int stride, int off, int rot, float eps) {
    int hs = blockIdx.x; if (hs >= H * S) return;
    int s = hs % S, h = hs / S, d = threadIdx.x;
    const float *sp = src + (size_t)s * stride + off + (size_t)h * hd;
    __shared__ float red[64], vec[64];
    float x = sp[d];
    red[d] = x * x; __syncthreads();
    for (int st = hd / 2; st > 0; st >>= 1) { if (d < st) red[d] += red[d + st]; __syncthreads(); }
    vec[d] = x * rsqrtf(red[0] / hd + eps) * w[d];
    __syncthreads();
    int rh = rot / 2; float res;
    if (rcos == NULL || d >= rot)  res = vec[d];
    else if (d < rh) res = vec[d] * rcos[(size_t)s * rh + d]      - vec[d + rh] * rsin[(size_t)s * rh + d];
    else { int i = d - rh; res = vec[d] * rcos[(size_t)s * rh + i] + vec[i] * rsin[(size_t)s * rh + i]; }
    out[(size_t)hs * hd + d] = res;
}
extern "C" void aria_cuda_rope_apply(float *x, const float *cos_t, const float *sin_t,
                                     int H, int N, int D, int rot_dim) {
    int rh = rot_dim / 2;
    float *dx = d_in(x, (size_t)H * N * D), *dc = d_in(cos_t, (size_t)N * rh), *ds = d_in(sin_t, (size_t)N * rh);
    k_rope<<<nblocks((size_t)H * N), THREADS>>>(dx, dc, ds, H, N, D, rot_dim); CK(cudaGetLastError());
    cudaFree(dc); cudaFree(ds); d_out(x, dx, (size_t)H * N * D);
}

/* ---- multi-head SDPA: out[h]=softmax(scale*q[h]@k[h]^T + mask)@v[h] ----
 * mirrors aria_attention (scale = 1/sqrt(D)); composes gemm_nt + scale + softmax + matmul. */
extern "C" void aria_cuda_attention(float *out, const float *q, const float *k, const float *v,
                                    int H, int Nq, int Nk, int D, const float *mask) {
    float scale = 1.0f / sqrtf((float)D);
    float *dq = d_in(q, (size_t)H * Nq * D), *dk = d_in(k, (size_t)H * Nk * D);
    float *dv = d_in(v, (size_t)H * Nk * D), *dout = d_new((size_t)H * Nq * D);
    float *dscores = d_new((size_t)Nq * Nk);
    float *dmask = mask ? d_in(mask, (size_t)Nq * Nk) : NULL;
    dim3 sblk(TILE, TILE), sgrid((Nk + TILE - 1) / TILE, (Nq + TILE - 1) / TILE);
    dim3 oblk(TILE, TILE), ogrid((D + TILE - 1) / TILE, (Nq + TILE - 1) / TILE);
    for (int h = 0; h < H; h++) {
        const float *qh = dq + (size_t)h * Nq * D, *kh = dk + (size_t)h * Nk * D, *vh = dv + (size_t)h * Nk * D;
        float *oh = dout + (size_t)h * Nq * D;
        aria_gemm_nt<<<sgrid, sblk>>>(dscores, qh, kh, NULL, Nq, D, Nk);   /* scores = qh @ kh^T */
        k_scale<<<nblocks((size_t)Nq * Nk), THREADS>>>(dscores, dscores, scale, Nq * Nk);
        k_softmax<<<Nq, RED_TH>>>(dscores, Nq, Nk, dmask);
        k_matmul<<<ogrid, oblk>>>(oh, dscores, vh, Nq, Nk, D);             /* out = scores @ vh */
    }
    CK(cudaGetLastError());
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dscores); if (dmask) cudaFree(dmask);
    d_out(out, dout, (size_t)H * Nq * D);
}

/* ---- conv1d (stride 1, pre-folded weights), one thread per (cout,l) ---- */
__global__ void k_conv1d(float *out, const float *in, const float *w, const float *bias,
                         int Cin, int Cout, int K, int pad, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   /* o*L + t */
    if (idx >= Cout * L) return;
    int o = idx / L, t = idx % L;
    const float *wo = w + (size_t)o * Cin * K;
    float acc = bias ? bias[o] : 0.0f;
    for (int i = 0; i < Cin; i++) {
        const float *wi = wo + (size_t)i * K; const float *ini = in + (size_t)i * L;
        for (int kk = 0; kk < K; kk++) { int ti = t + kk - pad; if (ti >= 0 && ti < L) acc += wi[kk] * ini[ti]; }
    }
    out[idx] = acc;
}
extern "C" void aria_cuda_conv1d(float *out, const float *in, const float *w, const float *bias,
                                 int Cin, int Cout, int K, int pad, int L) {
    float *din = d_in(in, (size_t)Cin * L), *dw = d_in(w, (size_t)Cout * Cin * K);
    float *db = bias ? d_in(bias, Cout) : NULL, *dout = d_new((size_t)Cout * L);
    k_conv1d<<<nblocks((size_t)Cout * L), THREADS>>>(dout, din, dw, db, Cin, Cout, K, pad, L); CK(cudaGetLastError());
    cudaFree(din); cudaFree(dw); if (db) cudaFree(db); d_out(out, dout, (size_t)Cout * L);
}

/* ===================== device-resident DiT (E8.5) =====================
 * Weights live on the device as fp16 (fp32 compute: loaded and converted in the
 * GEMM); activations stay fp32 in a device bump arena. The denoise loop runs on
 * the device -- only the per-step latent (x in) and velocity (v out) cross the
 * bus. Encoder/decoder stay on the CPU. Mirrors aria_sa3_dit.c step/block_core. */

__global__ void k_f32_to_f16(__half *o, const float *in, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = __float2half(in[i]);
}
static __half *upload_f16(const float *h, size_t n) {
    float *tmp = NULL; __half *d = NULL;
    if (cudaMalloc(&tmp, n * sizeof(float)) != cudaSuccess) return NULL;
    if (cudaMalloc(&d, n * sizeof(__half)) != cudaSuccess) { cudaFree(tmp); return NULL; }
    cudaMemcpy(tmp, h, n * sizeof(float), cudaMemcpyHostToDevice);
    k_f32_to_f16<<<nblocks(n), THREADS>>>(d, tmp, n);
    cudaFree(tmp);
    return d;
}
static float *upload_f32(const float *h, size_t n) {
    float *d = NULL; if (cudaMalloc(&d, n * sizeof(float)) != cudaSuccess) return NULL;
    cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice); return d;
}

/* y[M,N] = x[M,K] @ W[N,K]^T + b ; W is fp16 (converted on load), x/y/b fp32. */
__global__ void k_gemm_f16w(float *y, const float *x, const __half *W, const float *b, int M, int K, int N) {
    __shared__ float xs[TILE][TILE], ws[TILE][TILE];
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        int kx = k0 + tx, ky = k0 + ty;
        xs[ty][tx] = (row < M && kx < K) ? x[(size_t)row * K + kx] : 0.0f;
        ws[ty][tx] = (col < N && ky < K) ? __half2float(W[(size_t)col * K + ky]) : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE; i++) acc += xs[ty][i] * ws[i][tx];
        __syncthreads();
    }
    if (row < M && col < N) y[(size_t)row * N + col] = acc + (b ? b[col] : 0.0f);
}
/* y[m,n] += b[n]  (broadcast bias over rows; cuBLAS GEMM has no vector-bias epilogue) */
__global__ void k_add_bias(float *y, const float *b, int M, int N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)M * N) y[i] += b[i % N];
}

/* ---- glue kernels (device pointers) ---- */
__global__ void k_add(float *y, const float *a, const float *b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = a[i] + b[i];
}
__global__ void k_transpose(float *dst, const float *src, int A, int B) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   /* a*B+b over A*B -> dst[b*A+a] */
    if (idx >= (size_t)A * B) return;
    int a = (int)(idx / B), b = (int)(idx % B);
    dst[(size_t)b * A + a] = src[idx];
}
/* fused gate + residual add: out = residual + o * sigmoid(1 - gate[col]).
 * Replaces a k_gate + k_add pair (saves one read+write pass of [S,dim] + a launch). */
__global__ void k_gate_add(float *out, const float *residual, const float *o,
                           const float *gate, int S, int dim) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)S * dim) return;
    int i = (int)(idx % dim);
    out[idx] = residual[idx] + o[idx] * (1.0f / (1.0f + expf(-(1.0f - gate[i]))));
}
__global__ void k_extract_heads(float *dst, const float *src, int S, int H, int hd, int stride, int off) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   /* (h*S+s)*hd+d */
    if (idx >= (size_t)H * S * hd) return;
    int d = (int)(idx % hd), s = (int)((idx / hd) % S), h = (int)(idx / ((size_t)hd * S));
    dst[idx] = src[(size_t)s * stride + off + (size_t)h * hd + d];
}
__global__ void k_merge_heads(float *dst, const float *src, int S, int H, int hd) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   /* src (h*S+s)*hd+d */
    if (idx >= (size_t)H * S * hd) return;
    int d = (int)(idx % hd), s = (int)((idx / hd) % S), h = (int)(idx / ((size_t)hd * S));
    dst[(size_t)s * H * hd + (size_t)h * hd + d] = src[idx];
}
__global__ void k_ff_silugate(float *out, const float *proj, int S, int inner) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   /* s*inner+i */
    if (idx >= (size_t)S * inner) return;
    int s = (int)(idx / inner), i = (int)(idx % inner);
    const float *pr = proj + (size_t)s * 2 * inner;
    float g = pr[inner + i];
    out[idx] = (g / (1.0f + expf(-g))) * pr[i];
}

/* ---- device bump arena (pointer arithmetic only) ---- */
struct darena { char *base; size_t cap, used; };
static float *da_alloc(darena *a, size_t nf) {
    size_t off = (a->used + 63) & ~((size_t)63);
    a->used = off + nf * sizeof(float);
    return (float *)(a->base + off);
}
static __half *da_alloc_half(darena *a, size_t nh) {
    size_t off = (a->used + 63) & ~((size_t)63);
    a->used = off + nh * sizeof(__half);
    return (__half *)(a->base + off);
}
static size_t da_save(darena *a) { return a->used; }
static void da_restore(darena *a, size_t m) { a->used = m; }

/* multi-head SDPA, all heads batched (scale 1/sqrt(D)). scores holds [H,Nq,Nk].
 * fp16=1: q/k/v converted to fp16 in arena scratch and the QK^T/AV run on tensor
 * cores (fp16->fp32 scores, fp32 softmax, fp16 AV) -- ~2-4x the fp32 sgemm path, used
 * for the DiT (the dominant attention). fp16=0: the original fp32 cublasSgemm path,
 * kept for the cheap small-music decoder chunks where fp16 would only cost accuracy. */
static void attn_dev(cublasHandle_t cb, darena *ar, float *out, const float *q, const float *k,
                     const float *v, int H, int Nq, int Nk, int D, float *scores, int fp16) {
    const float scale = 1.0f / sqrtf((float)D), one = 1.0f, zero = 0.0f;
    size_t ns = (size_t)H * Nq * Nk;
    if (!fp16) {
        cublasSgemmStridedBatched(cb, CUBLAS_OP_T, CUBLAS_OP_N, Nk, Nq, D,
            &one, k, D, (long long)Nk * D, q, D, (long long)Nq * D,
            &zero, scores, Nk, (long long)Nq * Nk, H);
        k_scale<<<nblocks(ns), THREADS>>>(scores, scores, scale, H * Nq * Nk);
        k_softmax<<<H * Nq, RED_TH>>>(scores, H * Nq, Nk, NULL);
        cublasSgemmStridedBatched(cb, CUBLAS_OP_N, CUBLAS_OP_N, D, Nq, Nk,
            &one, v, D, (long long)Nk * D, scores, Nk, (long long)Nq * Nk,
            &zero, out, D, (long long)Nq * D, H);
        return;
    }
    size_t nq = (size_t)H * Nq * D, nk = (size_t)H * Nk * D;
    size_t mark = da_save(ar);
    __half *qh = da_alloc_half(ar, nq), *kh = da_alloc_half(ar, nk);
    __half *vh = da_alloc_half(ar, nk), *sh = da_alloc_half(ar, ns);
    k_f32_to_f16<<<nblocks(nq), THREADS>>>(qh, q, nq);
    k_f32_to_f16<<<nblocks(nk), THREADS>>>(kh, k, nk);
    k_f32_to_f16<<<nblocks(nk), THREADS>>>(vh, v, nk);
    cublasGemmStridedBatchedEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, Nk, Nq, D,
        &one, kh, CUDA_R_16F, D, (long long)Nk * D, qh, CUDA_R_16F, D, (long long)Nq * D,
        &zero, scores, CUDA_R_32F, Nk, (long long)Nq * Nk, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    k_scale<<<nblocks(ns), THREADS>>>(scores, scores, scale, H * Nq * Nk);
    k_softmax<<<H * Nq, RED_TH>>>(scores, H * Nq, Nk, NULL);
    k_f32_to_f16<<<nblocks(ns), THREADS>>>(sh, scores, ns);
    cublasGemmStridedBatchedEx(cb, CUBLAS_OP_N, CUBLAS_OP_N, D, Nq, Nk,
        &one, vh, CUDA_R_16F, D, (long long)Nk * D, sh, CUDA_R_16F, Nk, (long long)Nq * Nk,
        &zero, out, CUDA_R_32F, D, (long long)Nq * D, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    da_restore(ar, mark);
}

/* ---- device quantized weight (E9.4): packed in VRAM, dequant-on-use to fp16 ----
 * Q8/Q4 weights stay packed on the device (medium fits low VRAM); each GEMM
 * dequantizes its weight into a reused fp16 scratch, then runs the tensor-core
 * cuBLAS path. dt==F32 keeps the plain fp16 weight (the default). */
struct dqw {
    int dt, N, K;
    __half *f16;     /* ARIA_F32: the fp16 weight */
    void *q;         /* ARIA_Q8: int8_t* ; ARIA_Q4: uint8_t* (packed) */
    float *scale;    /* ARIA_Q8: [N]; ARIA_Q4: [N*nblocks] */
};

__global__ void k_dequant_q8(__half *W, const int8_t *q, const float *scale, int N, int K) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)N * K) return;
    W[idx] = __float2half((float)q[idx] * scale[idx / K]);
}
/* asymmetric (zero-point) int4: per block [min, scale]; W = min + nib*scale. */
__global__ void k_dequant_q4(__half *W, const uint8_t *q, const float *scale, int N, int K) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)N * K) return;
    int n = (int)(idx / K), k = (int)(idx % K);
    int nblk = (K + ARIA_Q4_BLOCK - 1) / ARIA_Q4_BLOCK;
    size_t rb = (size_t)((K + 1) / 2);
    uint8_t byte = q[(size_t)n * rb + (k >> 1)];
    int nib = (k & 1) ? (byte >> 4) : (byte & 0x0F);
    size_t si = ((size_t)n * nblk + (k / ARIA_Q4_BLOCK)) * 2;
    W[idx] = __float2half(scale[si] + (float)nib * scale[si + 1]);
}

/* host-quantize a [N,K] f32 weight per dt and upload it (packed) to the device. */
static int upload_dqw(dqw *w, const float *W, int N, int K, aria_dtype dt) {
    w->dt = dt; w->N = N; w->K = K; w->f16 = NULL; w->q = NULL; w->scale = NULL;
    if (dt == ARIA_Q8 || dt == ARIA_Q4) {
        size_t qb = (dt == ARIA_Q8) ? aria_q8_qbytes(N, K) : aria_q4_qbytes(N, K);
        size_t ns = (dt == ARIA_Q8) ? aria_q8_nscale(N, K) : aria_q4_nscale(N, K);
        void *hq = malloc(qb); float *hs = (float *)malloc(ns * sizeof(float));
        if (dt == ARIA_Q8) aria_q8_quant((int8_t *)hq, hs, W, N, K);
        else               aria_q4_quant((uint8_t *)hq, hs, W, N, K);
        CK(cudaMalloc(&w->q, qb)); CK(cudaMemcpy(w->q, hq, qb, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&w->scale, ns * sizeof(float)));
        CK(cudaMemcpy(w->scale, hs, ns * sizeof(float), cudaMemcpyHostToDevice));
        free(hq); free(hs);
        return w->q != NULL && w->scale != NULL;
    }
    w->f16 = upload_f16(W, (size_t)N * K);
    return w->f16 != NULL;
}
static void free_dqw(dqw *w) { cudaFree(w->f16); cudaFree(w->q); cudaFree(w->scale); }

struct blk_dev {
    const float *pre_norm, *cross_norm, *ff_norm, *sa_q_norm, *sa_k_norm, *ca_q_norm, *ca_k_norm;
    const float *ssg, *ff_in_b, *ff_out_b;
    dqw sa_to_qkv, sa_to_out, ca_to_q, ca_to_out, ca_to_kv, ff_in_w, ff_out_w;
};

struct aria_cuda_dit {
    int depth, ed, H, hd, inner, io_ch, n_mem, rot;
    int differential;         /* medium: differential self+cross attention */
    aria_dtype precision;     /* block GEMM weight precision (fp32->fp16 / q8 / q4) */
    __half *dqbuf;            /* dequant scratch: largest weight, reused per GEMM */
    __half *preprocess, *postprocess, *project_in, *project_out;
    float *memory_tokens;
    blk_dev *blocks;
    cublasHandle_t cublas;
    /* per-request */
    int T, S, n_cond, have_req;
    float *dx, *dv, *dgcond, *rope_cos, *rope_sin;
    float **cross_k, **cross_v, **cross_kd;
    darena arena;
};

/* a[i] -= b[i] (differential attention combine on the device) */
__global__ void k_sub(float *a, const float *b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] -= b[i];
}

/* y[M,N] = x[M,K] @ W[N,K]^T + b via cuBLAS GemmEx (fp16 in, fp32 accumulate ->
 * tensor cores on sm_80+). x (fp32 activations) is converted to fp16 into transient
 * arena scratch; W is already fp16. Single default stream, so reusing the scratch
 * region after the (queued) GEMM is safe by stream ordering. */
static void gemm_f16w(cublasHandle_t cb, darena *ar, float *y, const float *x, const __half *W,
                      const float *b, int M, int K, int N) {
    size_t mark = da_save(ar);
    __half *xh = da_alloc_half(ar, (size_t)M * K);
    k_f32_to_f16<<<nblocks((size_t)M * K), THREADS>>>(xh, x, (size_t)M * K);
    const float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                 W, CUDA_R_16F, K, xh, CUDA_R_16F, K, &beta,
                 y, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (b) k_add_bias<<<nblocks((size_t)M * N), THREADS>>>(y, b, M, N);
    da_restore(ar, mark);
}

/* GEMM with a (possibly quantized) weight: dequant Q8/Q4 into the reused fp16
 * scratch, then the fp16 tensor-core path. dt==F32 uses the fp16 weight directly.
 * Stream-ordered, so reusing h->dqbuf across calls in a block is safe. */
static void gemm_dqw(aria_cuda_dit *h, float *y, const float *x, const dqw *w, const float *b, int M) {
    if (w->dt == ARIA_F32) { gemm_f16w(h->cublas, &h->arena, y, x, w->f16, b, M, w->K, w->N); return; }
    size_t nk = (size_t)w->N * w->K;
    if (w->dt == ARIA_Q8)
        k_dequant_q8<<<nblocks(nk), THREADS>>>(h->dqbuf, (const int8_t *)w->q, w->scale, w->N, w->K);
    else
        k_dequant_q4<<<nblocks(nk), THREADS>>>(h->dqbuf, (const uint8_t *)w->q, w->scale, w->N, w->K);
    gemm_f16w(h->cublas, &h->arena, y, x, h->dqbuf, b, M, w->K, w->N);
}

static void dit_block_dev(aria_cuda_dit *h, float *seq, int blk) {
    int S = h->S, ed = h->ed, H = h->H, hd = h->hd, dim = ed, inner = h->inner, Sc = h->n_cond, rot = h->rot;
    blk_dev *w = &h->blocks[blk];
    darena *ar = &h->arena; size_t mark = da_save(ar);
    int nSd = (int)nblocks((size_t)S * dim);

    float *mod = da_alloc(ar, 6 * dim);
    k_add<<<nblocks((size_t)6 * dim), THREADS>>>(mod, w->ssg, h->dgcond, 6 * dim);
    float *scale_self = mod, *shift_self = mod + dim, *gate_self = mod + 2 * dim;
    float *scale_ff = mod + 3 * dim, *shift_ff = mod + 4 * dim, *gate_ff = mod + 5 * dim;
    float *residual = da_alloc(ar, (size_t)S * dim), *hh = da_alloc(ar, (size_t)S * dim);
    float *o = da_alloc(ar, (size_t)S * dim), *merged = da_alloc(ar, (size_t)S * dim);
    float *qh = da_alloc(ar, (size_t)H * S * hd), *kh = da_alloc(ar, (size_t)H * S * hd);
    float *vh = da_alloc(ar, (size_t)H * S * hd), *ao = da_alloc(ar, (size_t)H * S * hd);
    float *scores = da_alloc(ar, (size_t)H * S * (S > Sc ? S : Sc));   /* all heads (batched attn) */
    int diff = h->differential, nHShd = (int)nblocks((size_t)H * S * hd);
    size_t nHSd = (size_t)H * S * hd;
    float *qdh = NULL, *kdh = NULL, *aod = NULL;   /* differential: q/k_diff + 2nd attn out */
    if (diff) { qdh = da_alloc(ar, nHSd); kdh = da_alloc(ar, nHSd); aod = da_alloc(ar, nHSd); }

    /* self-attention (differential medium: out = attn(q,k,v) - attn(qd,kd,v)) */
    cudaMemcpy(residual, seq, (size_t)S * dim * sizeof(float), cudaMemcpyDeviceToDevice);
    k_rmsnorm_adaln<<<S, RED_TH>>>(hh, seq, w->pre_norm, scale_self, shift_self, S, dim, 1e-5f);
    int nq = diff ? 5 : 3;
    float *qkv = da_alloc(ar, (size_t)S * nq * dim);
    gemm_dqw(h, qkv, hh, &w->sa_to_qkv, NULL, S);
    k_extract_normrope<<<H * S, hd>>>(qh, qkv, w->sa_q_norm, h->rope_cos, h->rope_sin, S, H, hd, nq * dim, 0, rot, 1e-6f);
    k_extract_normrope<<<H * S, hd>>>(kh, qkv, w->sa_k_norm, h->rope_cos, h->rope_sin, S, H, hd, nq * dim, dim, rot, 1e-6f);
    k_extract_heads<<<nHShd, THREADS>>>(vh, qkv, S, H, hd, nq * dim, 2 * dim);
    attn_dev(h->cublas, ar, ao, qh, kh, vh, H, S, S, hd, scores, 1);
    if (diff) {
        k_extract_normrope<<<H * S, hd>>>(qdh, qkv, w->sa_q_norm, h->rope_cos, h->rope_sin, S, H, hd, nq * dim, 3 * dim, rot, 1e-6f);
        k_extract_normrope<<<H * S, hd>>>(kdh, qkv, w->sa_k_norm, h->rope_cos, h->rope_sin, S, H, hd, nq * dim, 4 * dim, rot, 1e-6f);
        attn_dev(h->cublas, ar, aod, qdh, kdh, vh, H, S, S, hd, scores, 1);
        k_sub<<<nblocks(nHSd), THREADS>>>(ao, aod, nHSd);
    }
    k_merge_heads<<<nHShd, THREADS>>>(merged, ao, S, H, hd);
    gemm_dqw(h, o, merged, &w->sa_to_out, NULL, S);
    k_gate_add<<<nSd, THREADS>>>(seq, residual, o, gate_self, S, dim);

    /* cross-attention (cached K/V; differential: q_diff + cross_kd) */
    cudaMemcpy(residual, seq, (size_t)S * dim * sizeof(float), cudaMemcpyDeviceToDevice);
    k_rmsnorm<<<S, RED_TH>>>(hh, seq, w->cross_norm, S, dim, 1e-5f, 0);
    int ncq = diff ? 2 : 1;
    float *q = da_alloc(ar, (size_t)S * ncq * dim);
    gemm_dqw(h, q, hh, &w->ca_to_q, NULL, S);
    k_extract_normrope<<<H * S, hd>>>(qh, q, w->ca_q_norm, NULL, NULL, S, H, hd, ncq * dim, 0, rot, 1e-6f);
    attn_dev(h->cublas, ar, ao, qh, h->cross_k[blk], h->cross_v[blk], H, S, Sc, hd, scores, 1);
    if (diff) {
        k_extract_normrope<<<H * S, hd>>>(qdh, q, w->ca_q_norm, NULL, NULL, S, H, hd, ncq * dim, dim, rot, 1e-6f);
        attn_dev(h->cublas, ar, aod, qdh, h->cross_kd[blk], h->cross_v[blk], H, S, Sc, hd, scores, 1);
        k_sub<<<nblocks(nHSd), THREADS>>>(ao, aod, nHSd);
    }
    k_merge_heads<<<nHShd, THREADS>>>(merged, ao, S, H, hd);
    gemm_dqw(h, o, merged, &w->ca_to_out, NULL, S);
    k_add<<<nSd, THREADS>>>(seq, residual, o, (size_t)S * dim);

    /* feed-forward (GLU) */
    cudaMemcpy(residual, seq, (size_t)S * dim * sizeof(float), cudaMemcpyDeviceToDevice);
    k_rmsnorm_adaln<<<S, RED_TH>>>(hh, seq, w->ff_norm, scale_ff, shift_ff, S, dim, 1e-5f);
    float *proj = da_alloc(ar, (size_t)S * 2 * inner), *gated = da_alloc(ar, (size_t)S * inner);
    gemm_dqw(h, proj, hh, &w->ff_in_w, w->ff_in_b, S);
    k_ff_silugate<<<nblocks((size_t)S * inner), THREADS>>>(gated, proj, S, inner);
    gemm_dqw(h, o, gated, &w->ff_out_w, w->ff_out_b, S);
    k_gate_add<<<nSd, THREADS>>>(seq, residual, o, gate_ff, S, dim);

    da_restore(ar, mark);
}

extern "C" aria_cuda_dit *aria_cuda_dit_create(const aria_sa3_dit_view *v, aria_dtype precision) {
    if (!aria_cuda_available()) return NULL;
    int ed = v->ed, inner = v->inner;
    if (precision != ARIA_Q8 && precision != ARIA_Q4) precision = ARIA_F32;  /* fp16/bf16 -> fp16 storage */
    /* Q4 is mixed precision (mirrors aria_sa3_dit_quantize): the 4 attention
     * projections (6*ed^2/block) stay Q8 and only the FFN (3*inner*ed/block) goes
     * asymmetric Q4 -- ~9% DiT velocity error vs ~23% for uniform Q4. Rough fit:
     * attention ~1 B/elem (q8), FFN ~1 B/elem (q4: 0.5 nibble + scale, rounded up),
     * fp16 = 2 B/elem, plus the dequant scratch. */
    /* differential (medium): sa_to_qkv 5*ed^2, ca_to_q 2*ed^2 (vs 3/1) -> 9*ed^2 attn */
    size_t attn_per = v->differential ? 9 : 6;
    size_t attn_elem = (size_t)v->depth * attn_per * (size_t)ed * ed;
    size_t ffn_elem  = (size_t)v->depth * 3 * (size_t)inner * ed;
    size_t wbytes = (precision == ARIA_F32) ? (attn_elem + ffn_elem) * 2 : (attn_elem + ffn_elem);
    size_t need = wbytes + (precision != ARIA_F32 ? (size_t)2 * inner * ed * sizeof(__half) : 0) + (64u << 20);
    size_t freeb = 0, totb = 0; cudaMemGetInfo(&freeb, &totb);
    if (freeb < need) { fprintf(stderr, "aria_cuda_dit: need ~%zu MiB, only %zu free\n", need >> 20, freeb >> 20); return NULL; }

    aria_cuda_dit *h = (aria_cuda_dit *)calloc(1, sizeof(aria_cuda_dit));
    h->depth = v->depth; h->ed = ed; h->H = v->num_heads; h->hd = v->head_dim;
    h->inner = inner; h->io_ch = v->io_ch; h->n_mem = v->n_mem; h->rot = v->rot_dim;
    h->precision = precision; h->differential = v->differential;
    if (cublasCreate(&h->cublas) != CUBLAS_STATUS_SUCCESS) { free(h); return NULL; }
    if (precision != ARIA_F32) { CK(cudaMalloc(&h->dqbuf, (size_t)2 * inner * ed * sizeof(__half))); }
    int C = v->io_ch;
    h->preprocess  = upload_f16(v->preprocess,  (size_t)C * C);
    h->postprocess = upload_f16(v->postprocess, (size_t)C * C);
    h->project_in  = upload_f16(v->project_in,  (size_t)ed * C);
    h->project_out = upload_f16(v->project_out, (size_t)C * ed);
    h->memory_tokens = upload_f32(v->memory_tokens, (size_t)v->n_mem * ed);
    h->blocks = (blk_dev *)calloc(v->depth, sizeof(blk_dev));
    int ok = h->preprocess && h->postprocess && h->project_in && h->project_out && h->memory_tokens;
    for (int b = 0; b < v->depth && ok; b++) {
        const aria_dit_block_w *s = &v->blocks[b]; blk_dev *d = &h->blocks[b];
        d->pre_norm   = upload_f32(s->pre_norm, ed);
        d->cross_norm = upload_f32(s->cross_norm, ed);
        d->ff_norm    = upload_f32(s->ff_norm, ed);
        d->sa_q_norm  = upload_f32(s->sa_q_norm, h->hd);
        d->sa_k_norm  = upload_f32(s->sa_k_norm, h->hd);
        d->ca_q_norm  = upload_f32(s->ca_q_norm, h->hd);
        d->ca_k_norm  = upload_f32(s->ca_k_norm, h->hd);
        d->ssg        = upload_f32(s->to_scale_shift_gate, (size_t)6 * ed);
        d->ff_in_b    = upload_f32(s->ff_in_b, (size_t)2 * inner);
        d->ff_out_b   = upload_f32(s->ff_out_b, ed);
        /* Q4 mixed precision: attention projections stay Q8 (error-sensitive).
         * differential (medium): sa_to_qkv is [5*ed,ed], ca_to_q [2*ed,ed]. */
        aria_dtype adt = (precision == ARIA_Q4) ? ARIA_Q8 : precision;
        int nq = v->differential ? 5 : 3, ncq = v->differential ? 2 : 1, nkv = v->differential ? 3 : 2;
        int u = 1;
        u &= upload_dqw(&d->sa_to_qkv, s->sa_to_qkv, nq * ed, ed, adt);
        u &= upload_dqw(&d->sa_to_out, s->sa_to_out, ed, ed, adt);
        u &= upload_dqw(&d->ca_to_q,   s->ca_to_q,   ncq * ed, ed, adt);
        u &= upload_dqw(&d->ca_to_out, s->ca_to_out, ed, ed, adt);
        /* cross K/V projection: device-resident, projected once per request from
         * cross_ed (kept f16/f32, error-sensitive -- not quantized). */
        u &= upload_dqw(&d->ca_to_kv,  s->ca_to_kv,  nkv * ed, ed, ARIA_F32);   /* cross-K/V is fidelity-sensitive: keep fp16 */
        u &= upload_dqw(&d->ff_in_w,   s->ff_in_w,   2 * inner, ed, precision);
        u &= upload_dqw(&d->ff_out_w,  s->ff_out_w,  ed, inner, precision);
        ok = u && d->ssg && d->pre_norm && d->ca_k_norm;
    }
    if (precision != ARIA_F32 && !h->dqbuf) ok = 0;
    if (!ok) { aria_cuda_dit_free(h); return NULL; }
    return h;
}

/* free per-request device buffers (so set_request can be re-called per generation) */
static void free_request(aria_cuda_dit *h) {
    if (!h->have_req) return;
    cudaFree(h->dx); cudaFree(h->dv); cudaFree(h->dgcond);
    cudaFree(h->rope_cos); cudaFree(h->rope_sin); cudaFree(h->arena.base);
    if (h->cross_k) for (int b = 0; b < h->depth; b++) cudaFree(h->cross_k[b]);
    if (h->cross_v) for (int b = 0; b < h->depth; b++) cudaFree(h->cross_v[b]);
    if (h->cross_kd) for (int b = 0; b < h->depth; b++) cudaFree(h->cross_kd[b]);
    free(h->cross_k); free(h->cross_v); free(h->cross_kd);
    h->cross_k = h->cross_v = h->cross_kd = NULL; h->dx = h->dv = h->dgcond = NULL;
    h->rope_cos = h->rope_sin = NULL; h->arena.base = NULL; h->have_req = 0;
}

extern "C" void aria_cuda_dit_set_request(aria_cuda_dit *h, const aria_sa3_dit_req_view *rv) {
    free_request(h);   /* persistent weights, fresh per-request caches */
    int ed = h->ed, C = h->io_ch, S = rv->S, T = rv->T, H = h->H, hd = h->hd, n_cond = rv->n_cond, rh = h->rot / 2;
    h->T = T; h->S = S; h->n_cond = n_cond;
    cudaMalloc(&h->dx, (size_t)C * T * sizeof(float));
    cudaMalloc(&h->dv, (size_t)C * T * sizeof(float));
    cudaMalloc(&h->dgcond, (size_t)6 * ed * sizeof(float));
    h->rope_cos = upload_f32(rv->rope_cos, (size_t)S * rh);
    h->rope_sin = upload_f32(rv->rope_sin, (size_t)S * rh);
    /* device arena: one block's peak + step buffers (mirrors the CPU sizing;
     * differential needs ~32*S*ed for the extra 5-way qkv + q/k_diff + 2nd attn).
     * Allocated before the cross-K/V projection, which borrows it for GEMM scratch. */
    /* +S*ed + H*S*max(S,n_cond): fp16 q/k/v + scores scratch for tensor-core attn_dev */
    size_t block_floats = (size_t)6 * ed + (h->differential ? 32 : 24) * (size_t)S * ed
                          + 2 * (size_t)H * S * (S > n_cond ? S : n_cond) + 2 * (size_t)S * ed;
    size_t step_floats = (size_t)S * ed + 8 * (size_t)T * C + 8 * (size_t)ed;
    h->arena.cap = (block_floats + step_floats + (1u << 20)) * sizeof(float);
    h->arena.used = 0;
    cudaMalloc(&h->arena.base, h->arena.cap);

    int nkv = h->differential ? 3 : 2;
    size_t khv = (size_t)H * n_cond * hd;
    h->cross_k = (float **)calloc(h->depth, sizeof(float *));
    h->cross_v = (float **)calloc(h->depth, sizeof(float *));
    h->cross_kd = h->differential ? (float **)calloc(h->depth, sizeof(float *)) : NULL;
    if (rv->cross_ed) {
        /* project cross K/V on the device from cross_ed [n_cond, ed], once per
         * request (was a per-request CPU GEMM + 24-block upload before). */
        float *ced = upload_f32(rv->cross_ed, (size_t)n_cond * ed);
        for (int b = 0; b < h->depth; b++) {
            blk_dev *w = &h->blocks[b];
            cudaMalloc(&h->cross_k[b], khv * sizeof(float));
            cudaMalloc(&h->cross_v[b], khv * sizeof(float));
            if (h->cross_kd) cudaMalloc(&h->cross_kd[b], khv * sizeof(float));
            size_t mark = da_save(&h->arena);
            float *kv = da_alloc(&h->arena, (size_t)n_cond * nkv * ed);     /* [n_cond, nkv*ed] */
            gemm_dqw(h, kv, ced, &w->ca_to_kv, NULL, n_cond);
            k_extract_heads<<<nblocks(khv), THREADS>>>(h->cross_k[b], kv, n_cond, H, hd, nkv * ed, 0);
            if (h->cross_kd) {   /* chunk order: k, k_diff, v */
                k_extract_heads<<<nblocks(khv), THREADS>>>(h->cross_kd[b], kv, n_cond, H, hd, nkv * ed, ed);
                k_extract_heads<<<nblocks(khv), THREADS>>>(h->cross_v[b],  kv, n_cond, H, hd, nkv * ed, 2 * ed);
                k_rmsnorm<<<H * n_cond, RED_TH>>>(h->cross_kd[b], h->cross_kd[b], w->ca_k_norm, H * n_cond, hd, 1e-6f, 0);
            } else {
                k_extract_heads<<<nblocks(khv), THREADS>>>(h->cross_v[b], kv, n_cond, H, hd, nkv * ed, ed);
            }
            k_rmsnorm<<<H * n_cond, RED_TH>>>(h->cross_k[b], h->cross_k[b], w->ca_k_norm, H * n_cond, hd, 1e-6f, 0);
            da_restore(&h->arena, mark);
        }
        cudaFree(ced);
    } else {
        for (int b = 0; b < h->depth; b++) {   /* legacy: CPU-projected K/V uploaded */
            h->cross_k[b] = upload_f32(rv->cross_k[b], khv);
            h->cross_v[b] = upload_f32(rv->cross_v[b], khv);
            if (h->cross_kd) h->cross_kd[b] = upload_f32(rv->cross_kd[b], khv);
        }
    }
    h->have_req = 1;
}

extern "C" void aria_cuda_dit_step(aria_cuda_dit *h, float *v_CT, const float *x_CT, const float *gcond) {
    int C = h->io_ch, ed = h->ed, Mt = h->n_mem, S = h->S, T = h->T;
    cudaMemcpy(h->dx, x_CT, (size_t)C * T * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(h->dgcond, gcond, (size_t)6 * ed * sizeof(float), cudaMemcpyHostToDevice);
    darena *ar = &h->arena; size_t mark = da_save(ar);

    float *xtc = da_alloc(ar, (size_t)T * C);
    k_transpose<<<nblocks((size_t)C * T), THREADS>>>(xtc, h->dx, C, T);   /* [C,T]->[T,C] */
    float *pre = da_alloc(ar, (size_t)T * C);
    gemm_f16w(h->cublas, &h->arena, pre, xtc, h->preprocess, NULL, T, C, C);
    k_add<<<nblocks((size_t)T * C), THREADS>>>(xtc, xtc, pre, (size_t)T * C);

    float *seq = da_alloc(ar, (size_t)S * ed);
    cudaMemcpy(seq, h->memory_tokens, (size_t)Mt * ed * sizeof(float), cudaMemcpyDeviceToDevice);
    gemm_f16w(h->cublas, &h->arena, seq + (size_t)Mt * ed, xtc, h->project_in, NULL, T, C, ed);

    for (int b = 0; b < h->depth; b++) dit_block_dev(h, seq, b);

    float *outtc = da_alloc(ar, (size_t)T * C);
    gemm_f16w(h->cublas, &h->arena, outtc, seq + (size_t)Mt * ed, h->project_out, NULL, T, ed, C);
    float *postc = da_alloc(ar, (size_t)T * C);
    gemm_f16w(h->cublas, &h->arena, postc, outtc, h->postprocess, NULL, T, C, C);
    k_add<<<nblocks((size_t)T * C), THREADS>>>(outtc, outtc, postc, (size_t)T * C);
    k_transpose<<<nblocks((size_t)T * C), THREADS>>>(h->dv, outtc, T, C);  /* [T,C]->[C,T] */

    da_restore(ar, mark);
    cudaMemcpy(v_CT, h->dv, (size_t)C * T * sizeof(float), cudaMemcpyDeviceToHost);
}

extern "C" void aria_cuda_dit_free(aria_cuda_dit *h) {
    if (!h) return;
    if (h->cublas) cublasDestroy(h->cublas);
    cudaFree(h->dqbuf);
    cudaFree(h->preprocess); cudaFree(h->postprocess); cudaFree(h->project_in); cudaFree(h->project_out);
    cudaFree(h->memory_tokens);
    if (h->blocks) for (int b = 0; b < h->depth; b++) {
        blk_dev *d = &h->blocks[b];
        cudaFree((void *)d->pre_norm); cudaFree((void *)d->cross_norm); cudaFree((void *)d->ff_norm);
        cudaFree((void *)d->sa_q_norm); cudaFree((void *)d->sa_k_norm); cudaFree((void *)d->ca_q_norm);
        cudaFree((void *)d->ca_k_norm);
        cudaFree((void *)d->ssg); cudaFree((void *)d->ff_in_b); cudaFree((void *)d->ff_out_b);
        free_dqw(&d->sa_to_qkv); free_dqw(&d->sa_to_out); free_dqw(&d->ca_to_q);
        free_dqw(&d->ca_to_out); free_dqw(&d->ca_to_kv); free_dqw(&d->ff_in_w); free_dqw(&d->ff_out_w);
    }
    free(h->blocks);
    free_request(h);
    free(h);
}

/* ===================== device-resident taae decoder (E8.5c) =====================
 * Mirrors aria_sa3_dec.c, but batches all S=34 chunks of a pass into one workload:
 * the per-token linears become [B*S, .] GEMMs, and the per-chunk differential
 * attention runs B*H batched. Weights fp16 (GEMMs) / fp32 (norms, conv, tokens). */

#define DC_D   768
#define DC_H   12
#define DC_HD  64
#define DC_ROT 32
#define DC_INNER 2304
#define DC_QKV 3840   /* 5*768 */
#define DC_S   34
#define DC_OUT 512

__global__ void k_subtract(float *a, const float *b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) a[i] -= b[i];
}
/* (dynamic-tanh kernel k_dyt is defined above, shared with aria_cuda_dynamic_tanh) */
/* [B*H,S,hd] from src[B*S, stride] at offset off (per chunk b, head h, pos s) */
__global__ void k_extract_heads_b(float *dst, const float *src, int B, int S, int H, int hd, int stride, int off) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)B * H * S * hd) return;
    int d = (int)(idx % hd), s = (int)((idx / hd) % S), h = (int)((idx / ((size_t)hd * S)) % H);
    int b = (int)(idx / ((size_t)hd * S * H));
    dst[idx] = src[(size_t)(b * S + s) * stride + off + (size_t)h * hd + d];
}
/* [B*S, H*hd] from src[B*H,S,hd] */
__global__ void k_merge_heads_b(float *dst, const float *src, int B, int S, int H, int hd) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)B * H * S * hd) return;
    int d = (int)(idx % hd), s = (int)((idx / hd) % S), h = (int)((idx / ((size_t)hd * S)) % H);
    int b = (int)(idx / ((size_t)hd * S * H));
    dst[(size_t)(b * S + s) * (H * hd) + (size_t)h * hd + d] = src[idx];
}
/* expand x[T,D] -> seq[Tp*17, D]: token t at row t*17 (zeros if padded), new_tokens at the next 16 */
__global__ void k_expand_tokens(float *seq, const float *x, const float *new_tokens, int T, int Tp, int D) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)Tp * 17 * D) return;
    int d = (int)(idx % D), p = (int)(idx / D), t = p / 17, pos = p % 17;
    if (pos == 0) seq[idx] = (t < T) ? x[(size_t)t * D + d] : 0.0f;
    else          seq[idx] = new_tokens[d];
}
/* feat[D, Lout] (channel-major) = last 16 of each 17-group: feat[d,t*16+j] = seq[(t*17+1+j)*D+d] */
__global__ void k_extract_feat(float *feat, const float *seq, int D, int Lout) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)D * Lout) return;
    int d = (int)(idx / Lout), col = (int)(idx % Lout), t = col / 16, j = col % 16;
    feat[idx] = seq[(size_t)(t * 17 + 1 + j) * D + d];
}
/* unpatch: audio[2, L*256] from dec[512, L]: audio[c, l*256+hh] = dec[c*256+hh, l] */
__global__ void k_unpatch(float *audio, const float *dec, int L) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)2 * L * 256) return;
    int c = (int)(idx / ((size_t)L * 256));
    size_t rem = idx % ((size_t)L * 256);
    int l = (int)(rem / 256), hh = (int)(rem % 256);
    audio[idx] = dec[(size_t)(c * 256 + hh) * L + l];
}

struct dec_blk_dev {
    float pre_alpha, qn_alpha, kn_alpha, ff_alpha;
    const float *pre_gamma, *pre_beta, *qn_gamma, *qn_beta, *kn_gamma, *kn_beta, *ff_gamma, *ff_beta;
    const float *ff_in_b, *ff_out_b;
    __half *to_qkv, *to_out, *ff_in_w, *ff_out_w;
};
struct aria_cuda_dec {
    float running_std;
    __half *proj_w; const float *proj_b, *new_tokens, *mapping_b;
    float *mapping_w;       /* fp32 (k_conv1d) */
    dec_blk_dev blocks[6];
    float *rcos, *rsin;
    cublasHandle_t cublas;
};

/* one taae block on seq[B*S, D] (B chunks of S=34), in place, scratch from `ar` */
static void taae_block_dev(aria_cuda_dec *h, float *seq, int B, const dec_blk_dev *w, darena *ar) {
    int S = DC_S, D = DC_D, H = DC_H, hd = DC_HD;
    int N = B * S, NH = B * H * S * hd;
    size_t mark = da_save(ar);
    float *res = da_alloc(ar, (size_t)N * D), *hh = da_alloc(ar, (size_t)N * D);
    float *q  = da_alloc(ar, (size_t)NH), *k  = da_alloc(ar, (size_t)NH), *v = da_alloc(ar, (size_t)NH);
    float *qd = da_alloc(ar, (size_t)NH), *kd = da_alloc(ar, (size_t)NH);
    float *ob = da_alloc(ar, (size_t)NH), *od = da_alloc(ar, (size_t)NH);
    float *merged = da_alloc(ar, (size_t)N * D), *o = da_alloc(ar, (size_t)N * D);
    float *scores = da_alloc(ar, (size_t)B * H * S * S);

    /* differential self-attention */
    cudaMemcpy(res, seq, (size_t)N * D * sizeof(float), cudaMemcpyDeviceToDevice);
    k_dyt<<<nblocks((size_t)N * D), THREADS>>>(hh, seq, w->pre_alpha, w->pre_gamma, w->pre_beta, (size_t)N * D, D);
    float *qkv = da_alloc(ar, (size_t)N * DC_QKV);
    gemm_f16w(h->cublas, ar, qkv, hh, w->to_qkv, NULL, N, D, DC_QKV);
    k_extract_heads_b<<<nblocks((size_t)NH), THREADS>>>(q,  qkv, B, S, H, hd, DC_QKV, 0);
    k_extract_heads_b<<<nblocks((size_t)NH), THREADS>>>(k,  qkv, B, S, H, hd, DC_QKV, 768);
    k_extract_heads_b<<<nblocks((size_t)NH), THREADS>>>(v,  qkv, B, S, H, hd, DC_QKV, 1536);
    k_extract_heads_b<<<nblocks((size_t)NH), THREADS>>>(qd, qkv, B, S, H, hd, DC_QKV, 2304);
    k_extract_heads_b<<<nblocks((size_t)NH), THREADS>>>(kd, qkv, B, S, H, hd, DC_QKV, 3072);
    size_t bhs = (size_t)B * H * S;
    k_dyt<<<nblocks((size_t)NH), THREADS>>>(q,  q,  w->qn_alpha, w->qn_gamma, w->qn_beta, (size_t)NH, hd);
    k_dyt<<<nblocks((size_t)NH), THREADS>>>(qd, qd, w->qn_alpha, w->qn_gamma, w->qn_beta, (size_t)NH, hd);
    k_dyt<<<nblocks((size_t)NH), THREADS>>>(k,  k,  w->kn_alpha, w->kn_gamma, w->kn_beta, (size_t)NH, hd);
    k_dyt<<<nblocks((size_t)NH), THREADS>>>(kd, kd, w->kn_alpha, w->kn_gamma, w->kn_beta, (size_t)NH, hd);
    k_rope<<<nblocks(bhs), THREADS>>>(q,  h->rcos, h->rsin, B * H, S, hd, DC_ROT);
    k_rope<<<nblocks(bhs), THREADS>>>(qd, h->rcos, h->rsin, B * H, S, hd, DC_ROT);
    k_rope<<<nblocks(bhs), THREADS>>>(k,  h->rcos, h->rsin, B * H, S, hd, DC_ROT);
    k_rope<<<nblocks(bhs), THREADS>>>(kd, h->rcos, h->rsin, B * H, S, hd, DC_ROT);
    attn_dev(h->cublas, ar, ob, q,  k,  v, B * H, S, S, hd, scores, 0);
    attn_dev(h->cublas, ar, od, qd, kd, v, B * H, S, S, hd, scores, 0);
    k_subtract<<<nblocks((size_t)NH), THREADS>>>(ob, od, (size_t)NH);
    k_merge_heads_b<<<nblocks((size_t)NH), THREADS>>>(merged, ob, B, S, H, hd);
    gemm_f16w(h->cublas, ar, o, merged, w->to_out, NULL, N, D, D);
    k_add<<<nblocks((size_t)N * D), THREADS>>>(seq, res, o, (size_t)N * D);

    /* SwiGLU feed-forward */
    cudaMemcpy(res, seq, (size_t)N * D * sizeof(float), cudaMemcpyDeviceToDevice);
    k_dyt<<<nblocks((size_t)N * D), THREADS>>>(hh, seq, w->ff_alpha, w->ff_gamma, w->ff_beta, (size_t)N * D, D);
    float *proj = da_alloc(ar, (size_t)N * 2 * DC_INNER), *gated = da_alloc(ar, (size_t)N * DC_INNER);
    gemm_f16w(h->cublas, ar, proj, hh, w->ff_in_w, w->ff_in_b, N, D, 2 * DC_INNER);
    k_ff_silugate<<<nblocks((size_t)N * DC_INNER), THREADS>>>(gated, proj, N, DC_INNER);
    gemm_f16w(h->cublas, ar, o, gated, w->ff_out_w, w->ff_out_b, N, DC_INNER, D);
    k_add<<<nblocks((size_t)N * D), THREADS>>>(seq, res, o, (size_t)N * D);

    da_restore(ar, mark);
}

/* run 3 blocks over seq[L,768] in S-chunks; shift=1 adds the midpoint halo */
static void taae_pass_dev(aria_cuda_dec *h, float *seq, int L, const dec_blk_dev *blocks, int shift, darena *ar) {
    int S = DC_S, D = DC_D, half = S / 2;
    if (!shift) {
        int B = L / S;
        for (int b = 0; b < 3; b++) taae_block_dev(h, seq, B, &blocks[b], ar);
    } else {
        int Lp = L + S, B = Lp / S;
        size_t mark = da_save(ar);
        float *pad = da_alloc(ar, (size_t)Lp * D);
        cudaMemcpy(pad, seq, (size_t)half * D * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(pad + (size_t)half * D, seq, (size_t)L * D * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(pad + (size_t)(half + L) * D, seq + (size_t)(L - half) * D, (size_t)half * D * sizeof(float), cudaMemcpyDeviceToDevice);
        for (int b = 0; b < 3; b++) taae_block_dev(h, pad, B, &blocks[b], ar);
        cudaMemcpy(seq, pad + (size_t)half * D, (size_t)L * D * sizeof(float), cudaMemcpyDeviceToDevice);
        da_restore(ar, mark);
    }
}

extern "C" aria_cuda_dec *aria_cuda_dec_create(const aria_sa3_dec_view *v) {
    if (!aria_cuda_available()) return NULL;
    aria_cuda_dec *h = (aria_cuda_dec *)calloc(1, sizeof(aria_cuda_dec));
    if (cublasCreate(&h->cublas) != CUBLAS_STATUS_SUCCESS) { free(h); return NULL; }
    h->running_std = v->running_std;
    h->proj_w     = upload_f16(v->proj_w, (size_t)DC_D * 256);
    h->proj_b     = upload_f32(v->proj_b, DC_D);
    h->new_tokens = upload_f32(v->new_tokens, DC_D);
    h->mapping_w  = upload_f32(v->mapping_w, (size_t)DC_OUT * DC_D * 3);
    h->mapping_b  = upload_f32(v->mapping_b, DC_OUT);
    int ok = h->proj_w && h->proj_b && h->new_tokens && h->mapping_w && h->mapping_b;
    for (int i = 0; i < 6 && ok; i++) {
        const aria_taae_block_view *s = &v->blocks[i]; dec_blk_dev *d = &h->blocks[i];
        d->pre_alpha = s->pre_alpha; d->qn_alpha = s->qn_alpha; d->kn_alpha = s->kn_alpha; d->ff_alpha = s->ff_alpha;
        d->pre_gamma = upload_f32(s->pre_gamma, DC_D); d->pre_beta = upload_f32(s->pre_beta, DC_D);
        d->qn_gamma  = upload_f32(s->qn_gamma, DC_HD); d->qn_beta  = upload_f32(s->qn_beta, DC_HD);
        d->kn_gamma  = upload_f32(s->kn_gamma, DC_HD); d->kn_beta  = upload_f32(s->kn_beta, DC_HD);
        d->ff_gamma  = upload_f32(s->ff_gamma, DC_D);  d->ff_beta  = upload_f32(s->ff_beta, DC_D);
        d->ff_in_b   = upload_f32(s->ff_in_b, 2 * DC_INNER); d->ff_out_b = upload_f32(s->ff_out_b, DC_D);
        d->to_qkv    = upload_f16(s->to_qkv, (size_t)DC_QKV * DC_D);
        d->to_out    = upload_f16(s->to_out, (size_t)DC_D * DC_D);
        d->ff_in_w   = upload_f16(s->ff_in_w, (size_t)2 * DC_INNER * DC_D);
        d->ff_out_w  = upload_f16(s->ff_out_w, (size_t)DC_D * DC_INNER);
        ok = d->to_qkv && d->to_out && d->ff_in_w && d->ff_out_w && d->pre_gamma && d->ff_in_b;
    }
    /* rope tables for S=34 (computed on host, uploaded) */
    int rh = DC_ROT / 2;
    float *cs = (float *)malloc((size_t)DC_S * rh * sizeof(float)), *sn = (float *)malloc((size_t)DC_S * rh * sizeof(float));
    for (int i = 0; i < rh; i++) {
        double inv = 1.0 / pow(10000.0, (double)(2 * i) / (double)DC_ROT);
        for (int p = 0; p < DC_S; p++) { cs[p * rh + i] = (float)cos((double)p * inv); sn[p * rh + i] = (float)sin((double)p * inv); }
    }
    h->rcos = upload_f32(cs, (size_t)DC_S * rh); h->rsin = upload_f32(sn, (size_t)DC_S * rh);
    free(cs); free(sn);
    if (!ok || !h->rcos) { aria_cuda_dec_free(h); return NULL; }
    return h;
}

extern "C" void aria_cuda_dec_forward(aria_cuda_dec *h, float *audio, const float *latent, int T) {
    int D = DC_D, Tp = (T % 2 == 0) ? T : T + 1, L = Tp * 17, Lout = Tp * 16, Lt = T * 16;
    /* arena sized for the batched pass (shifted pass is the largest) */
    int Bmax = (L + DC_S) / DC_S;
    size_t bs = (size_t)Bmax * DC_S;
    size_t cap = (bs * 19200 + (size_t)Bmax * DC_H * DC_S * DC_S        /* one block's peak */
                  + (size_t)(L + DC_S) * D * 2                          /* seq + pad */
                  + (size_t)D * Lout + (size_t)DC_OUT * Lout            /* feat + mapped */
                  + (size_t)T * 256 + (size_t)T * D + (1u << 20)) * sizeof(float);
    darena ar; cudaMalloc(&ar.base, cap); ar.cap = cap; ar.used = 0;

    float *d_lat = upload_f32(latent, (size_t)256 * T);
    size_t mark = da_save(&ar);
    /* softnorm: z = latent * running_std ; transpose [256,T]->[T,256] ; proj ->[T,768] */
    float *z = da_alloc(&ar, (size_t)256 * T);
    k_scale<<<nblocks((size_t)256 * T), THREADS>>>(z, d_lat, h->running_std, 256 * T);
    float *ztc = da_alloc(&ar, (size_t)T * 256);
    k_transpose<<<nblocks((size_t)256 * T), THREADS>>>(ztc, z, 256, T);
    float *x = da_alloc(&ar, (size_t)T * D);
    gemm_f16w(h->cublas, &ar, x, ztc, h->proj_w, h->proj_b, T, 256, D);

    /* expand to seq[L,768], two chunked passes (unshifted, then midpoint-shift) */
    float *seq = da_alloc(&ar, (size_t)L * D);
    k_expand_tokens<<<nblocks((size_t)L * D), THREADS>>>(seq, x, h->new_tokens, T, Tp, D);
    taae_pass_dev(h, seq, L, &h->blocks[0], 0, &ar);
    taae_pass_dev(h, seq, L, &h->blocks[3], 1, &ar);

    /* extract feat[768,Lout], conv1d mapping 768->512, trim to T*16, unpatch -> audio */
    float *feat = da_alloc(&ar, (size_t)D * Lout);
    k_extract_feat<<<nblocks((size_t)D * Lout), THREADS>>>(feat, seq, D, Lout);
    float *mapped = da_alloc(&ar, (size_t)DC_OUT * Lout);
    k_conv1d<<<nblocks((size_t)DC_OUT * Lout), THREADS>>>(mapped, feat, h->mapping_w, h->mapping_b, D, DC_OUT, 3, 1, Lout);
    float *dec = da_alloc(&ar, (size_t)DC_OUT * Lt);
    CK(cudaMemcpy2D(dec, (size_t)Lt * sizeof(float), mapped, (size_t)Lout * sizeof(float),
                    (size_t)Lt * sizeof(float), DC_OUT, cudaMemcpyDeviceToDevice));
    float *d_audio = da_alloc(&ar, (size_t)2 * Lt * 256);
    k_unpatch<<<nblocks((size_t)2 * Lt * 256), THREADS>>>(d_audio, dec, Lt);
    CK(cudaGetLastError());
    CK(cudaMemcpy(audio, d_audio, (size_t)2 * Lt * 256 * sizeof(float), cudaMemcpyDeviceToHost));

    da_restore(&ar, mark);
    cudaFree(d_lat); cudaFree(ar.base);
}

extern "C" void aria_cuda_dec_free(aria_cuda_dec *h) {
    if (!h) return;
    if (h->cublas) cublasDestroy(h->cublas);
    cudaFree(h->proj_w); cudaFree((void *)h->proj_b); cudaFree((void *)h->new_tokens);
    cudaFree(h->mapping_w); cudaFree((void *)h->mapping_b); cudaFree(h->rcos); cudaFree(h->rsin);
    for (int i = 0; i < 6; i++) {
        dec_blk_dev *d = &h->blocks[i];
        cudaFree((void *)d->pre_gamma); cudaFree((void *)d->pre_beta);
        cudaFree((void *)d->qn_gamma); cudaFree((void *)d->qn_beta);
        cudaFree((void *)d->kn_gamma); cudaFree((void *)d->kn_beta);
        cudaFree((void *)d->ff_gamma); cudaFree((void *)d->ff_beta);
        cudaFree((void *)d->ff_in_b); cudaFree((void *)d->ff_out_b);
        cudaFree(d->to_qkv); cudaFree(d->to_out); cudaFree(d->ff_in_w); cudaFree(d->ff_out_w);
    }
    free(h);
}

/* ===================== device-resident MEDIUM taae decoder =====================
 * dim 1536 / depth 12; the whole token sequence (T*17) runs through differential
 * DyT blocks with SLIDING-WINDOW (banded) attention (window [17,17]); blocks where
 * (12-i) < 8 use a sinusoidal FF. Weights fp16 (GEMMs) / fp32 (norms, conv). */
#define MD_D     1536
#define MD_H     24
#define MD_HD    64
#define MD_ROT   32
#define MD_INNER 4608
#define MD_QKV   7680   /* 5*1536 */
#define MD_OUT   512
#define MD_DEPTH 12
#define MD_WIN   17

/* Fused sliding-window attention: one warp per query (h,i) computes only the
 * [i-W, i+W] band (<= 2W+1 keys) -- QK via shuffle-reduce, softmax over the band in
 * shared, then AV. O(N*(2W+1)) instead of the full O(N^2)-then-mask, and no N*N score
 * buffer. Exact vs the masked-softmax version (masked keys contributed 0 anyway).
 * D must be a multiple of 32 (decoder hd=64 -> 2 dims/lane). */
__global__ void k_attn_band(float *out, const float *q, const float *k, const float *v,
                            int H, int N, int D, float scale, int W) {
    __shared__ float ssc[2 * MD_WIN + 1];
    int idx = blockIdx.x; if (idx >= H * N) return;
    int h = idx / N, i = idx % N, lane = threadIdx.x, dpl = D / 32;
    const float *qh = q + (size_t)idx * D;
    float qr[8];
    for (int c = 0; c < dpl; c++) qr[c] = qh[lane + c * 32];
    int lo = i - W < 0 ? 0 : i - W, hi = i + W >= N ? N - 1 : i + W, wb = hi - lo + 1;
    for (int j = 0; j < wb; j++) {
        const float *kk = k + ((size_t)h * N + (lo + j)) * D;
        float p = 0.0f;
        for (int c = 0; c < dpl; c++) p += qr[c] * kk[lane + c * 32];
        for (int o = 16; o > 0; o >>= 1) p += __shfl_down_sync(0xffffffffu, p, o);
        if (lane == 0) ssc[j] = p * scale;
    }
    __syncwarp();
    if (lane == 0) {                       /* softmax over the band (<= 35 keys) */
        float mx = -INFINITY;
        for (int j = 0; j < wb; j++) mx = fmaxf(mx, ssc[j]);
        float sum = 0.0f;
        for (int j = 0; j < wb; j++) { ssc[j] = expf(ssc[j] - mx); sum += ssc[j]; }
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int j = 0; j < wb; j++) ssc[j] *= inv;
    }
    __syncwarp();
    float orr[8];
    for (int c = 0; c < dpl; c++) orr[c] = 0.0f;
    for (int j = 0; j < wb; j++) {
        const float *vv = v + ((size_t)h * N + (lo + j)) * D;
        for (int c = 0; c < dpl; c++) orr[c] += ssc[j] * vv[lane + c * 32];
    }
    float *oo = out + (size_t)idx * D;
    for (int c = 0; c < dpl; c++) oo[lane + c * 32] = orr[c];
}

/* sinusoidal GLU gate: out = sin(pi*gate) * value */
__global__ void k_ff_singate(float *out, const float *proj, int S, int inner) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)S * inner) return;
    int s = (int)(idx / inner), i = (int)(idx % inner);
    const float *pr = proj + (size_t)s * 2 * inner;
    out[idx] = sinf(3.14159265359f * pr[inner + i]) * pr[i];
}

/* sliding-window self-attention (Nq=Nk=N), fused band kernel: one warp per query */
static void attn_dev_band(float *out, const float *q, const float *k, const float *v,
                          int H, int N, int D, int W) {
    k_attn_band<<<H * N, 32>>>(out, q, k, v, H, N, D, 1.0f / sqrtf((float)D), W);
}

struct aria_cuda_dec_medium {
    float running_std;
    __half *proj_w; const float *proj_b, *new_tokens, *mapping_b;
    float *mapping_w;       /* fp32 k1 [512,1536] */
    dec_blk_dev blocks[MD_DEPTH];
    cublasHandle_t cublas;
};

static void med_block_dev(aria_cuda_dec_medium *h, float *seq, int N, const float *rcos, const float *rsin,
                          const dec_blk_dev *w, int sinusoidal, darena *ar) {
    int D = MD_D, H = MD_H, hd = MD_HD;
    size_t NH = (size_t)H * N * hd;
    size_t mark = da_save(ar);
    /* res/hh/o persist across the block; the attention scratch is freed before the FF
     * (they're never live together) so the arena peak is the attention phase, not the
     * sum of both -- saves ~3*N*INNER (~100 MB on a 10 s medium clip). */
    float *res = da_alloc(ar, (size_t)N * D), *hh = da_alloc(ar, (size_t)N * D), *o = da_alloc(ar, (size_t)N * D);

    cudaMemcpy(res, seq, (size_t)N * D * sizeof(float), cudaMemcpyDeviceToDevice);
    k_dyt<<<nblocks((size_t)N * D), THREADS>>>(hh, seq, w->pre_alpha, w->pre_gamma, w->pre_beta, (size_t)N * D, D);
    size_t amark = da_save(ar);
    float *q  = da_alloc(ar, NH), *k = da_alloc(ar, NH), *v = da_alloc(ar, NH);
    float *qd = da_alloc(ar, NH), *kd = da_alloc(ar, NH);
    float *ob = da_alloc(ar, NH), *od = da_alloc(ar, NH), *merged = da_alloc(ar, (size_t)N * D);
    float *qkv = da_alloc(ar, (size_t)N * MD_QKV);
    gemm_f16w(h->cublas, ar, qkv, hh, w->to_qkv, NULL, N, D, MD_QKV);
    k_extract_heads<<<nblocks(NH), THREADS>>>(q,  qkv, N, H, hd, MD_QKV, 0);
    k_extract_heads<<<nblocks(NH), THREADS>>>(k,  qkv, N, H, hd, MD_QKV, D);
    k_extract_heads<<<nblocks(NH), THREADS>>>(v,  qkv, N, H, hd, MD_QKV, 2 * D);
    k_extract_heads<<<nblocks(NH), THREADS>>>(qd, qkv, N, H, hd, MD_QKV, 3 * D);
    k_extract_heads<<<nblocks(NH), THREADS>>>(kd, qkv, N, H, hd, MD_QKV, 4 * D);
    k_dyt<<<nblocks(NH), THREADS>>>(q,  q,  w->qn_alpha, w->qn_gamma, w->qn_beta, NH, hd);
    k_dyt<<<nblocks(NH), THREADS>>>(qd, qd, w->qn_alpha, w->qn_gamma, w->qn_beta, NH, hd);
    k_dyt<<<nblocks(NH), THREADS>>>(k,  k,  w->kn_alpha, w->kn_gamma, w->kn_beta, NH, hd);
    k_dyt<<<nblocks(NH), THREADS>>>(kd, kd, w->kn_alpha, w->kn_gamma, w->kn_beta, NH, hd);
    size_t hn = (size_t)H * N;
    k_rope<<<nblocks(hn), THREADS>>>(q,  rcos, rsin, H, N, hd, MD_ROT);
    k_rope<<<nblocks(hn), THREADS>>>(qd, rcos, rsin, H, N, hd, MD_ROT);
    k_rope<<<nblocks(hn), THREADS>>>(k,  rcos, rsin, H, N, hd, MD_ROT);
    k_rope<<<nblocks(hn), THREADS>>>(kd, rcos, rsin, H, N, hd, MD_ROT);
    attn_dev_band(ob, q,  k,  v, H, N, hd, MD_WIN);
    attn_dev_band(od, qd, kd, v, H, N, hd, MD_WIN);
    k_subtract<<<nblocks(NH), THREADS>>>(ob, od, NH);
    k_merge_heads<<<nblocks(NH), THREADS>>>(merged, ob, N, H, hd);
    gemm_f16w(h->cublas, ar, o, merged, w->to_out, NULL, N, D, D);
    k_add<<<nblocks((size_t)N * D), THREADS>>>(seq, res, o, (size_t)N * D);
    da_restore(ar, amark);   /* free q/k/v/qd/kd/ob/od/merged/qkv before the FF */

    cudaMemcpy(res, seq, (size_t)N * D * sizeof(float), cudaMemcpyDeviceToDevice);
    k_dyt<<<nblocks((size_t)N * D), THREADS>>>(hh, seq, w->ff_alpha, w->ff_gamma, w->ff_beta, (size_t)N * D, D);
    float *proj = da_alloc(ar, (size_t)N * 2 * MD_INNER), *gated = da_alloc(ar, (size_t)N * MD_INNER);
    gemm_f16w(h->cublas, ar, proj, hh, w->ff_in_w, w->ff_in_b, N, D, 2 * MD_INNER);
    if (sinusoidal) k_ff_singate<<<nblocks((size_t)N * MD_INNER), THREADS>>>(gated, proj, N, MD_INNER);
    else            k_ff_silugate<<<nblocks((size_t)N * MD_INNER), THREADS>>>(gated, proj, N, MD_INNER);
    gemm_f16w(h->cublas, ar, o, gated, w->ff_out_w, w->ff_out_b, N, MD_INNER, D);
    k_add<<<nblocks((size_t)N * D), THREADS>>>(seq, res, o, (size_t)N * D);

    da_restore(ar, mark);
}

extern "C" aria_cuda_dec_medium *aria_cuda_dec_medium_create(const aria_sa3_dec_medium_view *v) {
    if (!aria_cuda_available()) return NULL;
    aria_cuda_dec_medium *h = (aria_cuda_dec_medium *)calloc(1, sizeof(*h));
    if (cublasCreate(&h->cublas) != CUBLAS_STATUS_SUCCESS) { free(h); return NULL; }
    h->running_std = v->running_std;
    h->proj_w     = upload_f16(v->proj_w, (size_t)MD_D * 256);
    h->proj_b     = upload_f32(v->proj_b, MD_D);
    h->new_tokens = upload_f32(v->new_tokens, MD_D);
    h->mapping_w  = upload_f32(v->mapping_w, (size_t)MD_OUT * MD_D);   /* k1 */
    h->mapping_b  = upload_f32(v->mapping_b, MD_OUT);
    int ok = h->proj_w && h->proj_b && h->new_tokens && h->mapping_w && h->mapping_b;
    for (int i = 0; i < MD_DEPTH && ok; i++) {
        const aria_taae_block_view *s = &v->blocks[i]; dec_blk_dev *d = &h->blocks[i];
        d->pre_alpha = s->pre_alpha; d->qn_alpha = s->qn_alpha; d->kn_alpha = s->kn_alpha; d->ff_alpha = s->ff_alpha;
        d->pre_gamma = upload_f32(s->pre_gamma, MD_D); d->pre_beta = upload_f32(s->pre_beta, MD_D);
        d->qn_gamma  = upload_f32(s->qn_gamma, MD_HD); d->qn_beta  = upload_f32(s->qn_beta, MD_HD);
        d->kn_gamma  = upload_f32(s->kn_gamma, MD_HD); d->kn_beta  = upload_f32(s->kn_beta, MD_HD);
        d->ff_gamma  = upload_f32(s->ff_gamma, MD_D);  d->ff_beta  = upload_f32(s->ff_beta, MD_D);
        d->ff_in_b   = upload_f32(s->ff_in_b, 2 * MD_INNER); d->ff_out_b = upload_f32(s->ff_out_b, MD_D);
        d->to_qkv    = upload_f16(s->to_qkv, (size_t)MD_QKV * MD_D);
        d->to_out    = upload_f16(s->to_out, (size_t)MD_D * MD_D);
        d->ff_in_w   = upload_f16(s->ff_in_w, (size_t)2 * MD_INNER * MD_D);
        d->ff_out_w  = upload_f16(s->ff_out_w, (size_t)MD_D * MD_INNER);
        ok = d->to_qkv && d->to_out && d->ff_in_w && d->ff_out_w && d->pre_gamma && d->ff_in_b;
    }
    if (!ok) { aria_cuda_dec_medium_free(h); return NULL; }
    return h;
}

extern "C" void aria_cuda_dec_medium_forward(aria_cuda_dec_medium *h, float *audio, const float *latent, int T) {
    int D = MD_D, N = T * 17, Lt = T * 16, rh = MD_ROT / 2;
    /* rope tables for N (host -> device) */
    float *cs = (float *)malloc((size_t)N * rh * sizeof(float)), *sn = (float *)malloc((size_t)N * rh * sizeof(float));
    for (int i = 0; i < rh; i++) {
        double inv = 1.0 / pow(10000.0, (double)(2 * i) / (double)MD_ROT);
        for (int p = 0; p < N; p++) { cs[(size_t)p * rh + i] = (float)cos((double)p * inv); sn[(size_t)p * rh + i] = (float)sin((double)p * inv); }
    }
    float *rcos = upload_f32(cs, (size_t)N * rh), *rsin = upload_f32(sn, (size_t)N * rh);
    free(cs); free(sn);

    /* no N*N score buffer (fused band attn), and the FF scratch no longer co-resides
     * with the attention scratch (freed between phases) -- peak is the attention phase
     * (16*N*D + a GEMM-scratch margin); 17*N*D bounds it and the FF (3.5*N*INNER < 17*D). */
    size_t blockpk = (size_t)17 * N * D;
    size_t cap = ((size_t)N * D + blockpk + (size_t)D * Lt + (size_t)MD_OUT * Lt
                  + (size_t)T * 256 + (size_t)T * D + (size_t)2 * Lt * 256 + (1u << 20)) * sizeof(float);
    darena ar; cudaMalloc(&ar.base, cap); ar.cap = cap; ar.used = 0;

    float *d_lat = upload_f32(latent, (size_t)256 * T);
    size_t mark = da_save(&ar);
    float *z = da_alloc(&ar, (size_t)256 * T);
    k_scale<<<nblocks((size_t)256 * T), THREADS>>>(z, d_lat, h->running_std, 256 * T);
    float *ztc = da_alloc(&ar, (size_t)T * 256);
    k_transpose<<<nblocks((size_t)256 * T), THREADS>>>(ztc, z, 256, T);
    float *x = da_alloc(&ar, (size_t)T * D);
    gemm_f16w(h->cublas, &ar, x, ztc, h->proj_w, h->proj_b, T, 256, D);

    /* seq[N, 1536]: token t at row t*17, new_tokens at the next 16 (no even-pad) */
    float *seq = da_alloc(&ar, (size_t)N * D);
    k_expand_tokens<<<nblocks((size_t)N * D), THREADS>>>(seq, x, h->new_tokens, T, T, D);
    for (int b = 0; b < MD_DEPTH; b++)
        med_block_dev(h, seq, N, rcos, rsin, &h->blocks[b], (MD_DEPTH - b) < 8, &ar);

    float *feat = da_alloc(&ar, (size_t)D * Lt);
    k_extract_feat<<<nblocks((size_t)D * Lt), THREADS>>>(feat, seq, D, Lt);
    float *mapped = da_alloc(&ar, (size_t)MD_OUT * Lt);
    k_conv1d<<<nblocks((size_t)MD_OUT * Lt), THREADS>>>(mapped, feat, h->mapping_w, h->mapping_b, D, MD_OUT, 1, 0, Lt);
    float *d_audio = da_alloc(&ar, (size_t)2 * Lt * 256);
    k_unpatch<<<nblocks((size_t)2 * Lt * 256), THREADS>>>(d_audio, mapped, Lt);
    CK(cudaGetLastError());
    CK(cudaMemcpy(audio, d_audio, (size_t)2 * Lt * 256 * sizeof(float), cudaMemcpyDeviceToHost));

    da_restore(&ar, mark);
    cudaFree(d_lat); cudaFree(rcos); cudaFree(rsin); cudaFree(ar.base);
}

extern "C" void aria_cuda_dec_medium_free(aria_cuda_dec_medium *h) {
    if (!h) return;
    if (h->cublas) cublasDestroy(h->cublas);
    cudaFree(h->proj_w); cudaFree((void *)h->proj_b); cudaFree((void *)h->new_tokens);
    cudaFree(h->mapping_w); cudaFree((void *)h->mapping_b);
    for (int i = 0; i < MD_DEPTH; i++) {
        dec_blk_dev *d = &h->blocks[i];
        cudaFree((void *)d->pre_gamma); cudaFree((void *)d->pre_beta);
        cudaFree((void *)d->qn_gamma); cudaFree((void *)d->qn_beta);
        cudaFree((void *)d->kn_gamma); cudaFree((void *)d->kn_beta);
        cudaFree((void *)d->ff_gamma); cudaFree((void *)d->ff_beta);
        cudaFree((void *)d->ff_in_b); cudaFree((void *)d->ff_out_b);
        cudaFree(d->to_qkv); cudaFree(d->to_out); cudaFree(d->ff_in_w); cudaFree(d->ff_out_w);
    }
    free(h);
}
