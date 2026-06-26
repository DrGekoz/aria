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

/* ---- rmsnorm / gemma_rmsnorm (one thread per row, fp64 accum) ---- */
__global__ void k_rmsnorm(float *y, const float *x, const float *w, int rows, int dim, float eps, int gemma) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float *xr = x + (size_t)r * dim; float *yr = y + (size_t)r * dim;
    double ss = 0.0;
    for (int i = 0; i < dim; i++) ss += (double)xr[i] * xr[i];
    float inv = (float)(1.0 / sqrt(ss / dim + eps));
    for (int i = 0; i < dim; i++) {
        float g = w ? (gemma ? 1.0f + w[i] : w[i]) : 1.0f;
        yr[i] = xr[i] * inv * g;
    }
}
extern "C" void aria_cuda_rmsnorm(float *y, const float *x, const float *w, int rows, int dim, float eps) {
    float *dx = d_in(x, (size_t)rows * dim), *dw = w ? d_in(w, dim) : NULL, *dy = d_new((size_t)rows * dim);
    k_rmsnorm<<<nblocks(rows), THREADS>>>(dy, dx, dw, rows, dim, eps, 0); CK(cudaGetLastError());
    cudaFree(dx); if (dw) cudaFree(dw); d_out(y, dy, (size_t)rows * dim);
}
extern "C" void aria_cuda_gemma_rmsnorm(float *y, const float *x, const float *w, int rows, int dim, float eps) {
    float *dx = d_in(x, (size_t)rows * dim), *dw = d_in(w, dim), *dy = d_new((size_t)rows * dim);
    k_rmsnorm<<<nblocks(rows), THREADS>>>(dy, dx, dw, rows, dim, eps, 1); CK(cudaGetLastError());
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

/* ---- row softmax (one thread per row), optional additive mask ---- */
__global__ void k_softmax(float *x, int rows, int cols, const float *mask) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float *xr = x + (size_t)r * cols; const float *mr = mask ? mask + (size_t)r * cols : NULL;
    float mx = -INFINITY;
    for (int i = 0; i < cols; i++) { float v = xr[i] + (mr ? mr[i] : 0.0f); xr[i] = v; if (v > mx) mx = v; }
    float s = 0.0f;
    for (int i = 0; i < cols; i++) { float e = expf(xr[i] - mx); xr[i] = e; s += e; }
    float inv = s > 0.0f ? 1.0f / s : 0.0f;
    for (int i = 0; i < cols; i++) xr[i] *= inv;
}
__global__ void k_scale(float *y, const float *x, float a, int n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < (size_t)n) y[i] = x[i] * a;
}
extern "C" void aria_cuda_softmax(float *x, int rows, int cols, const float *mask) {
    float *dx = d_in(x, (size_t)rows * cols), *dm = mask ? d_in(mask, (size_t)rows * cols) : NULL;
    k_softmax<<<nblocks(rows), THREADS>>>(dx, rows, cols, dm); CK(cudaGetLastError());
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
        k_softmax<<<nblocks(Nq), THREADS>>>(dscores, Nq, Nk, dmask);
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
