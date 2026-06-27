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
__global__ void k_adaln(float *y, const float *scale, const float *shift, int S, int dim) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)S * dim) return;
    int i = (int)(idx % dim);
    y[idx] = y[idx] * (1.0f + scale[i]) + shift[i];
}
__global__ void k_gate(float *y, const float *gate, int S, int dim) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)S * dim) return;
    int i = (int)(idx % dim);
    y[idx] *= 1.0f / (1.0f + expf(-(1.0f - gate[i])));
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

/* multi-head SDPA on device-resident fp32 activations (scale 1/sqrt(D)). */
static void attn_dev(float *out, const float *q, const float *k, const float *v,
                     int H, int Nq, int Nk, int D, float *scores) {
    float scale = 1.0f / sqrtf((float)D);
    dim3 sblk(TILE, TILE), sgrid((Nk + TILE - 1) / TILE, (Nq + TILE - 1) / TILE);
    dim3 oblk(TILE, TILE), ogrid((D + TILE - 1) / TILE, (Nq + TILE - 1) / TILE);
    for (int h = 0; h < H; h++) {
        const float *qh = q + (size_t)h * Nq * D, *kh = k + (size_t)h * Nk * D, *vh = v + (size_t)h * Nk * D;
        float *oh = out + (size_t)h * Nq * D;
        aria_gemm_nt<<<sgrid, sblk>>>(scores, qh, kh, NULL, Nq, D, Nk);
        k_scale<<<nblocks((size_t)Nq * Nk), THREADS>>>(scores, scores, scale, Nq * Nk);
        k_softmax<<<nblocks(Nq), THREADS>>>(scores, Nq, Nk, NULL);
        k_matmul<<<ogrid, oblk>>>(oh, scores, vh, Nq, Nk, D);
    }
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

struct blk_dev {
    const float *pre_norm, *cross_norm, *ff_norm, *sa_q_norm, *sa_k_norm, *ca_q_norm;
    const float *ssg, *ff_in_b, *ff_out_b;
    __half *sa_to_qkv, *sa_to_out, *ca_to_q, *ca_to_out, *ff_in_w, *ff_out_w;
};

struct aria_cuda_dit {
    int depth, ed, H, hd, inner, io_ch, n_mem, rot;
    __half *preprocess, *postprocess, *project_in, *project_out;
    float *memory_tokens;
    blk_dev *blocks;
    cublasHandle_t cublas;
    /* per-request */
    int T, S, n_cond, have_req;
    float *dx, *dv, *dgcond, *rope_cos, *rope_sin;
    float **cross_k, **cross_v;
    darena arena;
};

/* y[M,N] = x[M,K] @ W[N,K]^T + b via cuBLAS GemmEx (fp16 in, fp32 accumulate ->
 * tensor cores on sm_80+). x (fp32 activations) is converted to fp16 into transient
 * arena scratch; W is already fp16. Single default stream, so reusing the scratch
 * region after the (queued) GEMM is safe by stream ordering. */
static void gemm_f16w(aria_cuda_dit *h, float *y, const float *x, const __half *W,
                      const float *b, int M, int K, int N) {
    darena *ar = &h->arena; size_t mark = da_save(ar);
    __half *xh = da_alloc_half(ar, (size_t)M * K);
    k_f32_to_f16<<<nblocks((size_t)M * K), THREADS>>>(xh, x, (size_t)M * K);
    const float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(h->cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                 W, CUDA_R_16F, K, xh, CUDA_R_16F, K, &beta,
                 y, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (b) k_add_bias<<<nblocks((size_t)M * N), THREADS>>>(y, b, M, N);
    da_restore(ar, mark);
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
    float *scores = da_alloc(ar, (size_t)S * (S > Sc ? S : Sc));
    int nHShd = (int)nblocks((size_t)H * S * hd);

    /* self-attention */
    cudaMemcpy(residual, seq, (size_t)S * dim * sizeof(float), cudaMemcpyDeviceToDevice);
    k_rmsnorm<<<nblocks(S), THREADS>>>(hh, seq, w->pre_norm, S, dim, 1e-5f, 0);
    k_adaln<<<nSd, THREADS>>>(hh, scale_self, shift_self, S, dim);
    float *qkv = da_alloc(ar, (size_t)S * 3 * dim);
    gemm_f16w(h, qkv, hh, w->sa_to_qkv, NULL, S, dim, 3 * dim);
    k_extract_heads<<<nHShd, THREADS>>>(qh, qkv, S, H, hd, 3 * dim, 0);
    k_extract_heads<<<nHShd, THREADS>>>(kh, qkv, S, H, hd, 3 * dim, dim);
    k_extract_heads<<<nHShd, THREADS>>>(vh, qkv, S, H, hd, 3 * dim, 2 * dim);
    k_rmsnorm<<<nblocks((size_t)H * S), THREADS>>>(qh, qh, w->sa_q_norm, H * S, hd, 1e-6f, 0);
    k_rmsnorm<<<nblocks((size_t)H * S), THREADS>>>(kh, kh, w->sa_k_norm, H * S, hd, 1e-6f, 0);
    k_rope<<<nblocks((size_t)H * S), THREADS>>>(qh, h->rope_cos, h->rope_sin, H, S, hd, rot);
    k_rope<<<nblocks((size_t)H * S), THREADS>>>(kh, h->rope_cos, h->rope_sin, H, S, hd, rot);
    attn_dev(ao, qh, kh, vh, H, S, S, hd, scores);
    k_merge_heads<<<nHShd, THREADS>>>(merged, ao, S, H, hd);
    gemm_f16w(h, o, merged, w->sa_to_out, NULL, S, dim, dim);
    k_gate<<<nSd, THREADS>>>(o, gate_self, S, dim);
    k_add<<<nSd, THREADS>>>(seq, residual, o, (size_t)S * dim);

    /* cross-attention (cached K/V) */
    cudaMemcpy(residual, seq, (size_t)S * dim * sizeof(float), cudaMemcpyDeviceToDevice);
    k_rmsnorm<<<nblocks(S), THREADS>>>(hh, seq, w->cross_norm, S, dim, 1e-5f, 0);
    float *q = da_alloc(ar, (size_t)S * dim);
    gemm_f16w(h, q, hh, w->ca_to_q, NULL, S, dim, dim);
    k_extract_heads<<<nHShd, THREADS>>>(qh, q, S, H, hd, dim, 0);
    k_rmsnorm<<<nblocks((size_t)H * S), THREADS>>>(qh, qh, w->ca_q_norm, H * S, hd, 1e-6f, 0);
    attn_dev(ao, qh, h->cross_k[blk], h->cross_v[blk], H, S, Sc, hd, scores);
    k_merge_heads<<<nHShd, THREADS>>>(merged, ao, S, H, hd);
    gemm_f16w(h, o, merged, w->ca_to_out, NULL, S, dim, dim);
    k_add<<<nSd, THREADS>>>(seq, residual, o, (size_t)S * dim);

    /* feed-forward (GLU) */
    cudaMemcpy(residual, seq, (size_t)S * dim * sizeof(float), cudaMemcpyDeviceToDevice);
    k_rmsnorm<<<nblocks(S), THREADS>>>(hh, seq, w->ff_norm, S, dim, 1e-5f, 0);
    k_adaln<<<nSd, THREADS>>>(hh, scale_ff, shift_ff, S, dim);
    float *proj = da_alloc(ar, (size_t)S * 2 * inner), *gated = da_alloc(ar, (size_t)S * inner);
    gemm_f16w(h, proj, hh, w->ff_in_w, w->ff_in_b, S, dim, 2 * inner);
    k_ff_silugate<<<nblocks((size_t)S * inner), THREADS>>>(gated, proj, S, inner);
    gemm_f16w(h, o, gated, w->ff_out_w, w->ff_out_b, S, inner, dim);
    k_gate<<<nSd, THREADS>>>(o, gate_ff, S, dim);
    k_add<<<nSd, THREADS>>>(seq, residual, o, (size_t)S * dim);

    da_restore(ar, mark);
}

extern "C" aria_cuda_dit *aria_cuda_dit_create(const aria_sa3_dit_view *v) {
    if (!aria_cuda_available()) return NULL;
    int ed = v->ed, inner = v->inner;
    /* rough fit check: fp16 weights ~ depth * (12*ed*ed-ish) */
    size_t need = (size_t)v->depth * ((size_t)3 * ed * ed + 3 * (size_t)ed * ed + (size_t)2 * inner * ed
                  + (size_t)inner * ed) * sizeof(__half) + (64u << 20);
    size_t freeb = 0, totb = 0; cudaMemGetInfo(&freeb, &totb);
    if (freeb < need) { fprintf(stderr, "aria_cuda_dit: need ~%zu MiB, only %zu free\n", need >> 20, freeb >> 20); return NULL; }

    aria_cuda_dit *h = (aria_cuda_dit *)calloc(1, sizeof(aria_cuda_dit));
    h->depth = v->depth; h->ed = ed; h->H = v->num_heads; h->hd = v->head_dim;
    h->inner = inner; h->io_ch = v->io_ch; h->n_mem = v->n_mem; h->rot = v->rot_dim;
    if (cublasCreate(&h->cublas) != CUBLAS_STATUS_SUCCESS) { free(h); return NULL; }
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
        d->ssg        = upload_f32(s->to_scale_shift_gate, (size_t)6 * ed);
        d->ff_in_b    = upload_f32(s->ff_in_b, (size_t)2 * inner);
        d->ff_out_b   = upload_f32(s->ff_out_b, ed);
        d->sa_to_qkv  = upload_f16(s->sa_to_qkv, (size_t)3 * ed * ed);
        d->sa_to_out  = upload_f16(s->sa_to_out, (size_t)ed * ed);
        d->ca_to_q    = upload_f16(s->ca_to_q, (size_t)ed * ed);
        d->ca_to_out  = upload_f16(s->ca_to_out, (size_t)ed * ed);
        d->ff_in_w    = upload_f16(s->ff_in_w, (size_t)2 * inner * ed);
        d->ff_out_w   = upload_f16(s->ff_out_w, (size_t)ed * inner);
        ok = d->sa_to_qkv && d->sa_to_out && d->ca_to_q && d->ca_to_out && d->ff_in_w && d->ff_out_w
             && d->ssg && d->pre_norm;
    }
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
    free(h->cross_k); free(h->cross_v);
    h->cross_k = h->cross_v = NULL; h->dx = h->dv = h->dgcond = NULL;
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
    h->cross_k = (float **)calloc(h->depth, sizeof(float *));
    h->cross_v = (float **)calloc(h->depth, sizeof(float *));
    for (int b = 0; b < h->depth; b++) {
        h->cross_k[b] = upload_f32(rv->cross_k[b], (size_t)H * n_cond * hd);
        h->cross_v[b] = upload_f32(rv->cross_v[b], (size_t)H * n_cond * hd);
    }
    /* device arena: one block's peak + step buffers (mirrors the CPU sizing) */
    size_t block_floats = (size_t)6 * ed + 24 * (size_t)S * ed + (size_t)S * (S > n_cond ? S : n_cond);
    size_t step_floats = (size_t)S * ed + 8 * (size_t)T * C + 8 * (size_t)ed;
    h->arena.cap = (block_floats + step_floats + (1u << 20)) * sizeof(float);
    h->arena.used = 0;
    cudaMalloc(&h->arena.base, h->arena.cap);
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
    gemm_f16w(h, pre, xtc, h->preprocess, NULL, T, C, C);
    k_add<<<nblocks((size_t)T * C), THREADS>>>(xtc, xtc, pre, (size_t)T * C);

    float *seq = da_alloc(ar, (size_t)S * ed);
    cudaMemcpy(seq, h->memory_tokens, (size_t)Mt * ed * sizeof(float), cudaMemcpyDeviceToDevice);
    gemm_f16w(h, seq + (size_t)Mt * ed, xtc, h->project_in, NULL, T, C, ed);

    for (int b = 0; b < h->depth; b++) dit_block_dev(h, seq, b);

    float *outtc = da_alloc(ar, (size_t)T * C);
    gemm_f16w(h, outtc, seq + (size_t)Mt * ed, h->project_out, NULL, T, ed, C);
    float *postc = da_alloc(ar, (size_t)T * C);
    gemm_f16w(h, postc, outtc, h->postprocess, NULL, T, C, C);
    k_add<<<nblocks((size_t)T * C), THREADS>>>(outtc, outtc, postc, (size_t)T * C);
    k_transpose<<<nblocks((size_t)T * C), THREADS>>>(h->dv, outtc, T, C);  /* [T,C]->[C,T] */

    da_restore(ar, mark);
    cudaMemcpy(v_CT, h->dv, (size_t)C * T * sizeof(float), cudaMemcpyDeviceToHost);
}

extern "C" void aria_cuda_dit_free(aria_cuda_dit *h) {
    if (!h) return;
    if (h->cublas) cublasDestroy(h->cublas);
    cudaFree(h->preprocess); cudaFree(h->postprocess); cudaFree(h->project_in); cudaFree(h->project_out);
    cudaFree(h->memory_tokens);
    if (h->blocks) for (int b = 0; b < h->depth; b++) {
        blk_dev *d = &h->blocks[b];
        cudaFree((void *)d->pre_norm); cudaFree((void *)d->cross_norm); cudaFree((void *)d->ff_norm);
        cudaFree((void *)d->sa_q_norm); cudaFree((void *)d->sa_k_norm); cudaFree((void *)d->ca_q_norm);
        cudaFree((void *)d->ssg); cudaFree((void *)d->ff_in_b); cudaFree((void *)d->ff_out_b);
        cudaFree(d->sa_to_qkv); cudaFree(d->sa_to_out); cudaFree(d->ca_to_q);
        cudaFree(d->ca_to_out); cudaFree(d->ff_in_w); cudaFree(d->ff_out_w);
    }
    free(h->blocks);
    free_request(h);
    free(h);
}
