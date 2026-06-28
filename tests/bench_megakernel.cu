/* bench_megakernel.cu - True-Megakernel feasibility benchmark (PAPER.md Direction 1).
 *
 * Question: can a hand-rolled WMMA tensor-core GEMM (+ on-chip fusion) beat cuBLAS at the
 * SA3 DiT's short-sequence GEMM shapes on Ampere (RTX 3070, sm_86)? PAPER.md predicts no:
 * the per-block weights (10MB+) must stream from HBM regardless, so the GEMMs are weight-
 * bound and a megakernel only saves the activation round-trips. This measures it.
 *
 * Stage 1 here: WMMA GEMM vs cuBLAS GemmEx at the FFN shapes (y = x @ W^T, fp16 in / fp32
 * accumulate, matching aria's gemm_f16w). Compute time + GFLOP/s + effective HBM GB/s.
 *
 * build (alienware): /usr/local/cuda/bin/nvcc -arch=sm_86 -O3 -lcublas tests/bench_megakernel.cu -o build/bench_mk
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("cuda err %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)

/* ---- cuBLAS baseline: y[M,N] = x[M,K] @ W[N,K]^T, fp16 in, fp32 out (mirrors gemm_f16w) ---- */
__global__ void k_f32_to_f16(__half *o, const float *x, size_t n){ size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n)o[i]=__float2half(x[i]); }

static void gemm_cublas(cublasHandle_t cb, float *y, const __half *xh, const __half *W, int M, int K, int N){
    const float a=1.f,b=0.f;
    cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &a,
                 W, CUDA_R_16F, K, xh, CUDA_R_16F, K, &b,
                 y, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

/* ---- hand-rolled WMMA tiled GEMM: y[M,N] = x[M,K] @ W[N,K]^T ----
 * block tile 64x64, 4 warps (2x2), each warp a 2x2 grid of 16x16 fragments. fp16 in,
 * fp32 accumulate. A=x row-major; B=W loaded col-major to realize W^T (= A @ B^T). */
using namespace nvcuda;
#define WM 16
#define BM 64
#define BN 64
#define BK 16
__global__ void gemm_wmma(float *__restrict__ y, const __half *__restrict__ x,
                          const __half *__restrict__ W, int M, int K, int N){
    int bm0 = blockIdx.y * BM, bn0 = blockIdx.x * BN;
    int warp = threadIdx.x / 32;                 /* 0..3 */
    int wr = warp / 2, wc = warp % 2;            /* warp's 32x32 sub-tile */
    int tid = threadIdx.x;                        /* 0..127 */
    __shared__ __half As[BM][BK], Bs[BN][BK];

    wmma::fragment<wmma::accumulator, WM, WM, WM, float> acc[2][2];
    for (int i=0;i<2;i++) for(int j=0;j<2;j++) wmma::fill_fragment(acc[i][j], 0.f);

    for (int k0=0; k0<K; k0+=BK) {
        for (int i=tid; i<BM*BK; i+=128){ int r=i/BK,c=i%BK,m=bm0+r;
            As[r][c] = (m<M) ? x[(size_t)m*K + k0+c] : __float2half(0.f); }
        for (int i=tid; i<BN*BK; i+=128){ int r=i/BK,c=i%BK,n=bn0+r;
            Bs[r][c] = (n<N) ? W[(size_t)n*K + k0+c] : __float2half(0.f); }
        __syncthreads();
        for (int fm=0; fm<2; fm++){
            wmma::fragment<wmma::matrix_a, WM,WM,WM, __half, wmma::row_major> af;
            wmma::load_matrix_sync(af, &As[wr*32 + fm*16][0], BK);
            for (int fn=0; fn<2; fn++){
                wmma::fragment<wmma::matrix_b, WM,WM,WM, __half, wmma::col_major> bf;
                wmma::load_matrix_sync(bf, &Bs[wc*32 + fn*16][0], BK);
                wmma::mma_sync(acc[fm][fn], af, bf, acc[fm][fn]);
            }
        }
        __syncthreads();
    }
    for (int fm=0; fm<2; fm++) for (int fn=0; fn<2; fn++){
        int m0 = bm0 + wr*32 + fm*16, n0 = bn0 + wc*32 + fn*16;
        if (m0 < M)  /* N is a multiple of 64 here; guard M only (store row-by-row) */
            wmma::store_matrix_sync(&y[(size_t)m0*N + n0], acc[fm][fn], N, wmma::mem_row_major);
    }
}

static double bench(const char *tag, void (*fn)(void*), void *ctx, int M, int K, int N){
    cudaEvent_t evs,eve; CK(cudaEventCreate(&evs)); CK(cudaEventCreate(&eve));
    for(int w=0;w<10;w++) fn(ctx);             /* warmup */
    CK(cudaDeviceSynchronize());
    int reps=200; CK(cudaEventRecord(evs));
    for(int r=0;r<reps;r++) fn(ctx);
    CK(cudaEventRecord(eve)); CK(cudaEventSynchronize(eve));
    float ms=0; CK(cudaEventElapsedTime(&ms,evs,eve)); double dt=ms/1e3/reps;
    double gf = 2.0*M*N*K/dt/1e9;
    double gb = ((double)M*K + (double)N*K)*2.0/dt/1e9;   /* fp16 inputs read (lower bound) */
    printf("  %-10s %7.1f GFLOP/s  %6.0f us  (weights+act read ~%.0f GB/s)\n", tag, gf, dt*1e6, gb);
    cudaEventDestroy(evs); cudaEventDestroy(eve);
    return gf;
}

struct Ctx { cublasHandle_t cb; float *y; __half *xh, *W; int M,K,N; };
static void run_cublas(void *p){ Ctx*c=(Ctx*)p; gemm_cublas(c->cb,c->y,c->xh,c->W,c->M,c->K,c->N); }
static void run_wmma(void *p){ Ctx*c=(Ctx*)p; dim3 g((c->N+BN-1)/BN,(c->M+BM-1)/BM); gemm_wmma<<<g,128>>>(c->y,c->xh,c->W,c->M,c->K,c->N); }

static void shape(cublasHandle_t cb, const char *name, int M, int K, int N){
    int Mp=((M+BM-1)/BM)*BM;                       /* pad rows so WMMA full-tile stores stay in-bounds */
    size_t xn=(size_t)M*K, wn=(size_t)N*K, yn=(size_t)M*N, ynp=(size_t)Mp*N;
    float *xf; __half *xh,*W; float *y1,*y2;
    CK(cudaMalloc(&xf,xn*4)); CK(cudaMalloc(&xh,xn*2)); CK(cudaMalloc(&W,wn*2));
    CK(cudaMalloc(&y1,ynp*4)); CK(cudaMalloc(&y2,ynp*4));
    /* deterministic fill */
    float *hx=(float*)malloc(xn*4); for(size_t i=0;i<xn;i++) hx[i]=((int)(i%17)-8)/8.f;
    __half *hw=(__half*)malloc(wn*2); for(size_t i=0;i<wn;i++) hw[i]=__float2half(((int)(i%13)-6)/6.f);
    CK(cudaMemcpy(xf,hx,xn*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(W,hw,wn*2,cudaMemcpyHostToDevice));
    k_f32_to_f16<<<(xn+255)/256,256>>>(xh,xf,xn);
    Ctx c{cb,y1,xh,W,M,K,N};
    /* correctness: max abs diff cublas vs wmma */
    gemm_cublas(cb,y1,xh,W,M,K,N); { dim3 g((N+BN-1)/BN,(M+BM-1)/BM); gemm_wmma<<<g,128>>>(y2,xh,W,M,K,N); }
    CK(cudaDeviceSynchronize());
    float *h1=(float*)malloc(yn*4),*h2=(float*)malloc(yn*4);
    CK(cudaMemcpy(h1,y1,yn*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(h2,y2,yn*4,cudaMemcpyDeviceToHost));
    double md=0,ref=0; for(size_t i=0;i<yn;i++){ double d=fabs(h1[i]-h2[i]); if(d>md)md=d; if(fabs(h1[i])>ref)ref=fabs(h1[i]); }
    printf("%s  M=%d K=%d N=%d   |wmma-cublas|max=%.3g (rel %.2g)\n", name,M,K,N,md,md/(ref+1e-9));
    c.y=y1; double gc=bench("cublas",run_cublas,&c,M,K,N);
    c.y=y2; double gw=bench("wmma",  run_wmma,  &c,M,K,N);
    printf("  -> wmma/cublas = %.2fx\n", gw/gc);
    free(hx);free(hw);free(h1);free(h2);
    cudaFree(xf);cudaFree(xh);cudaFree(W);cudaFree(y1);cudaFree(y2);
}

int main(){
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr,0));
    printf("device: %s  sm_%d%d  HBM ~%.0f GB/s\n\n", pr.name, pr.major, pr.minor,
           2.0*pr.memoryClockRate*1e3*(pr.memoryBusWidth/8)/1e9);
    cublasHandle_t cb; cublasCreate(&cb);
    /* DiT FFN shapes (small dim=1024 inner=4096), at S for ~10s / ~20s / ~60s gens */
    for (int S : {172, 279, 710}) {
        printf("== S=%d ==\n", S);
        shape(cb,"ff_in (d->2i)", S, 1024, 8192);
        shape(cb,"ff_out(i->d) ", S, 4096, 1024);
        shape(cb,"qkv (d->3d)  ", S, 1024, 3072);
    }
    cublasDestroy(cb);
    return 0;
}
