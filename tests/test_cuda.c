/* test_cuda.c - parity: CUDA kernels vs CPU ops (E8.2/E8.3/E8.4).
 * Built/run by `make test_cuda`; SKIPs cleanly when no CUDA device is present. */
#include "../src/aria_ops.h"
#include "../src/aria_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int fails = 0;

static void fill(float *p, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; i++) {
        seed = seed * 1664525u + 1013904223u;
        p[i] = (float)((seed >> 8) & 0xffff) / 32768.0f - 1.0f;  /* [-1,1) */
    }
}
static void cmp(const char *name, const float *got, const float *ref, size_t n, float atol, float rtol) {
    float md = 0.0f, maxref = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(got[i] - ref[i]); if (d > md) md = d;
        float a = fabsf(ref[i]);          if (a > maxref) maxref = a;
    }
    float thr = atol + rtol * maxref;
    if (md > thr) { printf("FAIL %-14s maxdiff=%.3e thr=%.3e (max|ref|=%.2f)\n", name, md, thr, maxref); fails++; }
    else printf("ok   %-14s maxdiff=%.3e (max|ref|=%.2f)\n", name, md, maxref);
}

static void t_linear(int M, int K, int N, int bias) {
    float *x = malloc((size_t)M*K*4), *W = malloc((size_t)N*K*4), *b = bias ? malloc((size_t)N*4) : NULL;
    float *yc = malloc((size_t)M*N*4), *yg = malloc((size_t)M*N*4);
    fill(x, (size_t)M*K, 1); fill(W, (size_t)N*K, 2); if (b) fill(b, N, 3);
    aria_linear(yc, x, W, b, M, K, N);
    aria_cuda_linear(yg, x, W, b, M, K, N);
    char nm[32]; snprintf(nm, sizeof nm, "linear%dx%dx%d", M, K, N);
    cmp(nm, yg, yc, (size_t)M*N, 1e-3f, 2e-4f);
    free(x); free(W); free(b); free(yc); free(yg);
}

int main(void) {
    if (!aria_cuda_available()) { printf("test_cuda: SKIP (no CUDA device)\n"); return 0; }
    char info[256]; aria_cuda_device_info(info, sizeof info);
    printf("test_cuda: device = %s\n", info);

    /* ---- linear (E8.2) ---- */
    t_linear(279, 1024, 3072, 1);
    t_linear(257, 1024, 2048, 0);
    t_linear(256, 2048,  768, 1);
    t_linear(33,   100,   17, 1);
    t_linear(1,   1024, 1024, 0);

    /* ---- matmul C=A@B ---- */
    { int M=200, K=128, N=96; float *A=malloc((size_t)M*K*4), *B=malloc((size_t)K*N*4);
      float *c=malloc((size_t)M*N*4), *g=malloc((size_t)M*N*4);
      fill(A,(size_t)M*K,5); fill(B,(size_t)K*N,6);
      aria_matmul(c,A,B,M,K,N); aria_cuda_matmul(g,A,B,M,K,N);
      cmp("matmul", g, c, (size_t)M*N, 1e-3f, 2e-4f); free(A);free(B);free(c);free(g); }

    /* ---- rmsnorm / gemma_rmsnorm ---- */
    { int R=300, D=1024; float *x=malloc((size_t)R*D*4), *w=malloc((size_t)D*4);
      float *c=malloc((size_t)R*D*4), *g=malloc((size_t)R*D*4);
      fill(x,(size_t)R*D,7); fill(w,D,8);
      aria_rmsnorm(c,x,w,R,D,1e-6f); aria_cuda_rmsnorm(g,x,w,R,D,1e-6f);
      cmp("rmsnorm", g, c, (size_t)R*D, 1e-4f, 1e-4f);
      aria_gemma_rmsnorm(c,x,w,R,D,1e-6f); aria_cuda_gemma_rmsnorm(g,x,w,R,D,1e-6f);
      cmp("gemma_rmsnorm", g, c, (size_t)R*D, 1e-4f, 1e-4f); free(x);free(w);free(c);free(g); }

    /* ---- dynamic_tanh ---- */
    { int R=256, D=64; float *x=malloc((size_t)R*D*4), *w=malloc((size_t)D*4), *b=malloc((size_t)D*4);
      float *c=malloc((size_t)R*D*4), *g=malloc((size_t)R*D*4);
      fill(x,(size_t)R*D,9); fill(w,D,10); fill(b,D,11);
      aria_dynamic_tanh(c,x,0.7f,w,b,R,D); aria_cuda_dynamic_tanh(g,x,0.7f,w,b,R,D);
      cmp("dynamic_tanh", g, c, (size_t)R*D, 1e-5f, 1e-5f); free(x);free(w);free(b);free(c);free(g); }

    /* ---- elementwise: softcap / silu / gelu_tanh ---- */
    { int n=4096; float *x=malloc((size_t)n*4), *c=malloc((size_t)n*4), *g=malloc((size_t)n*4);
      fill(x,n,12);
      for(int i=0;i<n;i++){c[i]=x[i]*30.0f;g[i]=x[i]*30.0f;}
      aria_softcap(c,n,50.0f); aria_cuda_softcap(g,n,50.0f); cmp("softcap", g, c, n, 1e-5f, 1e-5f);
      for(int i=0;i<n;i++){c[i]=x[i]*4.0f;g[i]=x[i]*4.0f;} aria_silu(c,n); aria_cuda_silu(g,n); cmp("silu", g, c, n, 1e-5f, 1e-5f);
      for(int i=0;i<n;i++){c[i]=x[i]*4.0f;g[i]=x[i]*4.0f;} aria_gelu_tanh(c,n); aria_cuda_gelu_tanh(g,n); cmp("gelu_tanh", g, c, n, 1e-5f, 1e-5f);
      free(x);free(c);free(g); }

    /* ---- silu_gate ---- */
    { int n=4096; float *gt=malloc((size_t)n*4), *up=malloc((size_t)n*4), *c=malloc((size_t)n*4), *g=malloc((size_t)n*4);
      fill(gt,n,13); fill(up,n,14);
      aria_silu_gate(c,gt,up,n); aria_cuda_silu_gate(g,gt,up,n); cmp("silu_gate", g, c, n, 1e-5f, 1e-5f);
      free(gt);free(up);free(c);free(g); }

    /* ---- softmax (with mask) ---- */
    { int R=64, C=80; float *x=malloc((size_t)R*C*4), *m=malloc((size_t)R*C*4);
      float *c=malloc((size_t)R*C*4), *g=malloc((size_t)R*C*4);
      fill(x,(size_t)R*C,15);
      for(int q=0;q<R;q++)for(int k=0;k<C;k++)m[q*C+k]=(k<C-7)?0.0f:-3.4e38f;
      for(size_t i=0;i<(size_t)R*C;i++){c[i]=x[i];g[i]=x[i];}
      aria_softmax_inplace(c,R,C,m); aria_cuda_softmax(g,R,C,m); cmp("softmax", g, c, (size_t)R*C, 1e-5f, 1e-5f);
      free(x);free(m);free(c);free(g); }

    /* ---- rope_apply ---- */
    { int H=4, N=60, D=64, rot=32; float *x=malloc((size_t)H*N*D*4);
      float *cs=malloc((size_t)N*(rot/2)*4), *sn=malloc((size_t)N*(rot/2)*4);
      float *c=malloc((size_t)H*N*D*4), *g=malloc((size_t)H*N*D*4);
      fill(x,(size_t)H*N*D,16); aria_rope_freqs(cs,sn,N,rot,10000.0f);
      for(size_t i=0;i<(size_t)H*N*D;i++){c[i]=x[i];g[i]=x[i];}
      aria_rope_apply(c,cs,sn,H,N,D,rot); aria_cuda_rope_apply(g,cs,sn,H,N,D,rot);
      cmp("rope_apply", g, c, (size_t)H*N*D, 1e-5f, 1e-5f); free(x);free(cs);free(sn);free(c);free(g); }

    /* ---- attention (no mask) ---- */
    { int H=8, Nq=48, Nk=48, D=64; size_t qn=(size_t)H*Nq*D, kn=(size_t)H*Nk*D;
      float *q=malloc(qn*4), *k=malloc(kn*4), *v=malloc(kn*4), *c=malloc(qn*4), *g=malloc(qn*4);
      fill(q,qn,17); fill(k,kn,18); fill(v,kn,19);
      aria_attention(c,q,k,v,H,Nq,Nk,D,NULL,NULL); aria_cuda_attention(g,q,k,v,H,Nq,Nk,D,NULL);
      cmp("attention", g, c, qn, 1e-3f, 1e-3f); free(q);free(k);free(v);free(c);free(g); }

    /* ---- conv1d ---- */
    { int Cin=16, Cout=32, K=3, pad=1, L=120; float *in=malloc((size_t)Cin*L*4);
      float *w=malloc((size_t)Cout*Cin*K*4), *b=malloc((size_t)Cout*4);
      float *c=malloc((size_t)Cout*L*4), *g=malloc((size_t)Cout*L*4);
      fill(in,(size_t)Cin*L,20); fill(w,(size_t)Cout*Cin*K,21); fill(b,Cout,22);
      aria_conv1d(c,in,w,b,Cin,Cout,K,pad,L); aria_cuda_conv1d(g,in,w,b,Cin,Cout,K,pad,L);
      cmp("conv1d", g, c, (size_t)Cout*L, 1e-4f, 1e-4f); free(in);free(w);free(b);free(c);free(g); }

    if (fails) { printf("test_cuda: %d FAILURES\n", fails); return 1; }
    printf("test_cuda: OK\n");
    return 0;
}
