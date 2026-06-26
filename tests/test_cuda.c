/* test_cuda.c - parity: CUDA aria_cuda_linear vs CPU aria_linear (E8.2).
 * Built/run by `make test_cuda`; SKIPs cleanly when no CUDA device is present. */
#include "../src/aria_ops.h"
#include "../src/aria_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int fails = 0;

static void check(int M, int K, int N, int with_bias) {
    float *x = malloc((size_t)M * K * sizeof(float));
    float *W = malloc((size_t)N * K * sizeof(float));
    float *b = with_bias ? malloc((size_t)N * sizeof(float)) : NULL;
    float *yc = malloc((size_t)M * N * sizeof(float));
    float *yg = malloc((size_t)M * N * sizeof(float));
    for (size_t i = 0; i < (size_t)M * K; i++) x[i] = (float)((int)(i % 17) - 8) / 8.0f;
    for (size_t i = 0; i < (size_t)N * K; i++) W[i] = (float)((int)(i % 13) - 6) / 6.0f;
    if (b) for (int i = 0; i < N; i++) b[i] = (float)((i % 5) - 2) / 4.0f;

    aria_linear(yc, x, W, b, M, K, N);        /* CPU reference */
    aria_cuda_linear(yg, x, W, b, M, K, N);   /* GPU */

    float md = 0.0f, maxref = 0.0f;
    for (size_t i = 0; i < (size_t)M * N; i++) {
        float d = fabsf(yc[i] - yg[i]); if (d > md) md = d;
        float a = fabsf(yc[i]);         if (a > maxref) maxref = a;
    }
    float thr = 1e-3f + 2e-4f * maxref;  /* fp32 GEMM accumulation order differs */
    if (md > thr) { printf("FAIL linear M=%d K=%d N=%d bias=%d: maxdiff=%.3e thr=%.3e\n",
                           M, K, N, with_bias, md, thr); fails++; }
    else printf("ok   linear M=%4d K=%5d N=%5d bias=%d: maxdiff=%.3e (max|ref|=%.2f)\n",
               M, K, N, with_bias, md, maxref);
    free(x); free(W); free(b); free(yc); free(yg);
}

int main(void) {
    if (!aria_cuda_available()) { printf("test_cuda: SKIP (no CUDA device)\n"); return 0; }
    char info[256]; aria_cuda_device_info(info, sizeof info);
    printf("test_cuda: device = %s\n", info);

    check(279, 1024, 3072, 1);   /* DiT to_qkv */
    check(257, 1024, 2048, 0);   /* DiT cross to_kv */
    check(256,  768, 2048, 1);   /* T5 mlp in */
    check(256, 2048,  768, 1);   /* T5 mlp out */
    check(64,    64,   64, 0);   /* small, exact tile */
    check(33,   100,   17, 1);   /* odd dims -> edge masking */
    check(1,   1024, 1024, 0);   /* single row */

    if (fails) { printf("test_cuda: %d FAILURES\n", fails); return 1; }
    printf("test_cuda: OK\n");
    return 0;
}
