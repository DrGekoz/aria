/* bench_gemm.c - microbenchmark for aria_linear on the dominant SA3 shapes. */
#include "../src/aria_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }

static void bench(int M, int K, int N, const char *name) {
    float *x = malloc((size_t)M * K * sizeof(float));
    float *W = malloc((size_t)N * K * sizeof(float));
    float *y = malloc((size_t)M * N * sizeof(float));
    for (size_t i = 0; i < (size_t)M * K; i++) x[i] = (float)((int)(i % 17) - 8) / 8.0f;
    for (size_t i = 0; i < (size_t)N * K; i++) W[i] = (float)((int)(i % 13) - 6) / 6.0f;
    aria_linear(y, x, W, NULL, M, K, N);  /* warmup */
    int reps = 30;
    double t0 = now();
    for (int r = 0; r < reps; r++) aria_linear(y, x, W, NULL, M, K, N);
    double dt = (now() - t0) / reps;
    printf("%-14s M=%4d K=%5d N=%5d  %6.2f ms  %6.1f GFLOP/s\n",
           name, M, K, N, dt * 1e3, 2.0 * M * N * K / dt / 1e9);
    free(x); free(W); free(y);
}

int main(void) {
    bench(279, 1024, 3072, "dit_to_qkv");
    bench(279, 1024, 8192, "dit_ff_in");
    bench(279, 4096, 1024, "dit_ff_out");
    bench(257, 1024, 2048, "dit_cross_kv");
    bench(256,  768, 2048, "t5_mlp_in");
    bench(256, 2048,  768, "t5_mlp_out");
    return 0;
}
