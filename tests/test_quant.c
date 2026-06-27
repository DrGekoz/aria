/* test_quant.c - hermetic tests for Q8/Q4 weight quantization + dequant GEMM.
 * Checks: dtype parse, round-trip max error within the format's resolution, and
 * quantized-GEMM vs fp32 reference within the expected quantization tolerance. */
#include "../src/aria_quant.h"
#include "../src/aria_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int fails = 0;
static unsigned rng = 12345u;
static float frand(void) { rng = rng * 1664525u + 1013904223u; return (float)(rng >> 8) / (float)(1u << 24); }
static float frandn(void) { /* ~N(0,1) via two uniforms (Box-Muller) */
    float u1 = frand() + 1e-7f, u2 = frand();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* dequant a Q8 weight back to f32 for round-trip checks */
static void q8_dequant(float *W, const int8_t *q, const float *scale, int N, int K) {
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++) W[(size_t)n * K + k] = (float)q[(size_t)n * K + k] * scale[n];
}

static void check_le(const char *what, float got, float thr) {
    if (got > thr) { printf("FAIL %s: %.3e > %.3e\n", what, got, thr); fails++; }
    else printf("ok   %s: %.3e (<= %.3e)\n", what, got, thr);
}

/* max relative GEMM error: max|y_q - y_ref| / max|y_ref| */
static float gemm_relerr(const float *yq, const float *yref, int n) {
    float md = 0, mr = 0;
    for (int i = 0; i < n; i++) { float d = fabsf(yq[i] - yref[i]); if (d > md) md = d;
                                  float a = fabsf(yref[i]); if (a > mr) mr = a; }
    return mr > 0 ? md / mr : md;
}

int main(void) {
    /* dtype parse round-trip */
    {
        const char *names[] = {"fp32", "fp16", "bf16", "q8", "q4"};
        aria_dtype dts[] = {ARIA_F32, ARIA_F16, ARIA_BF16, ARIA_Q8, ARIA_Q4};
        for (int i = 0; i < 5; i++) {
            aria_dtype d;
            if (aria_dtype_parse(names[i], &d) != 0 || d != dts[i] ||
                strcmp(aria_dtype_name(dts[i]), names[i]) != 0) { printf("FAIL dtype %s\n", names[i]); fails++; }
        }
        aria_dtype d;
        if (aria_dtype_parse("nope", &d) == 0) { printf("FAIL dtype: accepted junk\n"); fails++; }
        printf("ok   dtype parse/name\n");
    }

    /* shapes mirroring SA3 linears: K (contracted) x N (out), plus an odd K */
    int shapes[][2] = { {768, 1024}, {1024, 3072}, {64, 64}, {1024, 257}, {130, 96} };
    int M = 5;
    for (int si = 0; si < (int)(sizeof(shapes) / sizeof(shapes[0])); si++) {
        int K = shapes[si][0], N = shapes[si][1];
        float *W = malloc((size_t)N * K * sizeof(float));
        float *x = malloc((size_t)M * K * sizeof(float));
        for (int i = 0; i < N * K; i++) W[i] = 0.1f * frandn();    /* weight-like scale */
        for (int i = 0; i < M * K; i++) x[i] = frandn();
        float *yref = malloc((size_t)M * N * sizeof(float));
        aria_linear(yref, x, W, NULL, M, K, N);

        char tag[64];

        /* ---- Q8 ---- */
        {
            int8_t *q = malloc(aria_q8_qbytes(N, K));
            float *scale = malloc(aria_q8_nscale(N, K) * sizeof(float));
            aria_q8_quant(q, scale, W, N, K);
            /* round-trip: per-row error <= scale (half-step is scale/2; allow scale) */
            float *Wd = malloc((size_t)N * K * sizeof(float));
            q8_dequant(Wd, q, scale, N, K);
            float rmax = 0;
            for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) {
                float e = fabsf(Wd[(size_t)n * K + k] - W[(size_t)n * K + k]);
                float s = scale[n] * 0.5f + 1e-9f;            /* worst-case round error */
                if (e / s > rmax) rmax = e / s;
            }
            snprintf(tag, sizeof(tag), "q8 roundtrip[%dx%d] (err/halfstep)", N, K);
            check_le(tag, rmax, 1.001f);
            float *yq = malloc((size_t)M * N * sizeof(float));
            aria_linear_q8(yq, x, q, scale, NULL, M, K, N);
            snprintf(tag, sizeof(tag), "q8 gemm relerr[%dx%d]", N, K);
            check_le(tag, gemm_relerr(yq, yref, M * N), 0.03f);
            free(q); free(scale); free(Wd); free(yq);
        }
        /* ---- Q4 ---- */
        {
            uint8_t *q = malloc(aria_q4_qbytes(N, K));
            float *scale = malloc(aria_q4_nscale(N, K) * sizeof(float));
            aria_q4_quant(q, scale, W, N, K);
            float *yq = malloc((size_t)M * N * sizeof(float));
            aria_linear_q4(yq, x, q, scale, NULL, M, K, N);
            snprintf(tag, sizeof(tag), "q4 gemm relerr[%dx%d]", N, K);
            check_le(tag, gemm_relerr(yq, yref, M * N), 0.12f);
            free(q); free(scale); free(yq);
        }
        free(W); free(x); free(yref);
    }

    /* bias path sanity (Q8) */
    {
        int K = 64, N = 8, Mb = 2;
        float W[64 * 8], x[2 * 64], b[8], yref[16], yq[16];
        for (int i = 0; i < N * K; i++) W[i] = 0.1f * frandn();
        for (int i = 0; i < Mb * K; i++) x[i] = frandn();
        for (int i = 0; i < N; i++) b[i] = frandn();
        aria_linear(yref, x, W, b, Mb, K, N);
        int8_t q[64 * 8]; float scale[8];
        aria_q8_quant(q, scale, W, N, K);
        aria_linear_q8(yq, x, q, scale, b, Mb, K, N);
        check_le("q8 gemm+bias relerr", gemm_relerr(yq, yref, Mb * N), 0.03f);
    }

    if (fails) { printf("test_quant: %d failures\n", fails); return 1; }
    printf("test_quant: OK\n");
    return 0;
}
