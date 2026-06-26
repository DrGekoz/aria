/* test_ops.c - known-answer tests for the CPU op kernels. */
#include "../src/aria_ops.h"
#include <stdio.h>
#include <math.h>

static int fails = 0;
static void check(const char *name, float got, float want) {
    float tol = 1e-4f;
    if (fabsf(got - want) > tol) {
        printf("FAIL %s: got %.6f want %.6f\n", name, got, want);
        fails++;
    }
}

int main(void) {
    /* aria_linear: y = x @ W^T + b */
    {
        float x[2] = {1.0f, 2.0f};            /* M=1, K=2 */
        float W[6] = {1,0, 0,1, 1,1};         /* N=3, K=2 */
        float b[3] = {0,0,0};
        float y[3];
        aria_linear(y, x, W, b, 1, 2, 3);
        check("linear[0]", y[0], 1.0f);
        check("linear[1]", y[1], 2.0f);
        check("linear[2]", y[2], 3.0f);
    }
    /* aria_matmul: C = A@B */
    {
        float A[6] = {1,2,3, 4,5,6};   /* 2x3 */
        float B[3] = {1,1,1};          /* 3x1 */
        float C[2];
        aria_matmul(C, A, B, 2, 3, 1);
        check("matmul[0]", C[0], 6.0f);
        check("matmul[1]", C[1], 15.0f);
    }
    /* aria_rmsnorm: x=[3,4], dim=2 */
    {
        float x[2] = {3.0f, 4.0f};
        float y[2];
        aria_rmsnorm(y, x, NULL, 1, 2, 0.0f);
        check("rmsnorm[0]", y[0], 0.848528f);
        check("rmsnorm[1]", y[1], 1.131371f);
    }
    /* aria_silu */
    {
        float v[2] = {0.0f, 1.0f};
        aria_silu(v, 2);
        check("silu(0)", v[0], 0.0f);
        check("silu(1)", v[1], 0.731059f);
    }
    /* aria_gelu_tanh(1.0) ~= 0.841192 */
    {
        float v[1] = {1.0f};
        aria_gelu_tanh(v, 1);
        check("gelu_tanh(1)", v[0], 0.841192f);
    }
    /* aria_softmax row [1,2,3] */
    {
        float x[3] = {1.0f, 2.0f, 3.0f};
        aria_softmax_inplace(x, 1, 3, NULL);
        check("softmax[0]", x[0], 0.090031f);
        check("softmax[1]", x[1], 0.244728f);
        check("softmax[2]", x[2], 0.665241f);
    }
    /* aria_silu_gate */
    {
        float gate[1] = {1.0f}, up[1] = {2.0f}, out[1];
        aria_silu_gate(out, gate, up, 1);
        check("silu_gate", out[0], 0.731059f * 2.0f);
    }

    if (fails) { printf("test_ops: %d failures\n", fails); return 1; }
    printf("test_ops: OK\n");
    return 0;
}
