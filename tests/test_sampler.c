/* test_sampler.c - hermetic: pingpong update, schedule shape, RNG stats. */
#include "../src/aria_sampler.h"
#include <stdio.h>
#include <math.h>

static int fails = 0;
static void check(const char *n, float got, float want, float tol) {
    if (fabsf(got - want) > tol) { printf("FAIL %s: got %.6f want %.6f\n", n, got, want); fails++; }
}

/* denoiser: v = 0.5 (constant), independent of x */
static void const_denoiser(void *ctx, const float *x, float t, float *v, int n) {
    (void)ctx; (void)x; (void)t;
    for (int i = 0; i < n; i++) v[i] = 0.5f;
}

int main(void) {
    /* pingpong with injected zero noise: hand-computed trajectory */
    {
        float x[3] = {1.0f, 1.0f, 1.0f};
        float sigmas[3] = {1.0f, 0.5f, 0.0f};
        float noise[6] = {0,0,0, 0,0,0};
        aria_pingpong(x, 3, sigmas, 2, const_denoiser, NULL, NULL, noise);
        /* step0: denoised=1-1*0.5=0.5; x=(1-0.5)*0.5+0.5*0=0.25
           step1: denoised=0.25-0.5*0.5=0.0; x=(1-0)*0+0*z=0.0 */
        for (int i = 0; i < 3; i++) check("pingpong_zero", x[i], 0.0f, 1e-6f);
    }
    /* pingpong with nonzero injected noise on a single step */
    {
        float x[1] = {2.0f};
        float sigmas[2] = {1.0f, 0.5f};   /* 1 step */
        float noise[1] = {1.0f};
        aria_pingpong(x, 1, sigmas, 1, const_denoiser, NULL, NULL, noise);
        /* denoised=2-1*0.5=1.5; x=(1-0.5)*1.5+0.5*1=0.75+0.5=1.25 */
        check("pingpong_noise", x[0], 1.25f, 1e-6f);
    }
    /* schedule: endpoints + monotonic descending */
    {
        float sched[9];
        aria_logsnr_schedule(sched, 8, 1.0f, -6.2f, 2000.0f, 1.0f, 2.0f, 323.0f);
        check("sched_first", sched[0], 1.0f, 1e-6f);
        check("sched_last", sched[8], 0.0f, 1e-6f);
        int mono = 1;
        for (int i = 1; i < 9; i++) if (sched[i] > sched[i-1] + 1e-7f) mono = 0;
        if (!mono) { printf("FAIL schedule not descending\n"); fails++; }
    }
    /* RNG: gaussian mean ~0, var ~1 */
    {
        aria_rng r; aria_rng_seed(&r, 12345);
        int N = 200000; double m = 0, v = 0;
        for (int i = 0; i < N; i++) { float z = aria_rng_gaussian(&r); m += z; v += (double)z*z; }
        m /= N; v = v / N - m*m;
        if (fabs(m) > 0.02) { printf("FAIL rng mean=%.4f\n", m); fails++; }
        if (fabs(v - 1.0) > 0.03) { printf("FAIL rng var=%.4f\n", v); fails++; }
    }

    if (fails) { printf("test_sampler: %d failures\n", fails); return 1; }
    printf("test_sampler: OK\n");
    return 0;
}
