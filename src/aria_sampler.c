/*
 * aria_sampler.c - RNG, LogSNR schedule, pingpong sampler.
 */

#include "aria_sampler.h"
#include <stdlib.h>
#include <math.h>

/* ---- xoshiro256** ---- */
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

void aria_rng_seed(aria_rng *r, uint64_t seed) {
    /* SplitMix64 to fill the state */
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B97F4A7C15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        r->s[i] = z ^ (z >> 31);
    }
}

static uint64_t rng_next(aria_rng *r) {
    uint64_t *s = r->s;
    uint64_t result = rotl(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl(s[3], 45);
    return result;
}

double aria_rng_next_double(aria_rng *r) {
    /* 53-bit mantissa in [0,1) */
    return (rng_next(r) >> 11) * (1.0 / 9007199254740992.0);
}

float aria_rng_gaussian(aria_rng *r) {
    double u1 = aria_rng_next_double(r);
    double u2 = aria_rng_next_double(r);
    if (u1 < 1e-300) u1 = 1e-300;
    return (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2));
}

void aria_rng_randn(aria_rng *r, float *out, int n) {
    for (int i = 0; i < n; i++) out[i] = aria_rng_gaussian(r);
}

/* ---- schedule ---- */
void aria_logsnr_schedule(float *out, int steps, float sigma_max,
                          float anchor_logsnr, float anchor_length, float rate,
                          float logsnr_end, float seq_len) {
    int N = steps + 1;
    double logsnr_start = (double)anchor_logsnr - (double)rate * log2((double)seq_len / (double)anchor_length);
    for (int i = 0; i < N; i++) {
        double ti = (N > 1) ? (double)sigma_max * (1.0 - (double)i / (double)(N - 1)) : (double)sigma_max;
        double logsnr = (double)logsnr_end - ti * ((double)logsnr_end - logsnr_start);
        double t_out = 1.0 / (1.0 + exp(logsnr));  /* sigmoid(-logsnr) */
        if (ti <= 0.0) t_out = 0.0;
        if (ti >= 1.0) t_out = 1.0;
        out[i] = (float)t_out;
    }
    out[0] = sigma_max;  /* pin first to sigma_max */
}

/* ---- pingpong ---- */
void aria_pingpong_cb(float *x, int n, const float *sigmas, int steps,
                      aria_denoiser_fn denoise, void *ctx,
                      aria_rng *rng, const float *injected_noise,
                      void (*progress)(int step, int total, void *user), void *user) {
    float *v = malloc((size_t)n * sizeof(float));
    float *denoised = malloc((size_t)n * sizeof(float));
    for (int i = 0; i < steps; i++) {
        float tc = sigmas[i], tn = sigmas[i + 1];
        denoise(ctx, x, tc, v, n);
        for (int j = 0; j < n; j++) denoised[j] = x[j] - tc * v[j];
        const float *noise = injected_noise ? injected_noise + (size_t)i * n : NULL;
        for (int j = 0; j < n; j++) {
            float z = noise ? noise[j] : aria_rng_gaussian(rng);
            x[j] = (1.0f - tn) * denoised[j] + tn * z;
        }
        if (progress) progress(i + 1, steps, user);
    }
    free(v);
    free(denoised);
}

void aria_pingpong(float *x, int n, const float *sigmas, int steps,
                   aria_denoiser_fn denoise, void *ctx,
                   aria_rng *rng, const float *injected_noise) {
    aria_pingpong_cb(x, n, sigmas, steps, denoise, ctx, rng, injected_noise, NULL, NULL);
}
