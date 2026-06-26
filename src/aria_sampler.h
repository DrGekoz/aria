/*
 * aria_sampler.h - RNG, LogSNR schedule, and the pingpong rectified-flow sampler.
 *
 * Reference: stable_audio_tools inference/sampling.py (build_schedule, LogSNRShift,
 * sample_flow_pingpong). Pingpong update per step:
 *   denoised = x - t_curr * v ;  x = (1 - t_next) * denoised + t_next * noise
 * where v = denoiser(x, t_curr). Noise is drawn every step (zeroed when t_next=0).
 */

#ifndef ARIA_SAMPLER_H
#define ARIA_SAMPLER_H

#include <stdint.h>

/* xoshiro256** PRNG (seeded via SplitMix64), with N(0,1) via Box-Muller. */
typedef struct { uint64_t s[4]; } aria_rng;
void  aria_rng_seed(aria_rng *r, uint64_t seed);
double aria_rng_next_double(aria_rng *r);   /* uniform [0,1) */
float aria_rng_gaussian(aria_rng *r);       /* N(0,1) */
void  aria_rng_randn(aria_rng *r, float *out, int n);

/* build_schedule + LogSNRShift. out holds steps+1 values, descending from
 * sigma_max to 0 (first pinned to sigma_max). logsnr_start = anchor_logsnr -
 * rate*log2(seq_len/anchor_length); logsnr = logsnr_end - t*(logsnr_end -
 * logsnr_start); t_out = sigmoid(-logsnr). */
void aria_logsnr_schedule(float *out, int steps, float sigma_max,
                          float anchor_logsnr, float anchor_length, float rate,
                          float logsnr_end, float seq_len);

/* Denoiser callback: write velocity v_out[n] for input x[n] at timestep t. */
typedef void (*aria_denoiser_fn)(void *ctx, const float *x, float t, float *v_out, int n);

/* Pingpong loop over `steps` (sigmas has steps+1 entries). x[n] is updated in
 * place. injected_noise[steps*n] overrides the RNG when non-NULL (for parity). */
void aria_pingpong(float *x, int n, const float *sigmas, int steps,
                   aria_denoiser_fn denoise, void *ctx,
                   aria_rng *rng, const float *injected_noise);

#endif /* ARIA_SAMPLER_H */
