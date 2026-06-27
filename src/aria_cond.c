/*
 * aria_cond.c - Conditioning primitives.
 *
 * Parity target: stable_audio_tools NumberConditioner + ExpoFourierFeatures
 * (models/conditioners.py, models/adp.py, models/blocks.py).
 */

#include "aria_cond.h"
#include "aria_ops.h"
#include <math.h>
#include <stddef.h>

void aria_expo_fourier(float *out, float t, int dim, float min_freq, float max_freq) {
    /* Computed in double precision: arg = t*freq*2pi reaches ~6e4 at max
     * freq/duration, where float32 cos/sin is dominated by rounding noise
     * (and diverges across libm implementations). Double gives the intended,
     * stable feature; the result lands in [-1,1] and is stored as f32. */
    int half = dim / 2;
    double log_min = log((double)min_freq);
    double log_max = log((double)max_freq);
    const double two_pi = 6.283185307179586477;
    for (int i = 0; i < half; i++) {
        /* ramp = linspace(0,1,half): ramp[i] = i/(half-1) */
        double ramp = (half > 1) ? (double)i / (double)(half - 1) : 0.0;
        double freq = exp(ramp * (log_max - log_min) + log_min);
        double arg = (double)t * freq * two_pi;
        out[i] = (float)cos(arg);          /* cos block */
        out[half + i] = (float)sin(arg);   /* sin block */
    }
}

void aria_number_embed(float *out, float value, float min_val, float max_val,
                       const float *W, const float *b, int features) {
    /* clamp then normalize to [0,1] */
    float v = value;
    if (v < min_val) v = min_val;
    if (v > max_val) v = max_val;
    float t = (max_val > min_val) ? (v - min_val) / (max_val - min_val) : 0.0f;

    float fourier[256];
    aria_expo_fourier(fourier, t, 256, 0.5f, 10000.0f);
    aria_linear(out, fourier, W, b, 1, 256, features);
}

void aria_inpaint_mask_latent(float *mask_lat, const float *mask_audio, int audio_len, int T) {
    for (int t = 0; t < T; t++)
        mask_lat[t] = mask_audio[(size_t)t * audio_len / T];  /* torch nearest: floor(t*in/out) */
}

void aria_inpaint_local_cond(float *local_TC, const float *latent_CT, const float *mask_lat, int T) {
    const int C = 256;
    for (int t = 0; t < T; t++) {
        float mt = mask_lat[t];
        float *row = local_TC + (size_t)t * (C + 1);
        row[0] = mt;                                       /* channel 0 = inpaint_mask */
        for (int c = 0; c < C; c++)
            row[1 + c] = latent_CT[(size_t)c * T + t] * mt; /* masked_input = latent*mask */
    }
}
