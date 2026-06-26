/*
 * aria_cond.h - Conditioning: NumberConditioner (duration), timestep features,
 * and (Phase 1) cross-attention / global / local conditioning assembly.
 */

#ifndef ARIA_COND_H
#define ARIA_COND_H

/* ExpoFourierFeatures(dim) -> [cos(t*freqs*2pi), sin(...)], dim outputs total.
 * freqs = exp(linspace(0,1,dim/2) * (log(max_freq)-log(min_freq)) + log(min_freq)).
 * SA3 uses dim=256, min_freq=0.5, max_freq=10000. `out` holds `dim` floats. */
void aria_expo_fourier(float *out, float t, int dim, float min_freq, float max_freq);

/* NumberConditioner(seconds_total): clamp+normalize, ExpoFourier(dim=256),
 * then Linear(256 -> features) with W[features,256], b[features].
 * out holds `features` floats (768 for SA3). */
void aria_number_embed(float *out, float value, float min_val, float max_val,
                       const float *W, const float *b, int features);

#endif /* ARIA_COND_H */
