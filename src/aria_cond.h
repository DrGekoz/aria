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

/* ---- inpaint / continue conditioning ---- */

/* Down-sample an audio-resolution inpaint mask (1=keep provided audio, 0=inpaint)
 * to latent resolution by nearest-neighbour interpolation, matching torch
 * F.interpolate(mode='nearest'): mask_lat[t] = mask_audio[(t*audio_len)/T].
 * mask_lat holds T floats. */
void aria_inpaint_mask_latent(float *mask_lat, const float *mask_audio, int audio_len, int T);

/* Assemble the DiT local-additive conditioning local[T, 257] from a latent
 * [256, T] (channel-major) and a latent-space mask[T]:
 *   masked_input = latent * mask (per token);
 *   local[t] = [ mask[t] | masked_input[0..255, t] ]  (channel 0 = mask).
 * Matches generation.py: inpaint_input = latent*mask, then concat
 * [inpaint_mask, inpaint_masked_input] (rearranged b c t -> b t c). */
void aria_inpaint_local_cond(float *local_TC, const float *latent_CT, const float *mask_lat, int T);

#endif /* ARIA_COND_H */
