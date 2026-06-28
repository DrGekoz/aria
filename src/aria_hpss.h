/*
 * aria_hpss.h - harmonic/percussive source separation (HPSS) for streaming.
 *
 * Median-filtering HPSS (Fitzgerald 2010): the percussive layer (drums, transients)
 * is broadband/vertical in the spectrogram; the harmonic layer (tonal: melody, pads,
 * bass) is narrowband/horizontal. A horizontal median enhances harmonics, a vertical
 * median enhances percussives, and soft masks split the signal. Dependency-free
 * (own radix-2 FFT). Used by the --stream loop to hold a steady percussive groove
 * across chunks while the harmonic layer evolves.
 */
#ifndef ARIA_HPSS_H
#define ARIA_HPSS_H

#include <stdint.h>

/* Separate interleaved audio `in` [nframes*ch] into `harmonic` + `percussive`
 * (both [nframes*ch], pre-allocated). harmonic+percussive ~= in. */
void aria_hpss_separate(const float *in, int64_t nframes, int ch,
                        float *harmonic, float *percussive);

#endif /* ARIA_HPSS_H */
