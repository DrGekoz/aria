/*
 * aria_wav.h - Minimal RIFF/WAVE reader & writer (no libsndfile)
 *
 * Audio is stored INTERLEAVED in float (natural for WAV files), nominal
 * range [-1, 1]. The SA3 autoencoder works in planar [channels, frames]
 * layout; conversion happens at the model boundary, not here.
 */

#ifndef ARIA_WAV_H
#define ARIA_WAV_H

#include <stdint.h>

typedef struct {
    int sample_rate;
    int channels;
    int64_t num_frames;   /* samples per channel */
    float *data;          /* interleaved, channels * num_frames floats */
} aria_audio;

aria_audio *aria_audio_alloc(int sample_rate, int channels, int64_t num_frames);
void aria_audio_free(aria_audio *a);

/* bits: 16 -> PCM16, 32 -> IEEE float32. Returns 0 on success, -1 on error. */
int aria_wav_write(const char *path, const aria_audio *a, int bits);

/* Reads PCM16 / PCM24 / PCM32 / float32 WAV into interleaved float. NULL on error. */
aria_audio *aria_wav_read(const char *path);

#endif /* ARIA_WAV_H */
