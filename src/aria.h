/*
 * aria.h - Public API for the aria audio-diffusion inference runtime.
 *
 *   aria_ctx *ctx = aria_load("path/to/model");      // dir with model_config.json
 *   aria_gen_params p = ARIA_GEN_PARAMS_DEFAULT;
 *   p.prompt = "warm romantic piano"; p.seconds_total = 15.0f;
 *   aria_audio *audio = NULL;
 *   aria_generate(ctx, &p, &audio);
 *   aria_wav_write("out.wav", audio, 32);
 *   aria_audio_free(audio); aria_free(ctx);
 */

#ifndef ARIA_H
#define ARIA_H

#include <stdint.h>
#include "aria_wav.h"
#include "aria_quant.h"   /* aria_dtype (precision selection) */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aria_ctx aria_ctx;

typedef enum { ARIA_DEVICE_CPU = 0, ARIA_DEVICE_CUDA = 1, ARIA_DEVICE_AUTO = 2 } aria_device;

typedef struct {
    const char *prompt;             /* text prompt (Phase 1) */
    const char *prompt_embed_path;  /* optional precomputed cross-attn embedding, f32 [256,768] */
    float seconds_total;            /* requested duration in seconds */
    int   steps;                    /* denoising steps (default 8) */
    float cfg_scale;                /* CFG scale (default 1.0 = off) */
    int64_t seed;                   /* RNG seed (-1 = random) */
    aria_device device;             /* DiT backend: auto (default) / cpu / cuda (E8.5) */
    aria_dtype precision;           /* DiT weight precision: fp32 (default) / q8 / q4 (E9, CPU) */
    const char *load_quant;         /* optional pre-quantized DiT (.aria) from aria-quantize (E9.2) */
    /* continue / inpaint (E7): regenerate part of an existing clip, keep the rest.
     * Set init_audio to a WAV path. inpaint_from_s..inpaint_to_s marks the region
     * to regenerate (seconds; to_s <= 0 means "to seconds_total"). inpaint_continue
     * = regenerate [clip_duration, seconds_total] (extend the clip). Runs on the CPU
     * DiT (the device DiT has no local-cond path yet). */
    const char *init_audio;         /* WAV path, or NULL for plain text->audio */
    float inpaint_from_s;           /* start of the regenerated region (seconds) */
    float inpaint_to_s;             /* end of the regenerated region; <= 0 = seconds_total */
    int   inpaint_continue;         /* 1 = continue mode (regenerate the tail) */
} aria_gen_params;

#define ARIA_GEN_PARAMS_DEFAULT { NULL, NULL, 15.0f, 8, 1.0f, 0, ARIA_DEVICE_AUTO, ARIA_F32, NULL, NULL, 0.0f, 0.0f, 0 }

/* Load a model from a directory containing model_config.json + model.safetensors.
 * Returns NULL on error (see aria_last_error()). */
aria_ctx *aria_load(const char *model_dir);
void aria_free(aria_ctx *ctx);

/* Introspection */
const char *aria_model_type(const aria_ctx *ctx);
int aria_sample_rate(const aria_ctx *ctx);
int aria_audio_channels(const aria_ctx *ctx);
int aria_num_tensors(const aria_ctx *ctx);
void aria_list_tensors(const aria_ctx *ctx, const char *prefix); /* prefix may be NULL */

/* Text-to-audio generation (Phase 1). Sets *out on success. Returns 0 / <0. */
int aria_generate(aria_ctx *ctx, const aria_gen_params *p, aria_audio **out);

const char *aria_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* ARIA_H */
