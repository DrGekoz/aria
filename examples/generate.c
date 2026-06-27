/*
 * generate.c - minimal external program using the aria public API.
 *
 * Demonstrates the whole surface a third-party caller needs: it includes only
 * the installed public headers (aria.h, which pulls in aria_wav.h) and links
 * against libaria.a. Doubles as the E11.1 link check.
 *
 *   make example                 # build ./example against build/libaria.a
 *   ./example <model_dir> [out.wav] [seconds]
 *
 * Generates an unconditional clip (no text encoder / tokenizer needed) so it
 * runs against any model dir with model_config.json + model.safetensors.
 */
#include "aria.h"
#include "aria_wav.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model_dir> [out.wav] [seconds]\n", argv[0]);
        return 2;
    }
    const char *model_dir = argv[1];
    const char *out_path  = argc > 2 ? argv[2] : "example.wav";
    float seconds         = argc > 3 ? (float)atof(argv[3]) : 5.0f;

    aria_ctx *ctx = aria_load(model_dir);
    if (!ctx) { fprintf(stderr, "load error: %s\n", aria_last_error()); return 1; }
    printf("loaded %s (type=%s, %d Hz, %d ch)\n", model_dir,
           aria_model_type(ctx), aria_sample_rate(ctx), aria_audio_channels(ctx));

    aria_gen_params p = ARIA_GEN_PARAMS_DEFAULT;  /* unconditional: prompt stays NULL */
    p.seconds_total = seconds;
    p.steps = 8;
    p.seed = 0;

    aria_audio *audio = NULL;
    int rc = aria_generate(ctx, &p, &audio);
    if (rc != 0 || !audio) { fprintf(stderr, "generate error: %s\n", aria_last_error()); aria_free(ctx); return 1; }

    if (aria_wav_write(out_path, audio, 32) != 0) {
        fprintf(stderr, "failed to write %s\n", out_path);
        aria_audio_free(audio); aria_free(ctx); return 1;
    }
    printf("wrote %s (%.2fs)\n", out_path, (double)audio->num_frames / audio->sample_rate);

    aria_audio_free(audio);
    aria_free(ctx);
    return 0;
}
