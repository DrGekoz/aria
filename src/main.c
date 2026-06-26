/*
 * main.c - aria CLI
 *
 * Phase 0 subcommands:
 *   aria -m <dir> --info
 *   aria -m <dir> --list-tensors [prefix]
 *   aria --wav-roundtrip <in.wav> <out.wav>
 *
 * Phase 1 (text-to-audio):
 *   aria -m <dir> -p "prompt" -d 30 -s 8 -o out.wav
 */

#include "aria.h"
#include "aria_wav.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog) {
    fprintf(stderr,
        "aria - audio diffusion inference runtime\n\n"
        "Usage:\n"
        "  %s -m <model_dir> --info\n"
        "  %s -m <model_dir> --list-tensors [prefix]\n"
        "  %s --wav-roundtrip <in.wav> <out.wav>\n"
        "  %s -m <model_dir> -p \"prompt\" -d <seconds> -s <steps> -o out.wav   (Phase 1)\n",
        prog, prog, prog, prog);
}

static int cmd_wav_roundtrip(const char *in, const char *out) {
    aria_audio *a = aria_wav_read(in);
    if (!a) { fprintf(stderr, "failed to read %s\n", in); return 1; }
    printf("read: sr=%d ch=%d frames=%lld (%.2fs)\n",
           a->sample_rate, a->channels, (long long)a->num_frames,
           (double)a->num_frames / a->sample_rate);
    int rc = aria_wav_write(out, a, 32);
    if (rc != 0) { fprintf(stderr, "failed to write %s\n", out); aria_audio_free(a); return 1; }
    printf("wrote: %s (float32)\n", out);
    aria_audio_free(a);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    const char *model_dir = NULL;
    const char *prompt = NULL;
    const char *out_path = "out.wav";
    const char *list_prefix = NULL;
    int do_info = 0, do_list = 0, do_generate = 0;
    float seconds = 15.0f;
    int steps = 8;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--wav-roundtrip") == 0 && i + 2 < argc) {
            return cmd_wav_roundtrip(argv[i + 1], argv[i + 2]);
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--info") == 0) {
            do_info = 1;
        } else if (strcmp(argv[i], "--list-tensors") == 0) {
            do_list = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') list_prefix = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i]; do_generate = 1;
        } else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            seconds = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]); return 0;
        } else {
            fprintf(stderr, "unknown/incomplete arg: %s\n", argv[i]);
            usage(argv[0]); return 1;
        }
    }

    if (!model_dir) { usage(argv[0]); return 1; }

    aria_ctx *ctx = aria_load(model_dir);
    if (!ctx) { fprintf(stderr, "load error: %s\n", aria_last_error()); return 1; }

    printf("model: %s | type=%s | sr=%d ch=%d | tensors=%d\n",
           model_dir, aria_model_type(ctx), aria_sample_rate(ctx),
           aria_audio_channels(ctx), aria_num_tensors(ctx));

    int rc = 0;
    if (do_list) {
        aria_list_tensors(ctx, list_prefix);
    } else if (do_generate) {
        aria_gen_params p = ARIA_GEN_PARAMS_DEFAULT;
        p.prompt = prompt;
        p.seconds_total = seconds;
        p.steps = steps;
        aria_audio *audio = NULL;
        rc = aria_generate(ctx, &p, &audio);
        if (rc == 0 && audio) {
            aria_wav_write(out_path, audio, 32);
            printf("wrote %s (%.2fs)\n", out_path, (double)audio->num_frames / audio->sample_rate);
            aria_audio_free(audio);
        } else {
            fprintf(stderr, "generate error: %s\n", aria_last_error());
            rc = 1;
        }
    } else if (do_info) {
        /* header already printed */
    }

    aria_free(ctx);
    return rc;
}
