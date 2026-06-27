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
#include <time.h>
#if defined(ARIA_BLAS) && defined(_OPENMP)
#include <omp.h>
#endif

static void usage(const char *prog) {
    fprintf(stderr,
        "aria - audio diffusion inference runtime\n\n"
        "Usage:\n"
        "  %s -m <model_dir> --info\n"
        "  %s -m <model_dir> --list-tensors [prefix]\n"
        "  %s --wav-roundtrip <in.wav> <out.wav>\n"
        "  %s -m <dir> -p \"prompt\" -d <seconds> -s <steps> --seed <n> [--device auto|cpu|cuda] -o out.wav\n"
        "  %s -m <dir> --uncond -d <seconds> -o out.wav\n"
        "  %s -m <dir> --prompt-embed <prompt.atns> -d <seconds> -o out.wav\n"
        "  %s -m <dir> -p \"prompt\" -d <total> --continue <in.wav> -o out.wav   (extend a clip)\n"
        "  %s -m <dir> -p \"prompt\" --inpaint <in.wav> --from <s> --to <s> -o out.wav  (regenerate a region)\n"
        "    --device auto (default) runs the DiT device-resident on the GPU when one fits, else CPU\n"
        "    --precision fp32|q8|q4 selects the CPU DiT weight precision (q8/q4 force the CPU path)\n"
        "    --load-quant <file.aria> loads a pre-quantized DiT from aria-quantize (CPU)\n"
        "    --continue/--inpaint run on the CPU DiT; the init WAV must be at the model sample rate\n",
        prog, prog, prog, prog, prog, prog, prog, prog);
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
#if defined(ARIA_BLAS) && defined(_OPENMP)
    /* The installed OpenBLAS uses its own pthread pool; let it own the cores and
     * keep aria's OpenMP single-threaded to avoid oversubscription. (With the
     * OpenMP build of OpenBLAS, both share one runtime and this isn't needed.) */
    omp_set_num_threads(1);
#endif
    if (argc < 2) { usage(argv[0]); return 1; }

    const char *model_dir = NULL;
    const char *prompt = NULL;
    const char *prompt_embed = NULL;
    const char *out_path = "out.wav";
    const char *list_prefix = NULL;
    int do_info = 0, do_list = 0, do_generate = 0;
    float seconds = 15.0f;
    int steps = 8;
    long long seed = -1;
    aria_device device = ARIA_DEVICE_AUTO;
    int bench = 1;   /* --bench N: generate N times (model resident) for warm timing */
    const char *init_audio = NULL;          /* continue / inpaint source WAV */
    float inpaint_from = 0.0f, inpaint_to = 0.0f;
    int inpaint_continue = 0;
    aria_dtype precision = ARIA_F32;
    const char *load_quant = NULL;

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
        } else if (strcmp(argv[i], "--prompt-embed") == 0 && i + 1 < argc) {
            prompt_embed = argv[++i]; do_generate = 1;
        } else if (strcmp(argv[i], "--uncond") == 0) {
            do_generate = 1;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            seconds = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            const char *d = argv[++i];
            if (strcmp(d, "cpu") == 0) device = ARIA_DEVICE_CPU;
            else if (strcmp(d, "cuda") == 0 || strcmp(d, "gpu") == 0) device = ARIA_DEVICE_CUDA;
            else device = ARIA_DEVICE_AUTO;
        } else if (strcmp(argv[i], "--bench") == 0 && i + 1 < argc) {
            bench = atoi(argv[++i]); if (bench < 1) bench = 1;
        } else if (strcmp(argv[i], "--inpaint") == 0 && i + 1 < argc) {
            init_audio = argv[++i]; do_generate = 1;
        } else if (strcmp(argv[i], "--continue") == 0 && i + 1 < argc) {
            init_audio = argv[++i]; inpaint_continue = 1; do_generate = 1;
        } else if (strcmp(argv[i], "--from") == 0 && i + 1 < argc) {
            inpaint_from = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--to") == 0 && i + 1 < argc) {
            inpaint_to = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--precision") == 0 && i + 1 < argc) {
            if (aria_dtype_parse(argv[++i], &precision) != 0) {
                fprintf(stderr, "unknown --precision %s (use fp32|fp16|bf16|q8|q4)\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--load-quant") == 0 && i + 1 < argc) {
            load_quant = argv[++i];
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
        p.prompt_embed_path = prompt_embed;
        p.seconds_total = seconds;
        p.steps = steps;
        p.seed = seed;
        p.device = device;
        p.precision = precision;
        p.load_quant = load_quant;
        p.init_audio = init_audio;
        p.inpaint_from_s = inpaint_from;
        p.inpaint_to_s = inpaint_to;
        p.inpaint_continue = inpaint_continue;
        double best = 1e9;
        for (int b = 0; b < bench && rc == 0; b++) {
            aria_audio *audio = NULL;
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            rc = aria_generate(ctx, &p, &audio);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double gen_s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
            if (rc != 0 || !audio) { fprintf(stderr, "generate error: %s\n", aria_last_error()); rc = 1; break; }
            if (gen_s < best) best = gen_s;
            if (bench > 1) fprintf(stderr, "  [bench %d/%d] %.2fs\n", b + 1, bench, gen_s);
            if (b == bench - 1) {
                aria_wav_write(out_path, audio, 32);
                printf("wrote %s (%.2fs audio, generated in %.2fs%s)\n",
                       out_path, (double)audio->num_frames / audio->sample_rate, best,
                       bench > 1 ? " warm-min" : "");
            }
            aria_audio_free(audio);
        }
    } else if (do_info) {
        /* header already printed */
    }

    aria_free(ctx);
    return rc;
}
