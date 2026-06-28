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
#include <unistd.h>      /* isatty */
#include <sys/select.h>  /* non-blocking stdin for live --stream prompting */

/* per-step progress, drawn on one line; only attached when stderr is a TTY */
static void cli_progress(int step, int total, void *user) {
    (void)user;
    fprintf(stderr, "\r  denoising [%d/%d]%s", step, total, step == total ? "\n" : "");
    fflush(stderr);
}

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
        "    --rng xoshiro (default) | torch  (torch = PyTorch-matched noise for reproduction)\n"
        "    --continue/--inpaint run on the CPU DiT; the init WAV must be at the model sample rate\n"
        "  Progress is drawn per denoise step when stderr is a terminal.\n",
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

/* read a new prompt from stdin if a line is waiting (non-blocking); 1 if updated */
static int stream_poll_prompt(char *buf, size_t cap) {
    fd_set fds; FD_ZERO(&fds); FD_SET(0, &fds);
    struct timeval tv = {0, 0};
    if (select(1, &fds, NULL, NULL, &tv) > 0 && FD_ISSET(0, &fds) && fgets(buf, (int)cap, stdin)) {
        size_t n = strlen(buf);
        while (n && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = 0;
        return n > 0;
    }
    return 0;
}

/* interleaved frames [start, start+len) of `a` -> new aria_audio */
static aria_audio *stream_slice(const aria_audio *a, int64_t start, int64_t len) {
    if (start < 0) start = 0;
    if (start + len > a->num_frames) len = a->num_frames - start;
    if (len < 0) len = 0;
    aria_audio *o = aria_audio_alloc(a->sample_rate, a->channels, len);
    if (o && len) memcpy(o->data, a->data + start * a->channels, (size_t)len * a->channels * sizeof(float));
    return o;
}

/* append `src` onto `outp` at *written, linearly crossfading its first `xf` frames over
 * the last `xf` already-written frames (smooths the seam between segments). */
static void stream_append_xfade(aria_audio *outp, int64_t *written, int64_t cap,
                                const aria_audio *src, int64_t xf) {
    int ch = outp->channels; int64_t n = src->num_frames;
    if (xf > *written) xf = *written;
    if (xf > n) xf = n;
    for (int64_t i = 0; i < xf; i++) {
        float w = (float)(i + 1) / (float)(xf + 1);
        int64_t o = (*written - xf + i) * ch;
        for (int c = 0; c < ch; c++)
            outp->data[o + c] = (1.0f - w) * outp->data[o + c] + w * src->data[i * ch + c];
    }
    int64_t rest = n - xf;
    if (*written + rest > cap) rest = cap - *written;
    if (rest > 0) memcpy(outp->data + (*written) * ch, src->data + xf * ch, (size_t)rest * ch * sizeof(float));
    *written += rest;
}

/* Streaming / interactive generation: keep the model resident and emit `emit_s`-second
 * segments by sliding-window continuation over a `context_s` rolling context. The naive
 * tail-inpaint fades (SA3 makes the regenerated end an outro), so each continuation
 * regenerates context + skip + emit + tail and emits only the STRONG BODY — skipping the
 * ~1.5 s post-context seam and discarding the ~3 s fade-out (profile measured on a long
 * continuation) — crossfaded onto the output. Each step re-reads the prompt (live
 * re-steering on a TTY). `emit_s` is the --chunk value. */
static int cmd_stream(aria_ctx *ctx, aria_gen_params *p, float emit_s, float context_s,
                      int n_chunks, const char *out_path) {
    int sr = aria_sample_rate(ctx), ch = aria_audio_channels(ctx);
    const float skip_s = 1.5f, tail_s = 3.0f, xfade_s = 0.25f;   /* seam / fade / crossfade */
    int64_t ctx_fr = (int64_t)(context_s * sr), xf_fr = (int64_t)(xfade_s * sr);
    int interactive = isatty(fileno(stdin));
    char promptbuf[1024];

    int64_t cap = (int64_t)((context_s + emit_s) * sr) + (int64_t)n_chunks * (int64_t)(emit_s * sr) + sr;
    aria_audio *outp = aria_audio_alloc(sr, ch, cap);
    if (!outp) { fprintf(stderr, "stream: OOM\n"); return 1; }
    int64_t written = 0;
    aria_audio *context = NULL;

    fprintf(stderr, "[stream] emit=%.1fs context=%.1fs (skip %.1f / tail %.1f / xfade %.2f) steps=%d (CPU)%s\n",
            emit_s, context_s, skip_s, tail_s, xfade_s, p->steps,
            interactive ? " — type a prompt + Enter to re-steer" : "");

    for (int i = 0; i < n_chunks; i++) {
        /* poll stdin (TTY or pipe) for a new prompt -> live re-steering between chunks */
        if (stream_poll_prompt(promptbuf, sizeof promptbuf)) {
            p->prompt = promptbuf;
            fprintf(stderr, "[stream] prompt -> \"%s\"\n", promptbuf);
        }
        /* i==0: plain text->audio, emit the strong front [0, context+emit] (drop its fade).
         * i>0 : continuation; emit the body [context+skip, context+skip+emit]. */
        float win_s = (i == 0) ? (context_s + emit_s + tail_s)
                               : (context_s + skip_s + emit_s + tail_s);
        int64_t emit_start = (i == 0) ? 0 : (int64_t)((context_s + skip_s) * sr);
        int64_t emit_len   = (i == 0) ? (int64_t)((context_s + emit_s) * sr)
                                      : (int64_t)(emit_s * sr);
        p->seconds_total = win_s;
        p->init_audio_mem = (i == 0) ? NULL : context;
        p->inpaint_continue = (i == 0) ? 0 : 1;

        aria_audio *win = NULL;
        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
        int rc = aria_generate(ctx, p, &win);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (rc != 0 || !win) { fprintf(stderr, "stream: generate failed: %s\n", aria_last_error()); aria_audio_free(outp); aria_audio_free(context); return 1; }
        double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

        if (emit_start + emit_len > win->num_frames) emit_len = win->num_frames - emit_start;
        aria_audio *emit = stream_slice(win, emit_start, emit_len);
        stream_append_xfade(outp, &written, cap, emit, (i == 0) ? 0 : xf_fr);
        aria_audio_free(emit);

        double es = emit_len / (double)sr;
        fprintf(stderr, "[stream] chunk %d/%d: emit %.1fs (win %.1fs) in %.2fs (RTF %.1fx)  \"%s\"\n",
                i + 1, n_chunks, es, win_s, dt, dt > 0 ? es / dt : 0.0, p->prompt ? p->prompt : "");

        aria_audio_free(context);
        context = stream_slice(outp, written - ctx_fr, ctx_fr);   /* strong, just-emitted tail */
        aria_audio_free(win);
    }
    outp->num_frames = written;
    int wrc = aria_wav_write(out_path, outp, 32);
    fprintf(stderr, "[stream] wrote %s (%.1fs total)\n", out_path, written / (double)sr);
    aria_audio_free(outp); aria_audio_free(context);
    return wrc;
}

int main(int argc, char **argv) {
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
    int rng_torch = 0;
    int do_stream = 0, stream_chunks = 8;
    float stream_chunk = 2.0f, stream_context = 6.0f;

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
        } else if (strcmp(argv[i], "--rng") == 0 && i + 1 < argc) {
            const char *m = argv[++i];
            if (strcmp(m, "torch") == 0) rng_torch = 1;
            else if (strcmp(m, "xoshiro") == 0) rng_torch = 0;
            else { fprintf(stderr, "unknown --rng %s (use xoshiro|torch)\n", m); return 1; }
        } else if (strcmp(argv[i], "--stream") == 0) {
            do_stream = 1;
        } else if (strcmp(argv[i], "--chunk") == 0 && i + 1 < argc) {
            stream_chunk = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--context") == 0 && i + 1 < argc) {
            stream_context = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--chunks") == 0 && i + 1 < argc) {
            stream_chunks = atoi(argv[++i]);
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
    if (do_stream) {
        aria_gen_params p = ARIA_GEN_PARAMS_DEFAULT;
        p.prompt = prompt; p.steps = steps; p.seed = seed;
        p.precision = precision; p.device = ARIA_DEVICE_CPU;  /* continuation is CPU-only */
        p.rng_torch = rng_torch;
        rc = cmd_stream(ctx, &p, stream_chunk, stream_context, stream_chunks, out_path);
    } else if (do_list) {
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
        p.rng_torch = rng_torch;
        if (isatty(fileno(stderr))) p.progress = cli_progress;  /* live progress on a terminal */
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
