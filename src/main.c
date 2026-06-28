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
#include "aria_hpss.h"
#include "aria_parity.h"   /* .atns reader for --steer direction loading (E12.7) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>        /* sqrt for the phase-aligned crossfade */
#include <pthread.h>     /* writer thread so generation overlaps playback (--stream -o -) */
#include <unistd.h>      /* isatty */
#include <sys/select.h>  /* non-blocking stdin for live --stream prompting */

/* per-step progress, drawn on one line; only attached when stderr is a TTY */
static void cli_progress(int step, int total, void *user) {
    (void)user;
    fprintf(stderr, "\r  denoising [%d/%d]%s", step, total, step == total ? "\n" : "");
    fflush(stderr);
}

/* ---- unified table-driven CLI (argp-like, no external deps) ---------------------
 * OPTS is the single source of truth: it drives both parsing and --help, so they
 * cannot drift. cli_parse fills a cli_config; main() dispatches off its fields. */
typedef struct {
    const char *model_dir, *prompt, *prompt_embed, *out_path, *list_prefix;
    const char *init_audio, *load_quant, *util_a, *util_b;
    float seconds, inpaint_from, inpaint_to, stream_chunk, stream_context;
    int steps, bench, stream_chunks;
    long long seed;
    aria_device device;
    aria_dtype precision;
    int do_info, do_list, do_generate, do_stream, inpaint_continue, stream_hold, rng_torch;
    int util;                               /* 0 none / 1 wav-roundtrip / 2 hpss-test */
    const char *steer_specs[16]; int n_steer;   /* E12: repeatable --steer site:layer:dir.atns:scale:lo-hi */
} cli_config;

static cli_config cli_defaults(void) {
    cli_config c = {0};
    c.out_path = "out.wav"; c.seconds = 15.0f; c.steps = 8; c.bench = 1;
    c.seed = -1; c.device = ARIA_DEVICE_AUTO; c.precision = ARIA_F32;
    c.stream_chunk = 2.0f; c.stream_context = 6.0f; c.stream_chunks = 8;
    return c;
}

enum {   /* keys: short flags use their ASCII char; long-only options use ids past 256 */
    K_MODEL='m', K_PROMPT='p', K_DUR='d', K_STEPS='s', K_OUT='o', K_HELP='h',
    K_SEED=256, K_DEVICE, K_BENCH, K_UNCOND, K_PEMB, K_INFO, K_LIST,
    K_INPAINT, K_CONTINUE, K_FROM, K_TO, K_PREC, K_LOADQ, K_RNG,
    K_STREAM, K_CHUNK, K_CTX, K_CHUNKS, K_HOLD, K_WAVRT, K_HPSS, K_STEER,
};
typedef enum { A_NONE, A_ONE, A_TWO, A_OPT } argkind;  /* flag / 1 arg / 2 args / optional 1 arg */
typedef struct {
    int key; const char *lng; char shrt; argkind arg;
    const char *meta, *group, *help;
} opt_spec;

static const opt_spec OPTS[] = {
    {K_MODEL,    "model",        'm', A_ONE,  "<dir>",      "core",     "model directory (required, except for utilities)"},
    {K_OUT,      "out",          'o', A_ONE,  "<file>",     "core",     "output WAV ('-' = raw f32 to stdout, for --stream)"},
    {K_HELP,     "help",         'h', A_NONE, NULL,         "core",     "show this help and exit"},

    {K_PROMPT,   "prompt",       'p', A_ONE,  "<text>",     "generate", "text prompt -> audio"},
    {K_PEMB,     "prompt-embed",  0,  A_ONE,  "<file>",     "generate", "precomputed prompt embedding (.atns)"},
    {K_UNCOND,   "uncond",        0,  A_NONE, NULL,         "generate", "unconditional generation"},
    {K_DUR,      "",             'd', A_ONE,  "<sec>",      "generate", "duration in seconds (default 15)"},
    {K_STEPS,    "",             's', A_ONE,  "<n>",        "generate", "denoise steps (default 8)"},
    {K_SEED,     "seed",          0,  A_ONE,  "<n>",        "generate", "RNG seed (default random)"},
    {K_DEVICE,   "device",        0,  A_ONE,  "<dev>",      "generate", "cpu | cuda | auto (default auto)"},
    {K_PREC,     "precision",     0,  A_ONE,  "<p>",        "generate", "fp32 | fp16 | bf16 | q8 | q4 (q8/q4 force CPU)"},
    {K_LOADQ,    "load-quant",    0,  A_ONE,  "<file>",     "generate", "load a pre-quantized .aria DiT overlay"},
    {K_RNG,      "rng",           0,  A_ONE,  "<mode>",     "generate", "xoshiro (default) | torch (parity)"},
    {K_STEER,    "steer",         0,  A_ONE,  "<spec>",     "generate", "activation steering site:layer:dir.atns:scale:lo-hi (repeatable; forces CPU)"},
    {K_BENCH,    "bench",         0,  A_ONE,  "<n>",        "generate", "generate N times resident, report warm-min"},

    {K_CONTINUE, "continue",      0,  A_ONE,  "<in.wav>",   "edit",     "extend a clip (GPU or CPU)"},
    {K_INPAINT,  "inpaint",       0,  A_ONE,  "<in.wav>",   "edit",     "regenerate a region of a clip"},
    {K_FROM,     "from",          0,  A_ONE,  "<sec>",      "edit",     "inpaint region start"},
    {K_TO,       "to",            0,  A_ONE,  "<sec>",      "edit",     "inpaint region end"},

    {K_STREAM,   "stream",        0,  A_NONE, NULL,         "stream",   "continuous sliding-window generation"},
    {K_CHUNK,    "chunk",         0,  A_ONE,  "<sec>",      "stream",   "seconds emitted per chunk (default 2)"},
    {K_CTX,      "context",       0,  A_ONE,  "<sec>",      "stream",   "rolling context seconds (default 6)"},
    {K_CHUNKS,   "chunks",        0,  A_ONE,  "<n>",        "stream",   "number of chunks (default 8)"},
    {K_HOLD,     "hold",          0,  A_NONE, NULL,         "stream",   "hold a steady HPSS drum loop"},

    {K_INFO,     "info",          0,  A_NONE, NULL,         "inspect",  "print model info and exit"},
    {K_LIST,     "list-tensors",  0,  A_OPT,  "[prefix]",   "inspect",  "list tensors (optional name prefix)"},

    {K_WAVRT,    "wav-roundtrip", 0,  A_TWO,  "<in> <out>", "util",     "decode + re-encode a WAV (no model)"},
    {K_HPSS,     "hpss-test",     0,  A_ONE,  "<in.wav>",   "util",     "write harmonic.wav + percussive.wav (no model)"},
    {0}
};

/* apply one matched option to the config; returns 0 ok, 1 error (message printed). */
static int cli_apply(int key, char **a, cli_config *c) {
    switch (key) {
    case K_MODEL:  c->model_dir = a[0]; break;
    case K_OUT:    c->out_path = a[0]; break;
    case K_PROMPT: c->prompt = a[0]; c->do_generate = 1; break;
    case K_PEMB:   c->prompt_embed = a[0]; c->do_generate = 1; break;
    case K_UNCOND: c->do_generate = 1; break;
    case K_DUR:    c->seconds = (float)atof(a[0]); break;
    case K_STEPS:  c->steps = atoi(a[0]); break;
    case K_SEED:   c->seed = atoll(a[0]); break;
    case K_BENCH:  c->bench = atoi(a[0]); if (c->bench < 1) c->bench = 1; break;
    case K_DEVICE:
        if (!strcmp(a[0], "cpu")) c->device = ARIA_DEVICE_CPU;
        else if (!strcmp(a[0], "cuda") || !strcmp(a[0], "gpu")) c->device = ARIA_DEVICE_CUDA;
        else c->device = ARIA_DEVICE_AUTO;
        break;
    case K_PREC:
        if (aria_dtype_parse(a[0], &c->precision) != 0) {
            fprintf(stderr, "unknown --precision %s (use fp32|fp16|bf16|q8|q4)\n", a[0]); return 1;
        }
        break;
    case K_LOADQ:  c->load_quant = a[0]; break;
    case K_RNG:
        if (!strcmp(a[0], "torch")) c->rng_torch = 1;
        else if (!strcmp(a[0], "xoshiro")) c->rng_torch = 0;
        else { fprintf(stderr, "unknown --rng %s (use xoshiro|torch)\n", a[0]); return 1; }
        break;
    case K_CONTINUE: c->init_audio = a[0]; c->inpaint_continue = 1; c->do_generate = 1; break;
    case K_INPAINT:  c->init_audio = a[0]; c->do_generate = 1; break;
    case K_FROM:   c->inpaint_from = (float)atof(a[0]); break;
    case K_TO:     c->inpaint_to = (float)atof(a[0]); break;
    case K_STREAM: c->do_stream = 1; break;
    case K_CHUNK:  c->stream_chunk = (float)atof(a[0]); break;
    case K_CTX:    c->stream_context = (float)atof(a[0]); break;
    case K_CHUNKS: c->stream_chunks = atoi(a[0]); break;
    case K_HOLD:   c->stream_hold = 1; break;
    case K_STEER:
        if (c->n_steer >= 16) { fprintf(stderr, "too many --steer (max 16)\n"); return 1; }
        c->steer_specs[c->n_steer++] = a[0]; break;
    case K_INFO:   c->do_info = 1; break;
    case K_LIST:   c->do_list = 1; if (a[0]) c->list_prefix = a[0]; break;
    case K_WAVRT:  c->util = 1; c->util_a = a[0]; c->util_b = a[1]; break;
    case K_HPSS:   c->util = 2; c->util_a = a[0]; break;
    default: return 1;
    }
    return 0;
}

static void cli_help(const char *prog) {
    fprintf(stderr, "aria - audio diffusion inference runtime\n\nUsage: %s [options]\n", prog);
    static const char *groups[] = {"core","generate","edit","stream","inspect","util"};
    static const char *titles[] = {"Core","Generation","Continue / inpaint","Streaming","Inspect","Utilities (no model)"};
    for (size_t g = 0; g < sizeof(groups)/sizeof(*groups); g++) {
        fprintf(stderr, "\n%s:\n", titles[g]);
        for (const opt_spec *o = OPTS; o->key; o++) {
            if (strcmp(o->group, groups[g])) continue;
            char left[56]; int n = 0;
            if (o->shrt) n += snprintf(left+n, sizeof left-n, "-%c", o->shrt);
            if (o->shrt && o->lng && o->lng[0]) n += snprintf(left+n, sizeof left-n, ", ");
            if (o->lng && o->lng[0]) n += snprintf(left+n, sizeof left-n, "--%s", o->lng);
            if (o->meta) snprintf(left+n, sizeof left-n, " %s", o->meta);
            fprintf(stderr, "  %-26s %s\n", left, o->help);
        }
    }
    fprintf(stderr, "\nLive play + re-steer (type a new prompt + Enter any time):\n"
                    "  %s -m <dir> --stream -o - -p \"...\" | play -t raw -r <sr> -e float -b 32 -c <ch> -\n", prog);
}

/* parse argv into cfg; returns 0 ok, 1 error, or -1 when --help was shown. */
static int cli_parse(int argc, char **argv, cli_config *c) {
    *c = cli_defaults();
    for (int i = 1; i < argc; i++) {
        const char *tok = argv[i];
        const opt_spec *o = NULL;
        if (tok[0] == '-' && tok[1] == '-') {
            for (const opt_spec *s = OPTS; s->key; s++)
                if (s->lng && s->lng[0] && !strcmp(tok + 2, s->lng)) { o = s; break; }
        } else if (tok[0] == '-' && tok[1] && !tok[2]) {
            for (const opt_spec *s = OPTS; s->key; s++)
                if (s->shrt == tok[1]) { o = s; break; }
        }
        if (!o) { fprintf(stderr, "unknown option: %s\n", tok); cli_help(argv[0]); return 1; }
        if (o->key == K_HELP) { cli_help(argv[0]); return -1; }
        char *args[2] = {NULL, NULL};
        int need = (o->arg == A_ONE) ? 1 : (o->arg == A_TWO) ? 2 : 0;
        for (int k = 0; k < need; k++) {
            if (i + 1 >= argc) { fprintf(stderr, "option %s needs %d argument(s)\n", tok, need); return 1; }
            args[k] = argv[++i];
        }
        if (o->arg == A_OPT && i + 1 < argc && argv[i + 1][0] != '-') args[0] = argv[++i];
        if (cli_apply(o->key, args, c)) return 1;
    }
    return 0;
}

/* E12: parse the repeatable --steer specs (site:layer:dir.atns:scale:lo-hi) into a steer set,
 * loading each .atns direction. `items` is caller storage [>=n_steer]; on success fills `set`
 * (free each set->items[i].dir with free_steer_set after generation). Returns 0, or 1 on error. */
static int build_steer_set(const cli_config *cfg, aria_steer *items, aria_steer_set *set) {
    set->items = items; set->n = 0;
    for (int i = 0; i < cfg->n_steer; i++) {
        char buf[1024];
        snprintf(buf, sizeof buf, "%s", cfg->steer_specs[i]);
        char *site = strtok(buf, ":"), *layer = strtok(NULL, ":"), *path = strtok(NULL, ":");
        char *scale = strtok(NULL, ":"), *win = strtok(NULL, ":");
        if (!site || !layer || !path || !scale) {
            fprintf(stderr, "bad --steer '%s' (want site:layer:dir.atns:scale:lo-hi)\n", cfg->steer_specs[i]);
            return 1;
        }
        aria_steer_site s;
        if      (!strcmp(site, "residual")) s = ARIA_STEER_RESIDUAL;
        else if (!strcmp(site, "latent"))   s = ARIA_STEER_LATENT;
        else if (!strcmp(site, "cond"))     s = ARIA_STEER_COND;
        else { fprintf(stderr, "unknown --steer site '%s' (residual|latent|cond)\n", site); return 1; }
        aria_parity_tensor t;
        if (aria_parity_load(path, &t) != 0) { fprintf(stderr, "cannot load steer direction %s\n", path); return 1; }
        float *d = malloc((size_t)t.numel * sizeof(float));
        memcpy(d, t.data, (size_t)t.numel * sizeof(float));
        aria_parity_free(&t);
        int lo = 0, hi = 1 << 30;
        if (win) sscanf(win, "%d-%d", &lo, &hi);
        aria_steer *it = &items[set->n];
        it->site = s; it->layer = atoi(layer); it->dir = d; it->dim = (int)t.numel;
        it->scale = (float)atof(scale); it->step_lo = lo; it->step_hi = hi;
        fprintf(stderr, "[steer] %s layer=%d dim=%d scale=%.4g steps[%d,%d] <- %s\n",
                site, it->layer, it->dim, it->scale, lo, hi, path);
        set->n++;
    }
    return 0;
}
static void free_steer_set(aria_steer_set *set) {
    for (int i = 0; i < set->n; i++) free((void *)set->items[i].dir);
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

/* --hpss-test <in.wav>: separate -> harmonic.wav + percussive.wav, report split. */
static int cmd_hpss_test(const char *in) {
    aria_audio *a = aria_wav_read(in);
    if (!a) { fprintf(stderr, "hpss: cannot read %s\n", in); return 1; }
    int64_t nf = a->num_frames; int ch = a->channels;
    float *h = malloc((size_t)nf * ch * sizeof(float)), *p = malloc((size_t)nf * ch * sizeof(float));
    aria_hpss_separate(a->data, nf, ch, h, p);
    double res = 0, eh = 0, ep = 0, ei = 0; int64_t M = nf * ch;
    for (int64_t i = 0; i < M; i++) {
        double r = a->data[i] - (h[i] + p[i]);
        res += r * r; eh += (double)h[i] * h[i]; ep += (double)p[i] * p[i]; ei += (double)a->data[i] * a->data[i];
    }
    fprintf(stderr, "hpss: residual-rms=%.2e  harmonic=%.0f%%  percussive=%.0f%% (of input energy)\n",
            sqrt(res / M), 100 * eh / (ei + 1e-9), 100 * ep / (ei + 1e-9));
    aria_audio ho = { a->sample_rate, ch, nf, h }, po = { a->sample_rate, ch, nf, p };
    aria_wav_write("harmonic.wav", &ho, 32); aria_wav_write("percussive.wav", &po, 32);
    fprintf(stderr, "hpss: wrote harmonic.wav + percussive.wav\n");
    free(h); free(p); aria_audio_free(a);
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

/* Append `src` onto `outp`, phase-aligning it to the existing tail before crossfading.
 * The continuation is only beat-plausible, not sample-accurate, so a plain crossfade
 * "flams" misaligned drum hits; here we cross-correlate `src`'s start against the output's
 * last ~60 ms over a 0..`maxlag` search, shift `src` by the best lag, then linearly
 * crossfade `xf` frames. maxlag=0 -> plain crossfade (used for the first segment). */
static void stream_append_aligned(aria_audio *outp, int64_t *written, int64_t cap,
                                  const aria_audio *src, int64_t xf, int64_t maxlag) {
    int ch = outp->channels; int64_t n = src->num_frames, lag = 0;
    const int64_t clen = 1764, cstride = 2;   /* ~40 ms window, every 2nd sample */
    int64_t W = *written;
    if (W >= clen && maxlag > 0 && n >= clen + maxlag) {
        const float *ref = outp->data + (W - clen) * ch;   /* mono via channel sum */
        double best = -1e30;
        for (int64_t L = 0; L <= maxlag; L++) {
            const float *cand = src->data + L * ch;
            double dot = 0, ea = 0, eb = 0;
            for (int64_t i = 0; i < clen; i += cstride) {
                float a = ref[i * ch] + (ch > 1 ? ref[i * ch + 1] : 0.0f);
                float b = cand[i * ch] + (ch > 1 ? cand[i * ch + 1] : 0.0f);
                dot += (double)a * b; ea += (double)a * a; eb += (double)b * b;
            }
            double corr = dot / (sqrt(ea * eb) + 1e-9);
            if (corr > best) { best = corr; lag = L; }
        }
    }
    int64_t xfe = xf;
    if (xfe > *written) xfe = *written;
    if (lag + xfe > n) xfe = n - lag;
    if (xfe < 0) xfe = 0;
    for (int64_t i = 0; i < xfe; i++) {
        float w = (float)(i + 1) / (float)(xfe + 1);
        int64_t o = (*written - xfe + i) * ch;
        for (int c = 0; c < ch; c++)
            outp->data[o + c] = (1.0f - w) * outp->data[o + c] + w * src->data[(lag + i) * ch + c];
    }
    int64_t rest = n - lag - xfe;
    if (*written + rest > cap) rest = cap - *written;
    if (rest > 0) memcpy(outp->data + (*written) * ch, src->data + (lag + xfe) * ch, (size_t)rest * ch * sizeof(float));
    *written += rest;
}

/* add the looped percussive `held` (held_len frames) into `dst`[n] starting at output
 * position `pos` (modulo tiling) — the held drum groove, steady across chunks. */
static void stream_add_tiled(float *dst, int64_t n, int ch, int64_t pos,
                             const float *held, int64_t held_len) {
    if (held_len <= 0) return;
    for (int64_t j = 0; j < n; j++) {
        int64_t k = (pos + j) % held_len;
        for (int c = 0; c < ch; c++) dst[j * ch + c] += held[k * ch + c];
    }
}

/* Bounded FIFO of settled audio chunks. The generator (main thread) pushes; a writer
 * thread drains to stdout. This decouples generation from the blocking pipe write so the
 * next chunk is generated WHILE the player drains the current one, instead of alternating
 * gen/play. The bound throttles a faster-than-realtime producer (e.g. a future GPU path). */
typedef struct schunk { float *data; int64_t n; struct schunk *next; } schunk;
typedef struct {
    schunk *head, *tail; int count, done;
    pthread_mutex_t m; pthread_cond_t ne, nf;
} squeue;
#define SQ_MAX 32   /* up to ~32 settled regions buffered ahead */

static void sq_push(squeue *q, float *data, int64_t n) {
    schunk *c = malloc(sizeof *c); c->data = data; c->n = n; c->next = NULL;
    pthread_mutex_lock(&q->m);
    while (q->count >= SQ_MAX) pthread_cond_wait(&q->nf, &q->m);
    if (q->tail) q->tail->next = c; else q->head = c;
    q->tail = c; q->count++;
    pthread_cond_signal(&q->ne);
    pthread_mutex_unlock(&q->m);
}
static void *sq_writer(void *arg) {   /* drains the queue to stdout at the player's pace */
    squeue *q = arg;
    for (;;) {
        pthread_mutex_lock(&q->m);
        while (!q->head && !q->done) pthread_cond_wait(&q->ne, &q->m);
        schunk *c = q->head;
        if (!c) { pthread_mutex_unlock(&q->m); break; }   /* done and drained */
        q->head = c->next; if (!q->head) q->tail = NULL; q->count--;
        pthread_cond_signal(&q->nf);
        pthread_mutex_unlock(&q->m);
        fwrite(c->data, sizeof(float), (size_t)c->n, stdout);
        fflush(stdout);
        free(c->data); free(c);
    }
    return NULL;
}

/* settle the frames [*emitted, upto) of `outp` (the harmonic timeline) as raw interleaved
 * float32, mixing in the held drum loop, and hand them to the writer thread. Lets
 * `aria --stream -o - | play -t raw -r <sr> -e float -b 32 -c <ch> -` play live. */
static void stream_flush(const aria_audio *outp, const float *held, int64_t held_len,
                         int ch, int64_t *emitted, int64_t upto, squeue *q) {
    if (upto <= *emitted) return;
    int64_t n = upto - *emitted;
    float *tmp = malloc((size_t)n * ch * sizeof(float));
    memcpy(tmp, outp->data + (*emitted) * ch, (size_t)n * ch * sizeof(float));
    stream_add_tiled(tmp, n, ch, *emitted, held, held_len);   /* no-op if held_len==0 */
    sq_push(q, tmp, n * ch);   /* the writer thread frees tmp after writing it out */
    *emitted = upto;
}

/* Streaming / interactive generation: keep the model resident and emit `emit_s`-second
 * segments by sliding-window continuation over a `context_s` rolling context. The naive
 * tail-inpaint fades (SA3 makes the regenerated end an outro), so each continuation
 * regenerates context + skip + emit + tail and emits only the STRONG BODY — skipping the
 * ~1.5 s post-context seam and discarding the ~3 s fade-out (profile measured on a long
 * continuation) — crossfaded onto the output. Each step re-reads the prompt (live
 * re-steering on a TTY). `emit_s` is the --chunk value. */
static int cmd_stream(aria_ctx *ctx, aria_gen_params *p, float emit_s, float context_s,
                      int n_chunks, int hold, const char *out_path) {
    int sr = aria_sample_rate(ctx), ch = aria_audio_channels(ctx);
    const float skip_s = 1.5f, tail_s = 3.0f, xfade_s = 0.25f;   /* seam / fade / crossfade */
    int64_t ctx_fr = (int64_t)(context_s * sr), xf_fr = (int64_t)(xfade_s * sr);
    int64_t maxlag = (int64_t)(0.18f * sr);   /* phase-align search range (~< one beat) */
    int interactive = isatty(fileno(stdin));
    char promptbuf[1024];
    float *held = NULL; int64_t held_len = 0; const char *held_prompt = NULL;  /* --hold drum loop */
    int to_stdout = (strcmp(out_path, "-") == 0);   /* -o - : stream raw f32 to stdout for a player */
    int64_t emitted = 0;
    squeue q = { .m = PTHREAD_MUTEX_INITIALIZER, .ne = PTHREAD_COND_INITIALIZER, .nf = PTHREAD_COND_INITIALIZER };
    pthread_t writer; int have_writer = 0;
    if (to_stdout) { pthread_create(&writer, NULL, sq_writer, &q); have_writer = 1; }

    int64_t cap = (int64_t)((context_s + emit_s) * sr) + (int64_t)n_chunks * ((int64_t)(emit_s * sr) + maxlag) + sr;
    aria_audio *outp = aria_audio_alloc(sr, ch, cap);
    if (!outp) { fprintf(stderr, "stream: OOM\n"); return 1; }
    int64_t written = 0;
    aria_audio *context = NULL;

    const char *devname = p->device == ARIA_DEVICE_CPU ? "CPU" : p->device == ARIA_DEVICE_CUDA ? "CUDA" : "auto";
    fprintf(stderr, "[stream] emit=%.1fs context=%.1fs (skip %.1f / tail %.1f / xfade %.2f) steps=%d (%s)%s\n",
            emit_s, context_s, skip_s, tail_s, xfade_s, p->steps, devname,
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
        if (rc != 0 || !win) {
            fprintf(stderr, "stream: generate failed: %s\n", aria_last_error());
            if (have_writer) { pthread_mutex_lock(&q.m); q.done = 1; pthread_cond_signal(&q.ne); pthread_mutex_unlock(&q.m); pthread_join(writer, NULL); }
            aria_audio_free(outp); aria_audio_free(context); return 1;
        }
        double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

        /* extra `maxlag` frames so the phase-align search has room to shift into */
        int64_t slice_len = emit_len + (i == 0 ? 0 : maxlag);
        if (emit_start + slice_len > win->num_frames) slice_len = win->num_frames - emit_start;
        aria_audio *emit = stream_slice(win, emit_start, slice_len);
        if (hold) {
            /* split the emit; `outp` accumulates only the HARMONIC, the drum groove is a
             * held loop tiled later -> drums stay perfectly steady across chunks. */
            int64_t en = emit->num_frames;
            float *H = malloc((size_t)en * ch * sizeof(float)), *P = malloc((size_t)en * ch * sizeof(float));
            aria_hpss_separate(emit->data, en, ch, H, P);
            if (!held || p->prompt != held_prompt) {   /* establish/reset the loop on chunk 0 or prompt change */
                free(held); held_len = en; held = malloc((size_t)held_len * ch * sizeof(float));
                memcpy(held, P, (size_t)held_len * ch * sizeof(float));
                held_prompt = p->prompt;
                fprintf(stderr, "[stream] drum loop set (%.1fs)\n", held_len / (double)sr);
            }
            memcpy(emit->data, H, (size_t)en * ch * sizeof(float));   /* emit -> harmonic only */
            free(H); free(P);
        }
        stream_append_aligned(outp, &written, cap, emit, (i == 0) ? 0 : xf_fr, (i == 0) ? 0 : maxlag);
        aria_audio_free(emit);

        double es = emit_len / (double)sr;
        fprintf(stderr, "[stream] chunk %d/%d: emit %.1fs (win %.1fs) in %.2fs (RTF %.1fx)  \"%s\"\n",
                i + 1, n_chunks, es, win_s, dt, dt > 0 ? es / dt : 0.0, p->prompt ? p->prompt : "");

        aria_audio_free(context);
        context = stream_slice(outp, written - ctx_fr, ctx_fr);   /* just-emitted tail */
        if (hold) stream_add_tiled(context->data, ctx_fr, ch, written - ctx_fr, held, held_len);
        aria_audio_free(win);

        /* live playback: hand settled audio to the writer thread, holding back the crossfade
         * region (xf_fr) so the next chunk can still crossfade into it. The next chunk
         * generates while the writer feeds the player. */
        if (to_stdout) stream_flush(outp, held, held_len, ch, &emitted, written - xf_fr, &q);
    }
    int wrc = 0;
    if (to_stdout) {
        stream_flush(outp, held, held_len, ch, &emitted, written, &q);   /* final tail */
        pthread_mutex_lock(&q.m); q.done = 1; pthread_cond_signal(&q.ne); pthread_mutex_unlock(&q.m);
        pthread_join(writer, NULL); have_writer = 0;
        fprintf(stderr, "[stream] streamed %.1fs to stdout\n", written / (double)sr);
    } else {
        if (hold) stream_add_tiled(outp->data, written, ch, 0, held, held_len);   /* mix the steady drums in */
        outp->num_frames = written;
        wrc = aria_wav_write(out_path, outp, 32);
        fprintf(stderr, "[stream] wrote %s (%.1fs total)\n", out_path, written / (double)sr);
    }
    free(held);
    aria_audio_free(outp); aria_audio_free(context);
    return wrc;
}

int main(int argc, char **argv) {
    if (argc < 2) { cli_help(argv[0]); return 1; }

    cli_config cfg;
    int pr = cli_parse(argc, argv, &cfg);
    if (pr < 0) return 0;   /* --help shown */
    if (pr > 0) return 1;   /* parse error (message already printed) */

    /* standalone utilities run without loading a model */
    if (cfg.util == 1) return cmd_wav_roundtrip(cfg.util_a, cfg.util_b);
    if (cfg.util == 2) return cmd_hpss_test(cfg.util_a);

    if (!cfg.model_dir) { cli_help(argv[0]); return 1; }

    aria_ctx *ctx = aria_load(cfg.model_dir);
    if (!ctx) { fprintf(stderr, "load error: %s\n", aria_last_error()); return 1; }

    fprintf(stderr, "model: %s | type=%s | sr=%d ch=%d | tensors=%d\n",
            cfg.model_dir, aria_model_type(ctx), aria_sample_rate(ctx),
            aria_audio_channels(ctx), aria_num_tensors(ctx));

    int rc = 0;
    if (cfg.do_stream) {
        aria_gen_params p = ARIA_GEN_PARAMS_DEFAULT;
        p.prompt = cfg.prompt; p.steps = cfg.steps; p.seed = cfg.seed;
        p.precision = cfg.precision; p.device = cfg.device;  /* GPU continuation supported (sm_70+) */
        p.rng_torch = cfg.rng_torch;
        rc = cmd_stream(ctx, &p, cfg.stream_chunk, cfg.stream_context, cfg.stream_chunks, cfg.stream_hold, cfg.out_path);
    } else if (cfg.do_list) {
        aria_list_tensors(ctx, cfg.list_prefix);
    } else if (cfg.do_generate) {
        aria_gen_params p = ARIA_GEN_PARAMS_DEFAULT;
        p.prompt = cfg.prompt;
        p.prompt_embed_path = cfg.prompt_embed;
        p.seconds_total = cfg.seconds;
        p.steps = cfg.steps;
        p.seed = cfg.seed;
        p.device = cfg.device;
        p.precision = cfg.precision;
        p.load_quant = cfg.load_quant;
        p.init_audio = cfg.init_audio;
        p.inpaint_from_s = cfg.inpaint_from;
        p.inpaint_to_s = cfg.inpaint_to;
        p.inpaint_continue = cfg.inpaint_continue;
        p.rng_torch = cfg.rng_torch;
        aria_steer steer_items[16]; aria_steer_set steer_set = {0};   /* E12 activation steering */
        if (cfg.n_steer > 0) {
            if (build_steer_set(&cfg, steer_items, &steer_set) != 0) { aria_free(ctx); return 1; }
            p.steer = &steer_set;
        }
        if (isatty(fileno(stderr))) p.progress = cli_progress;  /* live progress on a terminal */
        double best = 1e9;
        for (int b = 0; b < cfg.bench && rc == 0; b++) {
            aria_audio *audio = NULL;
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            rc = aria_generate(ctx, &p, &audio);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double gen_s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
            if (rc != 0 || !audio) { fprintf(stderr, "generate error: %s\n", aria_last_error()); rc = 1; break; }
            if (gen_s < best) best = gen_s;
            if (cfg.bench > 1) fprintf(stderr, "  [bench %d/%d] %.2fs\n", b + 1, cfg.bench, gen_s);
            if (b == cfg.bench - 1) {
                aria_wav_write(cfg.out_path, audio, 32);
                printf("wrote %s (%.2fs audio, generated in %.2fs%s)\n",
                       cfg.out_path, (double)audio->num_frames / audio->sample_rate, best,
                       cfg.bench > 1 ? " warm-min" : "");
            }
            aria_audio_free(audio);
        }
        free_steer_set(&steer_set);
    } else if (cfg.do_info) {
        /* header already printed */
    }

    aria_free(ctx);
    return rc;
}
