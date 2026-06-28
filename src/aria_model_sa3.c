/*
 * aria_model_sa3.c - Stable Audio 3 model module: load + end-to-end generation
 * (text conditioning via precomputed prompt embedding until T5Gemma is ported).
 */

#include "aria_model.h"
#include "aria_sa3.h"
#include "aria_sa3_dit.h"
#include "aria_sa3_dec.h"
#include "aria_sa3_enc.h"
#include "aria_t5enc.h"
#include "aria_tokenizer.h"
#include "aria_sampler.h"
#include "aria_cond.h"
#include "aria_parity.h"
#include "aria_json.h"
#ifdef ARIA_CUDA
#include "aria_gpu.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/resource.h>

/* optional per-stage timing + memory (set ARIA_PROFILE=1) */
static double sa3_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }

typedef struct {
    aria_sa3_config cfg;
    aria_sa3_dit *dit;
    aria_sa3_dec *dec;
    aria_sa3_dec_medium *dec_med;  /* medium decoder (dim 1536); dec is NULL when medium */
    int is_medium;                 /* medium model: differential DiT + medium decoder (CPU) */
    aria_sa3_enc *enc_taae;   /* taae encoder, lazily loaded for continue/inpaint */
    int quant_loaded;         /* a packed .aria DiT overlay has been loaded (E9.2) */
    /* seconds_total NumberConditioner (borrowed from the mmap) */
    const float *sec_w;  /* [768,256] */
    const float *sec_b;  /* [768] */
    float sec_min, sec_max;
    int sample_rate;
    /* text path (lazily loaded on first text prompt) */
    aria_t5enc *enc;
    aria_tokenizer *tok;
    char *cached_prompt; float *cached_emb;   /* skip re-encoding an unchanged prompt (streaming) */
    /* persistent device DiT + decoder (weights uploaded once, reused across generations) */
    aria_cuda_dit *cdit;
    aria_cuda_dec *cdec;
    aria_cuda_dec_medium *cdec_med;   /* medium GPU decoder (dim 1536, banded attention) */
    int weights_advised;              /* madvise(DONTNEED) the host weights once (GPU path) */
} sa3_state;

/* Load the T5Gemma encoder + tokenizer on demand (text prompts only). */
static int sa3_ensure_text(aria_ctx *ctx, sa3_state *st) {
    if (st->enc && st->tok) return 0;
    if (!st->enc) st->enc = aria_t5enc_load(ctx->model_dir, "t5gemma-b-b-ul2", ctx->sf);
    if (!st->tok) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/t5gemma-b-b-ul2/aria_tokenizer.bin", ctx->model_dir);
        st->tok = aria_tokenizer_load(path);
    }
    if (!st->enc || !st->tok) {
        aria_set_error("generate: text prompts need the T5Gemma weights and the exported "
                       "tokenizer (run: python scripts/export_tokenizer.py <model_dir>)");
        return -1;
    }
    return 0;
}

static int sa3_detect(const char *model_type) {
    return strcmp(model_type, "diffusion_cond_inpaint") == 0 ||
           strcmp(model_type, "diffusion_cond") == 0;
}

static void *sa3_load(aria_ctx *ctx, safetensors_file_t *sf, const char *config_json) {
    sa3_state *st = calloc(1, sizeof(sa3_state));
    if (!st) { aria_set_error("sa3_load: oom"); return NULL; }

    if (aria_sa3_parse_config(config_json, &st->cfg) != 0) {
        aria_set_error("sa3_load: cannot parse diffusion/pretransform config");
        free(st); return NULL;
    }
    st->sample_rate = ctx->sample_rate;

    /* medium model: decoder.layers.1.weight is [1536,256] (vs [768,256] small-music) */
    const safetensor_t *projw = safetensors_find(sf, "pretransform.model.decoder.layers.1.weight");
    st->is_medium = projw && projw->ndim >= 1 && projw->shape[0] == 1536;

    st->dit = aria_sa3_dit_load(sf, &st->cfg);
    if (st->is_medium) st->dec_med = aria_sa3_dec_medium_load(sf);
    else               st->dec = aria_sa3_dec_load(sf);
    if (!st->dit || (st->is_medium ? !st->dec_med : !st->dec)) {
        aria_set_error("sa3_load: %s", aria_last_error()[0] ? aria_last_error() : "weight load failed");
        if (st->dit) aria_sa3_dit_free(st->dit);
        if (st->dec) aria_sa3_dec_free(st->dec);
        if (st->dec_med) aria_sa3_dec_medium_free(st->dec_med);
        free(st);
        return NULL;
    }

    const safetensor_t *tw = safetensors_find(sf, "conditioner.conditioners.seconds_total.embedder.embedding.1.weight");
    const safetensor_t *tb = safetensors_find(sf, "conditioner.conditioners.seconds_total.embedder.embedding.1.bias");
    if (!tw || !tb) { aria_set_error("sa3_load: missing seconds_total conditioner"); goto fail; }
    st->sec_w = safetensors_f32_ptr(sf, tw);
    st->sec_b = safetensors_f32_ptr(sf, tb);
    /* min/max from config (seconds_total conditioner) */
    double mn = 0.0, mx = 384.0;
    aria_json_get_number(config_json, "min_val", &mn);
    aria_json_get_number(config_json, "max_val", &mx);
    st->sec_min = (float)mn; st->sec_max = (float)mx;
    return st;

fail:
    aria_sa3_dit_free(st->dit); aria_sa3_dec_free(st->dec); aria_sa3_dec_medium_free(st->dec_med);
    free(st);
    return NULL;
}

static void sa3_unload(void *state) {
    sa3_state *st = state;
    if (!st) return;
#ifdef ARIA_CUDA
    if (st->cdit) aria_cuda_dit_free(st->cdit);
    if (st->cdec) aria_cuda_dec_free(st->cdec);
    if (st->cdec_med) aria_cuda_dec_medium_free(st->cdec_med);
#endif
    aria_sa3_dit_free(st->dit);
    aria_sa3_dec_free(st->dec);
    if (st->dec_med) aria_sa3_dec_medium_free(st->dec_med);
    if (st->enc_taae) aria_sa3_enc_free(st->enc_taae);
    if (st->enc) aria_t5enc_free(st->enc);
    if (st->tok) aria_tokenizer_free(st->tok);
    free(st->cached_prompt); free(st->cached_emb);
    free(st);   /* sec_w/sec_b are borrowed from the mmap */
    return;
}

/* denoiser closure for the pingpong sampler: one request context, reused across
 * all steps (cross_ed / RoPE / per-block cross K/V are cached in the req). The
 * device path runs the DiT on the GPU (gcond computed on the host per step). */
typedef struct {
    const aria_sa3_dit *dit; aria_sa3_dit_req *req;
    aria_cuda_dit *cdit; const float *global_seconds; float *gcond;
    int step;   /* E12: current denoise step (the pingpong calls us once per step, in order) */
} sa3_dctx;
static void sa3_denoise(void *c, const float *x, float t, float *v, int n) {
    (void)n;
    sa3_dctx *d = c;
#ifdef ARIA_CUDA
    if (d->cdit) {   /* steering is CPU-only in this slice (no device hook yet) */
        aria_sa3_dit_global_cond(d->dit, d->global_seconds, t, d->gcond);
        aria_cuda_dit_step(d->cdit, v, x, d->gcond);
        d->step++;
        return;
    }
#endif
    aria_sa3_dit_req_set_step(d->req, d->step);
    aria_sa3_dit_step(d->dit, d->req, v, x, t);
    d->step++;
}

/* Build the local-additive inpaint conditioning for a continue/inpaint request:
 * read + prepare the init clip (channel-major [2, T*4096], 44.1 kHz, pad/crop),
 * encode it to a latent, build the keep/regenerate mask, and assemble
 * local[T,257] = [mask | latent*mask]. Returns a malloc'd [T*257] buffer (caller
 * frees), or NULL on error. Matches inference/generation.py's inpaint path. */
static float *sa3_build_inpaint_local(aria_ctx *ctx, sa3_state *st,
                                      const aria_gen_params *p, int T) {
    int audio_len = T * 4096;
    /* in-memory context (streaming) takes precedence over the WAV path; not owned here */
    int owns_a = (p->init_audio_mem == NULL);
    aria_audio *a = owns_a ? aria_wav_read(p->init_audio) : (aria_audio *)p->init_audio_mem;
    if (!a) { aria_set_error("inpaint: cannot read %s", p->init_audio); return NULL; }
    if (a->sample_rate != st->sample_rate) {
        aria_set_error("inpaint: %s is %d Hz but the model is %d Hz "
                       "(resampling is not supported yet; resample to %d Hz first)",
                       p->init_audio ? p->init_audio : "(memory)", a->sample_rate, st->sample_rate, st->sample_rate);
        if (owns_a) aria_audio_free(a);
        return NULL;
    }
    int clip = (int)(a->num_frames < audio_len ? a->num_frames : audio_len);
    float *audio = malloc((size_t)2 * audio_len * sizeof(float));   /* channel-major */
    for (int c = 0; c < 2; c++) {
        int sc = (a->channels == 1) ? 0 : (c < a->channels ? c : a->channels - 1);
        for (int t = 0; t < audio_len; t++)
            audio[(size_t)c * audio_len + t] =
                (t < clip) ? a->data[(size_t)t * a->channels + sc] : 0.0f;
    }
    if (owns_a) aria_audio_free(a);

    if (!st->enc_taae) st->enc_taae = aria_sa3_enc_load(ctx->sf);
    if (!st->enc_taae) { aria_set_error("inpaint: taae encoder load failed"); free(audio); return NULL; }
    /* the SAME encoder rounds T_patch up to a multiple of 32 -> its latent length is
     * ceil(T/2)*2 (= T for even T, T+1 for odd T), which can exceed the DiT's T; size the
     * buffer to the encoder's actual output so the write fits, then use the first T. */
    int T_enc = aria_sa3_latent_len(aria_sa3_patch_len(audio_len));
    float *latent = malloc((size_t)256 * (T_enc > T ? T_enc : T) * sizeof(float));
    aria_sa3_enc_forward(st->enc_taae, latent, audio, audio_len);
    free(audio);

    /* keep/regenerate mask (1=keep, 0=regenerate). continue: regenerate the tail. */
    float from_s = p->inpaint_continue ? (float)clip / st->sample_rate : p->inpaint_from_s;
    float to_s   = p->inpaint_continue ? p->seconds_total
                 : (p->inpaint_to_s > 0 ? p->inpaint_to_s : p->seconds_total);
    int from = (int)(from_s * st->sample_rate), to = (int)(to_s * st->sample_rate);
    if (from < 0) from = 0;
    if (from > audio_len) from = audio_len;
    if (to < from) to = from;
    if (to > audio_len) to = audio_len;
    float *mask_audio = malloc((size_t)audio_len * sizeof(float));
    for (int i = 0; i < audio_len; i++) mask_audio[i] = (i >= from && i < to) ? 0.0f : 1.0f;

    float *mask_lat = malloc((size_t)T * sizeof(float));
    aria_inpaint_mask_latent(mask_lat, mask_audio, audio_len, T);
    float *local = malloc((size_t)T * 257 * sizeof(float));
    aria_inpaint_local_cond(local, latent, mask_lat, T);

    free(latent); free(mask_audio); free(mask_lat);
    return local;
}

static int sa3_generate(aria_ctx *ctx, void *state,
                        const aria_gen_params *p, aria_audio **out) {
    sa3_state *st = state;
    const int ED = 768;       /* cond_token_dim */
    const int N_PROMPT = 256; /* T5Gemma tokens */

    /* prompt embedding [256,768]: text -> tokenize+encode, or precomputed file, or zeros */
    float *prompt = calloc((size_t)N_PROMPT * ED, sizeof(float));
    aria_parity_tensor pe; int have_pe = 0;
    if (p->prompt && p->prompt[0]) {
        if (st->cached_emb && st->cached_prompt && strcmp(p->prompt, st->cached_prompt) == 0) {
            memcpy(prompt, st->cached_emb, (size_t)N_PROMPT * ED * sizeof(float));   /* reuse */
        } else {
            if (sa3_ensure_text(ctx, st) != 0) { free(prompt); return -1; }
            int ids[256];
            int n_real = aria_tokenizer_encode(st->tok, p->prompt, ids, N_PROMPT);
            if (n_real < 0) { aria_set_error("generate: tokenization failed"); free(prompt); return -1; }
            aria_t5enc_encode(st->enc, prompt, ids, N_PROMPT, n_real);
            free(st->cached_prompt); st->cached_prompt = strdup(p->prompt);
            if (!st->cached_emb) st->cached_emb = malloc((size_t)N_PROMPT * ED * sizeof(float));
            if (st->cached_emb) memcpy(st->cached_emb, prompt, (size_t)N_PROMPT * ED * sizeof(float));
        }
    } else if (p->prompt_embed_path) {
        if (aria_parity_load(p->prompt_embed_path, &pe) != 0 || pe.ndim != 2 ||
            pe.shape[0] != N_PROMPT || pe.shape[1] != ED) {
            aria_set_error("generate: prompt-embed must be a [256,768] .atns file");
            free(prompt); return -1;
        }
        memcpy(prompt, pe.data, (size_t)N_PROMPT * ED * sizeof(float));
        have_pe = 1;
    }
    /* else: unconditional (zeros) */

    /* seconds_total embedding [768] -> cross token + global cond */
    float sec_emb[768];
    aria_number_embed(sec_emb, p->seconds_total, st->sec_min, st->sec_max, st->sec_w, st->sec_b, ED);

    int n_cond = N_PROMPT + 1;
    float *cross = malloc((size_t)n_cond * ED * sizeof(float));
    memcpy(cross, prompt, (size_t)N_PROMPT * ED * sizeof(float));
    memcpy(cross + (size_t)N_PROMPT * ED, sec_emb, (size_t)ED * sizeof(float));
    free(prompt);
    if (have_pe) aria_parity_free(&pe);

    /* latent length from requested duration */
    int T = (int)lround((double)p->seconds_total * st->sample_rate / st->cfg.downsampling_ratio);
    if (T < 1) T = 1;
    int n = st->cfg.io_channels * T;  /* 256 * T */

    /* init noise */
    aria_rng rng;
    aria_rng_init(&rng, p->seed >= 0 ? (uint64_t)p->seed : 0x9E3779B97F4A7C15ULL,
                  p->rng_torch ? ARIA_RNG_TORCH : ARIA_RNG_XOSHIRO);
    float *x = malloc((size_t)n * sizeof(float));
    aria_rng_randn(&rng, x, n);

    /* schedule + pingpong */
    int steps = p->steps > 0 ? p->steps : 8;
    float *sched = malloc((size_t)(steps + 1) * sizeof(float));
    aria_logsnr_schedule(sched, steps, 1.0f, -6.2f, 2000.0f, 1.0f, 2.0f, (float)T);

    int profile = getenv("ARIA_PROFILE") != NULL;

    /* precision: q8/q4 quantize the DiT per-step GEMMs. Decide the backend FIRST so
     * the host-side per-request work matches it: on the GPU the weights are packed
     * in VRAM + dequantized there, so the host path stays fp32 (fast); the CPU
     * dequant overlay is built only when the DiT actually runs on the CPU. fp16/bf16
     * are not yet a distinct CPU format -> fp32 there. */
    aria_dtype prec = (p->precision == ARIA_Q8 || p->precision == ARIA_Q4) ? p->precision : ARIA_F32;
    int quant = (prec != ARIA_F32);

    /* E9.2: a pre-quantized DiT overlay (.aria from aria-quantize) -> CPU path,
     * skips on-the-fly quantization. Loaded once, then cached across generations. */
    int loaded_quant = (p->load_quant != NULL);
    if (loaded_quant && !st->quant_loaded) {
        if (aria_sa3_dit_quant_load(st->dit, p->load_quant) != 0) {
            aria_set_error("generate: cannot load packed quant '%s' (missing or shape mismatch)", p->load_quant);
            free(cross); free(x); free(sched); return -1;
        }
        st->quant_loaded = 1;
        fprintf(stderr, "[aria] DiT: loaded packed quant %s = %.0f MB (CPU)\n",
                p->load_quant, aria_sa3_dit_weight_bytes(st->dit) / 1048576.0);
    }
    if (loaded_quant) quant = 1;

    int on_gpu = 0;
#ifdef ARIA_CUDA
    /* auto: GPU when it's worth it (sm_70+) and fits; cuda: force any device
     * (CPU-fallback on OOM); cpu: never. Packed-quant stays on the CPU DiT. */
    /* medium runs its DiT on the GPU (the device DiT now has a differential path)
     * but keeps the medium decoder on the CPU (the device decoder is small-music).
     * continue/inpaint also runs on the GPU: the taae encoder builds the context latent
     * on the CPU, then the per-block local-additive cond is uploaded and added on device. */
    int has_steer = p->steer && p->steer->n > 0;   /* E12: residual hook is CPU-only this slice */
    int want_gpu = !loaded_quant && !has_steer &&
                   ((p->device == ARIA_DEVICE_CUDA && aria_cuda_available()) ||
                    (p->device == ARIA_DEVICE_AUTO && aria_cuda_recommended()));
    if (want_gpu && !st->cdit) {   /* upload + quantize weights once; reused across gens */
        aria_sa3_dit_view view; aria_sa3_dit_get_view(st->dit, &view);  /* f32 view (overlay-independent) */
        st->cdit = aria_cuda_dit_create(&view, prec);
        const char *how = p->device == ARIA_DEVICE_AUTO ? "auto" : "forced";
        const char *pn = quant ? aria_dtype_name(prec) : "fp16";
        if (st->cdit) fprintf(stderr, "[aria] DiT: GPU device-resident, %s (%s)\n", pn, how);
        else fprintf(stderr, "[aria] DiT: GPU insufficient VRAM, using CPU (%s)\n", how);
    }
    on_gpu = want_gpu && st->cdit;
#endif

    /* CPU dequant overlay only when running on the CPU; the GPU path keeps st->dit
     * f32 so the host cross-K/V projection (and any CPU fallback) stays fast. */
    if (!loaded_quant) {
        aria_sa3_dit_quantize(st->dit, (quant && !on_gpu) ? prec : ARIA_F32);
        if (quant && !on_gpu) {
            static int announced = 0;
            if (!announced) {
                fprintf(stderr, "[aria] DiT: %s block weights = %.0f MB (CPU)\n",
                        aria_dtype_name(prec), aria_sa3_dit_weight_bytes(st->dit) / 1048576.0);
                announced = 1;
            }
        }
    }

    double t0 = sa3_now();
    aria_sa3_dit_req *req = aria_sa3_dit_req_begin(st->dit, T, cross, n_cond, sec_emb, on_gpu);
    /* continue / inpaint: build + attach the local-additive conditioning (E7). */
    if (p->init_audio || p->init_audio_mem) {
        float *local_TC = sa3_build_inpaint_local(ctx, st, p, T);
        if (!local_TC) { aria_sa3_dit_req_end(req); free(cross); free(x); free(sched); return -1; }
        aria_sa3_dit_req_set_local(req, local_TC, T, 257);  /* projected once, then owned by req */
        free(local_TC);
    }
    aria_sa3_dit_req_set_steer(req, p->steer);   /* E12: residual-site steering (CPU path) */
    sa3_dctx dc = { st->dit, req, NULL, NULL, NULL, 0 };
#ifdef ARIA_CUDA
    if (on_gpu) {
        aria_sa3_dit_req_view rv; aria_sa3_dit_req_get_view(req, &rv);
        aria_cuda_dit_set_request(st->cdit, &rv);
        dc.cdit = st->cdit;
        dc.global_seconds = rv.global_seconds;
        dc.gcond = malloc((size_t)6 * st->cfg.embed_dim * sizeof(float));
        if (st->is_medium) {
            if (!st->cdec_med) {
                aria_sa3_dec_medium_view dv; aria_sa3_dec_medium_get_view(st->dec_med, &dv);
                st->cdec_med = aria_cuda_dec_medium_create(&dv);
                if (st->cdec_med) fprintf(stderr, "[aria] decoder: GPU device-resident (medium), fp16\n");
            }
        } else if (!st->cdec) {
            aria_sa3_dec_view dv; aria_sa3_dec_get_view(st->dec, &dv);
            st->cdec = aria_cuda_dec_create(&dv);
            if (st->cdec) fprintf(stderr, "[aria] decoder: GPU device-resident, fp16\n");
        }
    }
#endif
    double t1 = sa3_now();
    aria_pingpong_cb(x, n, sched, steps, sa3_denoise, &dc, &rng, NULL, p->progress, p->progress_user);
    double t2 = sa3_now();
#ifdef ARIA_CUDA
    free(dc.gcond);   /* the device handle persists on st; only gcond is per-call */
#endif
    aria_sa3_dit_req_end(req);

    /* decode -> interleaved stereo (GPU if the DiT ran there, else CPU) */
    float *audio = malloc((size_t)2 * T * 4096 * sizeof(float));
#ifdef ARIA_CUDA
    if (st->cdec_med)  aria_cuda_dec_medium_forward(st->cdec_med, audio, x, T);
    else if (st->cdec) aria_cuda_dec_forward(st->cdec, audio, x, T);
    else
#endif
    if (st->is_medium) aria_sa3_dec_medium_forward(st->dec_med, audio, x, T);
    else               aria_sa3_dec_forward(st->dec, audio, x, T);
#ifdef ARIA_CUDA
    /* once the DiT+decoder are device-resident, the host F32 weights aren't read on
     * the hot path again -- drop them to reclaim ~the model size of host RSS. The few
     * per-request host MLPs (to_cond/to_global/timestep) just re-fault cheaply. */
    if ((st->cdit || st->cdec || st->cdec_med) && !st->weights_advised) {
        safetensors_advise_dontneed(ctx->sf);
        st->weights_advised = 1;
    }
#endif
    double t3 = sa3_now();
    if (profile) {
        struct rusage ru; getrusage(RUSAGE_SELF, &ru);
        double cur_rss = 0;   /* current resident set (post host-weight drop on GPU) */
        FILE *sf = fopen("/proc/self/statm", "r");
        if (sf) { long sz = 0, res = 0; if (fscanf(sf, "%ld %ld", &sz, &res) == 2) cur_rss = res * 4096.0 / 1048576.0; fclose(sf); }
        fprintf(stderr, "[aria] profile: setup=%.2fs dit=%.2fs decode=%.2fs (T=%d steps=%d) | RSS cur %.0f / peak %.0f MB",
                t1 - t0, t2 - t1, t3 - t2, T, steps, cur_rss, ru.ru_maxrss / 1024.0);
#ifdef ARIA_CUDA
        if (st->cdit || st->cdec || st->cdec_med) {
            size_t used = 0, total = 0; aria_cuda_meminfo(&used, &total);
            fprintf(stderr, " | GPU %.0f/%.0f MB", used / 1048576.0, total / 1048576.0);
        }
#endif
        fprintf(stderr, "\n");
    }
    aria_audio *a = aria_audio_alloc(st->sample_rate, 2, (int64_t)T * 4096);
    for (int64_t i = 0; i < (int64_t)T * 4096; i++) {
        a->data[i * 2 + 0] = audio[i];
        a->data[i * 2 + 1] = audio[(int64_t)T * 4096 + i];
    }
    *out = a;

    free(cross); free(x); free(sched); free(audio);
    return 0;
}

const aria_model_module aria_module_sa3 = {
    .name = "stable-audio-3",
    .detect = sa3_detect,
    .load = sa3_load,
    .unload = sa3_unload,
    .generate = sa3_generate,
};
