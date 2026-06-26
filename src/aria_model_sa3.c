/*
 * aria_model_sa3.c - Stable Audio 3 model module: load + end-to-end generation
 * (text conditioning via precomputed prompt embedding until T5Gemma is ported).
 */

#include "aria_model.h"
#include "aria_sa3.h"
#include "aria_sa3_dit.h"
#include "aria_sa3_dec.h"
#include "aria_sampler.h"
#include "aria_cond.h"
#include "aria_parity.h"
#include "aria_json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    aria_sa3_config cfg;
    aria_sa3_dit *dit;
    aria_sa3_dec *dec;
    /* seconds_total NumberConditioner (borrowed from the mmap) */
    const float *sec_w;  /* [768,256] */
    const float *sec_b;  /* [768] */
    float sec_min, sec_max;
    int sample_rate;
} sa3_state;

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

    st->dit = aria_sa3_dit_load(sf, &st->cfg);
    st->dec = aria_sa3_dec_load(sf);
    if (!st->dit || !st->dec) {
        aria_set_error("sa3_load: %s", aria_last_error()[0] ? aria_last_error() : "weight load failed");
        if (st->dit) aria_sa3_dit_free(st->dit);
        if (st->dec) aria_sa3_dec_free(st->dec);
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
    aria_sa3_dit_free(st->dit); aria_sa3_dec_free(st->dec);
    free(st);
    return NULL;
}

static void sa3_unload(void *state) {
    sa3_state *st = state;
    if (!st) return;
    aria_sa3_dit_free(st->dit);
    aria_sa3_dec_free(st->dec);
    free(st);   /* sec_w/sec_b are borrowed from the mmap */
    return;
}

/* denoiser closure for the pingpong sampler */
typedef struct { const aria_sa3_dit *dit; const float *cross; const float *global; int T; int n_cond; } sa3_dctx;
static void sa3_denoise(void *c, const float *x, float t, float *v, int n) {
    (void)n;
    const sa3_dctx *d = c;
    aria_sa3_dit_forward(d->dit, v, x, d->T, t, d->cross, d->n_cond, d->global);
}

static int sa3_generate(aria_ctx *ctx, void *state,
                        const aria_gen_params *p, aria_audio **out) {
    (void)ctx;
    sa3_state *st = state;
    const int ED = 768;       /* cond_token_dim */
    const int N_PROMPT = 256; /* T5Gemma tokens */

    /* prompt embedding [256,768]: precomputed file, or zeros (unconditional) */
    float *prompt = calloc((size_t)N_PROMPT * ED, sizeof(float));
    aria_parity_tensor pe; int have_pe = 0;
    if (p->prompt_embed_path) {
        if (aria_parity_load(p->prompt_embed_path, &pe) != 0 || pe.ndim != 2 ||
            pe.shape[0] != N_PROMPT || pe.shape[1] != ED) {
            aria_set_error("generate: prompt-embed must be a [256,768] .atns file");
            free(prompt); return -1;
        }
        memcpy(prompt, pe.data, (size_t)N_PROMPT * ED * sizeof(float));
        have_pe = 1;
    } else if (p->prompt && p->prompt[0]) {
        aria_set_error("generate: text prompts need T5Gemma (not yet ported); pass a precomputed --prompt-embed");
        free(prompt); return -1;
    }

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
    aria_rng_seed(&rng, p->seed >= 0 ? (uint64_t)p->seed : 0x9E3779B97F4A7C15ULL);
    float *x = malloc((size_t)n * sizeof(float));
    aria_rng_randn(&rng, x, n);

    /* schedule + pingpong */
    int steps = p->steps > 0 ? p->steps : 8;
    float *sched = malloc((size_t)(steps + 1) * sizeof(float));
    aria_logsnr_schedule(sched, steps, 1.0f, -6.2f, 2000.0f, 1.0f, 2.0f, (float)T);

    sa3_dctx dc = { st->dit, cross, sec_emb, T, n_cond };
    aria_pingpong(x, n, sched, steps, sa3_denoise, &dc, &rng, NULL);

    /* decode -> interleaved stereo */
    float *audio = malloc((size_t)2 * T * 4096 * sizeof(float));
    aria_sa3_dec_forward(st->dec, audio, x, T);
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
