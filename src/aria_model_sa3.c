/*
 * aria_model_sa3.c - Stable Audio 3 model module.
 *
 * Phase 0: detection + weight-layout validation. The DiT forward, taae_v2
 * autoencoder, T5Gemma conditioning and pingpong sampler are wired in during
 * Phase 1 (this file gains aria_dit_sa3.c / aria_taae.c / aria_t5gemma.c
 * collaborators).
 */

#include "aria_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int n_conditioner;
    int n_dit;          /* "model." prefix */
    int n_pretransform; /* taae_v2 autoencoder */
} sa3_state;

static int sa3_detect(const char *model_type) {
    return strcmp(model_type, "diffusion_cond_inpaint") == 0 ||
           strcmp(model_type, "diffusion_cond") == 0;
}

static void *sa3_load(aria_ctx *ctx, safetensors_file_t *sf, const char *config_json) {
    (void)ctx; (void)config_json;
    sa3_state *st = calloc(1, sizeof(sa3_state));
    if (!st) { aria_set_error("sa3_load: oom"); return NULL; }

    st->n_conditioner   = safetensors_count_prefix(sf, "conditioner.");
    st->n_dit           = safetensors_count_prefix(sf, "model.");
    st->n_pretransform  = safetensors_count_prefix(sf, "pretransform.");

    if (st->n_dit == 0 || st->n_pretransform == 0) {
        aria_set_error("sa3_load: unexpected weight layout "
                       "(conditioner=%d model=%d pretransform=%d)",
                       st->n_conditioner, st->n_dit, st->n_pretransform);
        free(st);
        return NULL;
    }
    return st;
}

static void sa3_unload(void *state) { free(state); }

static int sa3_generate(aria_ctx *ctx, void *state,
                        const aria_gen_params *p, aria_audio **out) {
    (void)ctx; (void)state; (void)p; (void)out;
    aria_set_error("sa3_generate: not implemented yet (Phase 1)");
    return -1;
}

const aria_model_module aria_module_sa3 = {
    .name = "stable-audio-3",
    .detect = sa3_detect,
    .load = sa3_load,
    .unload = sa3_unload,
    .generate = sa3_generate,
};
