/*
 * aria_model.h - Internal model-module interface (the modularity seam).
 *
 * Each audio-diffusion architecture is one module describing how to detect,
 * load and run it on top of the shared op library. Adding AceStep 1.5 (or any
 * future model) means writing one module file and appending it to the registry
 * in aria.c -- no changes to the kernels, sampler, quantization or I/O.
 */

#ifndef ARIA_MODEL_H
#define ARIA_MODEL_H

#include "aria.h"
#include "aria_safetensors.h"

struct aria_ctx {
    char *model_dir;
    char *config_json;              /* contents of model_config.json */
    safetensors_file_t *sf;         /* mmap'd model.safetensors */
    const struct aria_model_module *module;
    void *state;                    /* module-specific model state */
    char model_type[64];
    int sample_rate;
    int channels;
};

typedef struct aria_model_module {
    const char *name;
    /* return 1 if this module handles the given model_type string */
    int  (*detect)(const char *model_type);
    /* build model state from weights + config json; opaque ptr or NULL on error */
    void *(*load)(aria_ctx *ctx, safetensors_file_t *sf, const char *config_json);
    void  (*unload)(void *state);
    /* text->audio generation; returns 0 on success */
    int   (*generate)(aria_ctx *ctx, void *state, const aria_gen_params *p, aria_audio **out);
} aria_model_module;

/* Pick the module whose detect() claims model_type, or NULL. */
const aria_model_module *aria_registry_detect(const char *model_type);

/* Set the process-wide last-error string (printf-style). */
void aria_set_error(const char *fmt, ...);

/* Module descriptors (defined in their own files). */
extern const aria_model_module aria_module_sa3;

#endif /* ARIA_MODEL_H */
