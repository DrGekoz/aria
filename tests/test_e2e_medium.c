/* test_e2e_medium.c - medium end-to-end parity (E10.2 -> M6): init noise ->
 * pingpong(medium differential DiT) -> medium decode -> audio, with injected
 * per-step noise, vs PyTorch. Needs ARIA_MODEL (medium) + ARIA_DUMPS (medium e2e). */
#include "../src/aria_safetensors.h"
#include "../src/aria_sa3.h"
#include "../src/aria_sa3_dit.h"
#include "../src/aria_sa3_dec.h"
#include "../src/aria_sampler.h"
#include "../src/aria_json.h"
#include "../src/aria_parity.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const char *DUMPS;
static int fails = 0;

static int load(const char *name, aria_parity_tensor *t) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/e2e/%s", DUMPS, name);
    if (aria_parity_load(path, t) != 0) { printf("FAIL: load %s\n", path); fails++; return -1; }
    return 0;
}
static void compare(const char *what, const float *got, const float *ref, int64_t n, float atol, float rtol) {
    float md = aria_parity_maxabsdiff(got, ref, n), maxref = 0.0f;
    for (int64_t i = 0; i < n; i++) { float a = fabsf(ref[i]); if (a > maxref) maxref = a; }
    float thr = atol + rtol * maxref;
    if (md > thr) { printf("FAIL %s: maxdiff=%.3e thr=%.3e (max|ref|=%.3f)\n", what, md, thr, maxref); fails++; }
    else printf("ok   %s: maxdiff=%.3e (max|ref|=%.3f)\n", what, md, maxref);
}

typedef struct { const aria_sa3_dit *dit; const float *cross; const float *global; int T; } dctx;
static void dit_denoise(void *c, const float *x, float t, float *v, int n) {
    (void)n; const dctx *d = c;
    aria_sa3_dit_forward(d->dit, v, x, d->T, t, d->cross, 257, d->global);
}

int main(void) {
    const char *model = getenv("ARIA_MODEL");
    DUMPS = getenv("ARIA_DUMPS");
    if (!model || !DUMPS) { printf("test_e2e_medium: SKIP (set ARIA_MODEL + ARIA_DUMPS)\n"); return 0; }

    char path[1024];
    snprintf(path, sizeof(path), "%s/model_config.json", model);
    char *cfgjson = aria_read_file(path, NULL);
    aria_sa3_config cfg;
    if (!cfgjson || aria_sa3_parse_config(cfgjson, &cfg) != 0) { printf("FAIL: config\n"); return 1; }
    free(cfgjson);
    snprintf(path, sizeof(path), "%s/model.safetensors", model);
    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) { printf("FAIL: open safetensors\n"); return 1; }
    aria_sa3_dit *dit = aria_sa3_dit_load(sf, &cfg);
    aria_sa3_dec_medium *dec = aria_sa3_dec_medium_load(sf);
    if (!dit || !dec) { printf("FAIL: medium load\n"); return 1; }

    aria_parity_tensor cross, glob, init_noise, step_noise, sched, latref, audref;
    if (load("cross.atns", &cross) || load("global.atns", &glob) ||
        load("init_noise.atns", &init_noise) || load("step_noise.atns", &step_noise) ||
        load("sched.atns", &sched) || load("latent.atns", &latref) || load("audio.atns", &audref)) return 1;
    int T = (int)init_noise.shape[1], n = 256 * T, steps = (int)sched.numel - 1;

    dctx ctx = { dit, cross.data, glob.data, T };
    float *x = malloc((size_t)n * sizeof(float));
    memcpy(x, init_noise.data, (size_t)n * sizeof(float));
    aria_pingpong(x, n, sched.data, steps, dit_denoise, &ctx, NULL, step_noise.data);
    compare("latent", x, latref.data, latref.numel, 2e-3f, 1e-2f);

    float *audio = malloc((size_t)2 * T * 4096 * sizeof(float));
    aria_sa3_dec_medium_forward(dec, audio, x, T);
    compare("audio", audio, audref.data, audref.numel, 2e-3f, 1e-2f);

    free(x); free(audio);
    aria_parity_free(&cross); aria_parity_free(&glob); aria_parity_free(&init_noise);
    aria_parity_free(&step_noise); aria_parity_free(&sched); aria_parity_free(&latref); aria_parity_free(&audref);
    aria_sa3_dit_free(dit); aria_sa3_dec_medium_free(dec); safetensors_close(sf);
    if (fails) { printf("test_e2e_medium: FAILED\n"); return 1; }
    printf("test_e2e_medium: OK\n");
    return 0;
}
