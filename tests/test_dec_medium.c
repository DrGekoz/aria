/* test_dec_medium.c - medium taae_v2 decoder parity (E10.2): latent -> audio.
 * Needs ARIA_MODEL (medium dir) + ARIA_DUMPS (medium dec dump from dump_decoder). */
#include "../src/aria_safetensors.h"
#include "../src/aria_sa3_dec.h"
#include "../src/aria_parity.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const char *DUMPS;
static int fails = 0;

static int load(const char *name, aria_parity_tensor *t) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/dec/%s", DUMPS, name);
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

int main(void) {
    const char *model = getenv("ARIA_MODEL");
    DUMPS = getenv("ARIA_DUMPS");
    if (!model || !DUMPS) { printf("test_dec_medium: SKIP (set ARIA_MODEL + ARIA_DUMPS)\n"); return 0; }

    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors", model);
    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) { printf("FAIL: open safetensors\n"); return 1; }
    aria_sa3_dec_medium *m = aria_sa3_dec_medium_load(sf);
    if (!m) { printf("FAIL: medium decoder load\n"); safetensors_close(sf); return 1; }

    aria_parity_tensor latent, audref;
    if (load("latent.atns", &latent) || load("audio.atns", &audref)) return 1;
    int T = (int)latent.shape[1];

    float *audio = malloc((size_t)2 * T * 4096 * sizeof(float));
    aria_sa3_dec_medium_forward(m, audio, latent.data, T);
    compare("audio", audio, audref.data, audref.numel, 1e-3f, 5e-3f);
    free(audio);

    aria_parity_free(&latent); aria_parity_free(&audref);
    aria_sa3_dec_medium_free(m); safetensors_close(sf);
    if (fails) { printf("test_dec_medium: FAILED\n"); return 1; }
    printf("test_dec_medium: OK\n");
    return 0;
}
