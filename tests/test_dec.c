/* test_dec.c - staged parity for the taae_v2 decoder vs PyTorch.
 * Stages: softnorm-inverse, one taae block, full SAME decoder, full audio.
 * Needs ARIA_MODEL + ARIA_DUMPS. */
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
/* optional load: missing dump -> skip (no failure) */
static int load_opt(const char *name, aria_parity_tensor *t) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/dec/%s", DUMPS, name);
    return aria_parity_load(path, t);
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
    if (!model || !DUMPS) { printf("test_dec: SKIP (set ARIA_MODEL and ARIA_DUMPS)\n"); return 0; }

    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors", model);
    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) { printf("FAIL: open safetensors\n"); return 1; }
    aria_sa3_dec *m = aria_sa3_dec_load(sf);
    if (!m) { printf("FAIL: decoder load\n"); safetensors_close(sf); return 1; }

    aria_parity_tensor latent, soft, decref, audref;
    if (load("latent.atns", &latent) || load("after_softnorm.atns", &soft) ||
        load("after_decoder.atns", &decref) || load("audio.atns", &audref)) return 1;

    int T = (int)latent.shape[1];

    /* stage 1: softnorm inverse */
    {
        float *z = malloc((size_t)256 * T * sizeof(float));
        aria_sa3_softnorm_decode(m, z, latent.data, T);
        compare("softnorm", z, soft.data, soft.numel, 1e-6f, 1e-6f);
        free(z);
    }
    /* stage 2: one taae block (optional -- only if the standalone dump exists) */
    {
        aria_parity_tensor blkx, blkout;
        if (load_opt("block_x.atns", &blkx) == 0 && load_opt("block_out.atns", &blkout) == 0) {
            int N = (int)blkx.shape[0];
            float *xc = malloc((size_t)blkx.numel * sizeof(float));
            memcpy(xc, blkx.data, (size_t)blkx.numel * sizeof(float));
            aria_sa3_dec_block_test(m, 0, xc, N);
            compare("taae_block", xc, blkout.data, blkout.numel, 5e-4f, 3e-3f);
            free(xc);
            aria_parity_free(&blkx); aria_parity_free(&blkout);
        } else {
            printf("--   taae_block: skipped (no standalone dump)\n");
        }
    }
    /* stage 3: full SAME decoder (latent -> [512, 16T]) */
    {
        int Lt = T * 16;
        float *z = malloc((size_t)256 * T * sizeof(float));
        aria_sa3_softnorm_decode(m, z, latent.data, T);
        float *dec = malloc((size_t)512 * Lt * sizeof(float));
        aria_sa3_same_decode(m, dec, z, T);
        compare("decoder", dec, decref.data, decref.numel, 1e-3f, 5e-3f);
        free(z); free(dec);
    }
    /* stage 4: full audio (latent -> [2, 4096T]) */
    {
        float *audio = malloc((size_t)2 * T * 4096 * sizeof(float));
        aria_sa3_dec_forward(m, audio, latent.data, T);
        compare("audio", audio, audref.data, audref.numel, 1e-3f, 5e-3f);
        free(audio);
    }

    aria_parity_free(&latent); aria_parity_free(&soft);
    aria_parity_free(&decref); aria_parity_free(&audref);
    aria_sa3_dec_free(m);
    safetensors_close(sf);

    if (fails) { printf("test_dec: FAILED\n"); return 1; }
    printf("test_dec: OK\n");
    return 0;
}
