/* test_enc.c - staged parity for the taae_v2 encoder vs PyTorch.
 * Stages: patchify, full SAME encoder, softnorm, full audio->latent (clean +
 * zero-padded length). Needs ARIA_MODEL + ARIA_DUMPS. */
#include "../src/aria_safetensors.h"
#include "../src/aria_sa3_enc.h"
#include "../src/aria_parity.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const char *DUMPS;
static int fails = 0;

static int load(const char *name, aria_parity_tensor *t) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/enc/%s", DUMPS, name);
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
    if (!model || !DUMPS) { printf("test_enc: SKIP (set ARIA_MODEL and ARIA_DUMPS)\n"); return 0; }

    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors", model);
    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) { printf("FAIL: open safetensors\n"); return 1; }
    aria_sa3_enc *m = aria_sa3_enc_load(sf);
    if (!m) { printf("FAIL: encoder load\n"); safetensors_close(sf); return 1; }

    aria_parity_tensor audio, patches, encref, latref, audpad, latpad;
    if (load("audio.atns", &audio) || load("patches.atns", &patches) ||
        load("enc.atns", &encref) || load("latent.atns", &latref) ||
        load("audio_pad.atns", &audpad) || load("latent_pad.atns", &latpad)) return 1;

    int L = (int)audio.shape[1];               /* audio[2,L] */
    int T_patch = (int)patches.shape[1];       /* patches[512,T_patch] */
    int T_lat = (int)latref.shape[1];          /* latent[256,T_lat] */

    /* stage 1: patchify */
    {
        float *p = malloc((size_t)512 * T_patch * sizeof(float));
        aria_sa3_patchify(p, audio.data, L, T_patch);
        compare("patchify", p, patches.data, patches.numel, 1e-6f, 1e-6f);
        free(p);
    }
    /* stage 2: full SAME encoder (patches -> enc[256,T_lat], pre-bottleneck) */
    {
        float *enc = malloc((size_t)256 * T_lat * sizeof(float));
        aria_sa3_same_encode(m, enc, patches.data, T_patch, T_lat);
        compare("encoder", enc, encref.data, encref.numel, 1e-3f, 5e-3f);
        free(enc);
    }
    /* stage 3: softnorm bottleneck (enc -> latent) */
    {
        float *lat = malloc((size_t)256 * T_lat * sizeof(float));
        aria_sa3_softnorm_encode(m, lat, encref.data, T_lat);
        compare("softnorm", lat, latref.data, latref.numel, 1e-5f, 1e-5f);
        free(lat);
    }
    /* stage 4: full encode audio -> latent (clean length) */
    {
        float *lat = malloc((size_t)256 * T_lat * sizeof(float));
        int t = aria_sa3_enc_forward(m, lat, audio.data, L);
        if (t != T_lat) { printf("FAIL full: T_lat=%d expected %d\n", t, T_lat); fails++; }
        compare("full", lat, latref.data, latref.numel, 1e-3f, 5e-3f);
        free(lat);
    }
    /* stage 5: full encode on the zero-padded length (T_patch not a multiple of 32) */
    {
        int Lp = (int)audpad.shape[1];
        int Tp = (int)latpad.shape[1];
        float *lat = malloc((size_t)256 * Tp * sizeof(float));
        int t = aria_sa3_enc_forward(m, lat, audpad.data, Lp);
        if (t != Tp) { printf("FAIL full_pad: T_lat=%d expected %d\n", t, Tp); fails++; }
        compare("full_pad", lat, latpad.data, latpad.numel, 1e-3f, 5e-3f);
        free(lat);
    }

    aria_parity_free(&audio); aria_parity_free(&patches);
    aria_parity_free(&encref); aria_parity_free(&latref);
    aria_parity_free(&audpad); aria_parity_free(&latpad);
    aria_sa3_enc_free(m);
    safetensors_close(sf);

    if (fails) { printf("test_enc: FAILED\n"); return 1; }
    printf("test_enc: OK\n");
    return 0;
}
