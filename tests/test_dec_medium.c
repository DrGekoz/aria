/* test_dec_medium.c - medium taae_v2 decoder parity (E10.2): latent -> audio.
 * Needs ARIA_MODEL (medium dir) + ARIA_DUMPS (medium dec dump from dump_decoder). */
#include "../src/aria_safetensors.h"
#include "../src/aria_sa3_dec.h"
#include "../src/aria_parity.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static const char *DUMPS;
static int fails = 0;

/* deterministic pseudo-random latent fill (LCG) for the hermetic windowed gate. */
static void fill_latent(float *lat, int64_t n, uint64_t seed) {
    uint64_t s = seed;
    for (int64_t i = 0; i < n; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        lat[i] = (float)(s >> 33) / 2147483648.0f - 1.0f;   /* ~U(-1,1) */
    }
}

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
    if (!model) { printf("test_dec_medium: SKIP (set ARIA_MODEL)\n"); return 0; }

    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors", model);
    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) { printf("FAIL: open safetensors\n"); return 1; }
    aria_sa3_dec_medium *m = aria_sa3_dec_medium_load(sf);
    if (!m) { printf("FAIL: medium decoder load\n"); safetensors_close(sf); return 1; }

    /* E16.1 gate: windowed decode vs monolithic (byte-identical; halo H=12 frames).
     * Model-backed but hermetic (fixed synthetic latent). Kept short but > 2 windows. */
    {
        int T = 72, W = 24;   /* W=24 => 3 windows [0,24)[24,48)[48,72), 2 interior seams */
        int64_t nl = (int64_t)256 * T, na = (int64_t)2 * T * 4096;
        float *lat = malloc((size_t)nl * sizeof(float));
        fill_latent(lat, nl, 0xC0FFEE11ULL);
        float *mono = malloc((size_t)na * sizeof(float));
        float *win  = malloc((size_t)na * sizeof(float));
        aria_sa3_dec_medium_forward(m, mono, lat, T);
        aria_sa3_dec_medium_forward_windowed(m, win, lat, T, W);
        float md = aria_parity_maxabsdiff(mono, win, na);
        if (md > 1e-6f) { printf("FAIL windowed(medium): T=%d W=%d maxdiff=%.3e\n", T, W, md); fails++; }
        else printf("ok   windowed(medium): T=%d W=%d maxdiff=%.3e%s\n", T, W, md,
                    md == 0.0f ? " (byte-identical)" : "");
        free(lat); free(mono); free(win);
    }

    if (!DUMPS) {
        printf("test_dec_medium: windowed gate only (set ARIA_DUMPS for PyTorch parity)\n");
        aria_sa3_dec_medium_free(m); safetensors_close(sf);
        if (fails) { printf("test_dec_medium: FAILED\n"); return 1; }
        printf("test_dec_medium: OK\n"); return 0;
    }

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
