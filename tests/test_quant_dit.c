/* test_quant_dit.c - quantization quality gate at the model level (E9.0/E9.1/E9.3).
 * Measures how much Q8/Q4 perturb the DiT's *function* on a fixed input: one
 * velocity forward at fp32 vs the same forward with the Q8/Q4 weight overlay.
 * (End-to-end waveform diff is meaningless here -- the pingpong sampler is chaotic,
 * so a 1% weight perturbation yields a different valid sample; the per-forward
 * velocity error is the real quantization signal.) Needs ARIA_MODEL only. */
#include "../src/aria_safetensors.h"
#include "../src/aria_sa3.h"
#include "../src/aria_sa3_dit.h"
#include "../src/aria_json.h"
#include "../src/aria_quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static unsigned rng = 99u;
static float frandn(void) {
    rng = rng * 1664525u + 1013904223u; float u1 = (float)(rng >> 8) / (float)(1u << 24) + 1e-7f;
    rng = rng * 1664525u + 1013904223u; float u2 = (float)(rng >> 8) / (float)(1u << 24);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}
static float relrms(const float *a, const float *ref, int n) {
    double d = 0, r = 0;
    for (int i = 0; i < n; i++) { double e = a[i] - ref[i]; d += e * e; r += (double)ref[i] * ref[i]; }
    return (float)sqrt(d / (r + 1e-12));
}

int main(void) {
    const char *model = getenv("ARIA_MODEL");
    if (!model) { printf("test_quant_dit: SKIP (set ARIA_MODEL)\n"); return 0; }

    char path[1024];
    snprintf(path, sizeof(path), "%s/model_config.json", model);
    char *cfgjson = aria_read_file(path, NULL);
    aria_sa3_config cfg;
    if (!cfgjson || aria_sa3_parse_config(cfgjson, &cfg) != 0) { printf("FAIL: config\n"); return 1; }
    free(cfgjson);
    snprintf(path, sizeof(path), "%s/model.safetensors", model);
    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) { printf("FAIL: open safetensors\n"); return 1; }
    aria_sa3_dit *m = aria_sa3_dit_load(sf, &cfg);
    if (!m) { printf("FAIL: dit load\n"); return 1; }

    int T = 16, ed = 768, n_cond = 257, C = 256;
    float *x = malloc((size_t)C * T * sizeof(float));
    float *cross = malloc((size_t)n_cond * ed * sizeof(float));
    float *glob = malloc((size_t)ed * sizeof(float));
    for (int i = 0; i < C * T; i++) x[i] = frandn();
    for (int i = 0; i < n_cond * ed; i++) cross[i] = frandn();
    for (int i = 0; i < ed; i++) glob[i] = frandn();
    float tt = 0.3f;

    float *vref = malloc((size_t)C * T * sizeof(float));
    float *vq = malloc((size_t)C * T * sizeof(float));

    /* fp32 reference */
    aria_sa3_dit_forward(m, vref, x, T, tt, cross, n_cond, glob);
    size_t f32_bytes = aria_sa3_dit_weight_bytes(m);

    int fails = 0;
    /* q8 (per-row int8): the high-fidelity option. q4 = asymmetric (zero-point)
     * int4 on the FFN + Q8 on the attention projections (error-sensitive) -- this
     * mixed/affine recipe brings q4 from ~30% (uniform symmetric) to ~9% velocity,
     * near q8, at a smaller footprint than q8. Both are hard gates now. */
    struct { aria_dtype dt; const char *name; float thr; } cases[] = {
        { ARIA_Q8, "q8", 0.05f }, { ARIA_Q4, "q4", 0.13f },
    };
    for (int c = 0; c < 2; c++) {
        aria_sa3_dit_quantize(m, cases[c].dt);
        aria_sa3_dit_forward(m, vq, x, T, tt, cross, n_cond, glob);
        float e = relrms(vq, vref, C * T);
        int finite = 1;
        for (int i = 0; i < C * T; i++) if (!isfinite(vq[i])) { finite = 0; break; }
        size_t qb = aria_sa3_dit_weight_bytes(m);
        int ok = finite && e <= cases[c].thr;
        printf("%s %s velocity rel-RMS=%.2f%% (bound %.0f%%)  weights %.0f MB (%.2fx vs fp32)\n",
               ok ? "ok  " : "FAIL", cases[c].name, e * 100, cases[c].thr * 100,
               qb / 1048576.0, (double)f32_bytes / qb);
        if (!ok) fails++;
    }

    /* fp32 round-trip: dropping the overlay must be bit-identical to the start */
    aria_sa3_dit_quantize(m, ARIA_F32);
    aria_sa3_dit_forward(m, vq, x, T, tt, cross, n_cond, glob);
    float e0 = relrms(vq, vref, C * T);
    if (e0 != 0.0f) { printf("FAIL fp32 round-trip: rel-RMS=%.3e (expected 0)\n", e0); fails++; }
    else printf("ok   fp32 round-trip bit-identical\n");

    free(x); free(cross); free(glob); free(vref); free(vq);
    aria_sa3_dit_free(m); safetensors_close(sf);
    if (fails) { printf("test_quant_dit: FAILED\n"); return 1; }
    printf("test_quant_dit: OK\n");
    return 0;
}
