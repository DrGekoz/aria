/* test_number_cond.c - parity: aria NumberConditioner vs PyTorch dumps.
 *
 * Requires ARIA_MODEL (model dir) and ARIA_DUMPS (parity dump dir). If either
 * is unset the test SKIPs (so `make test` stays hermetic). Driven by
 * `make parity`, which runs scripts/dump_phase1.py first.
 */
#include "../src/aria_safetensors.h"
#include "../src/aria_cond.h"
#include "../src/aria_parity.h"
#include "../src/aria_json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* keep in sync with scripts/dump_phase1.py SECONDS_VALUES */
static const float SECONDS_VALUES[] = {0.0f, 5.0f, 15.0f, 30.0f, 47.3f, 95.0f, 200.0f, 384.0f};
static const int N_VALUES = (int)(sizeof(SECONDS_VALUES) / sizeof(SECONDS_VALUES[0]));

int main(void) {
    const char *model = getenv("ARIA_MODEL");
    const char *dumps = getenv("ARIA_DUMPS");
    if (!model || !dumps) {
        printf("test_number_cond: SKIP (set ARIA_MODEL and ARIA_DUMPS)\n");
        return 0;
    }

    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors", model);
    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) { printf("FAIL: cannot open %s\n", path); return 1; }

    const safetensor_t *tw = safetensors_find(sf, "conditioner.conditioners.seconds_total.embedder.embedding.1.weight");
    const safetensor_t *tb = safetensors_find(sf, "conditioner.conditioners.seconds_total.embedder.embedding.1.bias");
    if (!tw || !tb) { printf("FAIL: missing seconds_total weights\n"); return 1; }
    int features = (int)tw->shape[0];   /* 768 */
    float *W = safetensors_get_f32(sf, tw);
    float *b = safetensors_get_f32(sf, tb);

    /* min/max from model_config.json (fallback 0/384) */
    double mn = 0.0, mx = 384.0;
    snprintf(path, sizeof(path), "%s/model_config.json", model);
    char *cfg = aria_read_file(path, NULL);
    if (cfg) {
        aria_json_get_number(cfg, "min_val", &mn);
        aria_json_get_number(cfg, "max_val", &mx);
        free(cfg);
    }

    float *out = malloc((size_t)features * sizeof(float));
    int fails = 0;
    float worst = 0.0f;
    for (int i = 0; i < N_VALUES; i++) {
        aria_number_embed(out, SECONDS_VALUES[i], (float)mn, (float)mx, W, b, features);
        snprintf(path, sizeof(path), "%s/number_cond/seconds_%d.atns", dumps, i);
        aria_parity_tensor ref;
        if (aria_parity_load(path, &ref) != 0) { printf("FAIL: cannot load %s\n", path); fails++; continue; }
        if (ref.numel != features) { printf("FAIL: shape mismatch %s (%lld vs %d)\n", path, (long long)ref.numel, features); fails++; aria_parity_free(&ref); continue; }
        float md = aria_parity_maxabsdiff(out, ref.data, features);
        /* combined atol+rtol gate: f32 linear accumulation order differs from
         * numpy/BLAS, so tolerate ~rtol * max|ref| on top of a small atol. */
        float maxref = 0.0f;
        for (int j = 0; j < features; j++) { float a = ref.data[j] < 0 ? -ref.data[j] : ref.data[j]; if (a > maxref) maxref = a; }
        float thresh = 3e-4f + 2e-3f * maxref;
        if (md > worst) worst = md;
        if (md > thresh) { printf("FAIL: seconds=%.1f maxdiff=%.6e (thresh %.3e, max|ref|=%.3f)\n", SECONDS_VALUES[i], md, thresh, maxref); fails++; }
        aria_parity_free(&ref);
    }

    free(out); free(W); free(b);
    safetensors_close(sf);
    if (fails) { printf("test_number_cond: %d failures (worst=%.3e)\n", fails, worst); return 1; }
    printf("test_number_cond: OK (worst maxdiff=%.3e over %d values)\n", worst, N_VALUES);
    return 0;
}
