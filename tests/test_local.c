/* test_local.c - E1.6 smoke: local-additive (inpaint) cond changes the DiT output
 * deterministically and stays finite; NULL local is a no-op. Needs ARIA_MODEL. */
#include "../src/aria_safetensors.h"
#include "../src/aria_sa3.h"
#include "../src/aria_sa3_dit.h"
#include "../src/aria_json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void step(const aria_sa3_dit *dit, int T, const float *cross, const float *glob,
                 const float *x, float *v, const float *local_raw, int local_dim) {
    aria_sa3_dit_req *r = aria_sa3_dit_req_begin(dit, T, cross, 257, glob, 0);
    if (local_raw) aria_sa3_dit_req_set_local(r, local_raw, T, local_dim);
    aria_sa3_dit_step(dit, r, v, x, 0.5f);
    aria_sa3_dit_req_end(r);
}

int main(void) {
    const char *model = getenv("ARIA_MODEL");
    if (!model) { printf("test_local: SKIP (set ARIA_MODEL)\n"); return 0; }
    char path[1024];
    snprintf(path, sizeof(path), "%s/model_config.json", model);
    char *cfgjson = aria_read_file(path, NULL);
    aria_sa3_config cfg;
    if (!cfgjson || aria_sa3_parse_config(cfgjson, &cfg) != 0) { printf("FAIL: config\n"); return 1; }
    free(cfgjson);
    snprintf(path, sizeof(path), "%s/model.safetensors", model);
    safetensors_file_t *sf = safetensors_open(path);
    aria_sa3_dit *dit = aria_sa3_dit_load(sf, &cfg);
    if (!dit) { printf("FAIL: dit load\n"); return 1; }

    int T = 16, C = cfg.io_channels, n = C * T, LD = cfg.local_add_cond_dim;  /* 256, 257 */
    float *x = malloc((size_t)n * 4);
    for (int i = 0; i < n; i++) x[i] = 0.1f * sinf((float)i * 0.01f);
    float *cross = calloc((size_t)257 * 768, 4), *glob = calloc(768, 4);   /* unconditional */
    float *v0 = malloc((size_t)n * 4), *v1 = malloc((size_t)n * 4);
    float *local_raw = malloc((size_t)T * LD * 4);
    for (int i = 0; i < T * LD; i++) local_raw[i] = (i % LD < LD / 2) ? 1.0f : 0.05f * (float)(i % 7);

    step(dit, T, cross, glob, x, v0, NULL, LD);          /* no local cond */
    step(dit, T, cross, glob, x, v1, local_raw, LD);     /* with synthetic local cond */

    float md = 0.0f; int nan = 0;
    for (int i = 0; i < n; i++) { if (!isfinite(v1[i])) nan++; float d = fabsf(v1[i] - v0[i]); if (d > md) md = d; }
    printf("local-cond: maxdiff(with,without)=%.3e nan=%d (T=%d local_dim=%d)\n", md, nan, T, LD);

    free(x); free(cross); free(glob); free(v0); free(v1); free(local_raw);
    aria_sa3_dit_free(dit); safetensors_close(sf);
    if (nan || md < 1e-6f) { printf("test_local: FAILED (cond had no finite effect)\n"); return 1; }
    printf("test_local: OK\n");
    return 0;
}
