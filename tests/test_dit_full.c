/* test_dit_full.c - parity: full SA3 DiT denoiser_forward vs PyTorch.
 * Loads ALL DiT weights from the model and runs the complete 20-block forward
 * (embedders + preprocess + project_in + memory tokens + blocks + project_out +
 * postprocess). Validates E1.3-1.5 and E4.2/E4.4 transitively.
 * Needs ARIA_MODEL + ARIA_DUMPS. */
#include "../src/aria_safetensors.h"
#include "../src/aria_sa3.h"
#include "../src/aria_sa3_dit.h"
#include "../src/aria_json.h"
#include "../src/aria_parity.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const char *DUMPS;

static int load_dump(const char *name, aria_parity_tensor *t) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/dit/%s", DUMPS, name);
    if (aria_parity_load(path, t) != 0) { printf("FAIL: cannot load %s\n", path); return -1; }
    return 0;
}

int main(void) {
    const char *model = getenv("ARIA_MODEL");
    DUMPS = getenv("ARIA_DUMPS");
    if (!model || !DUMPS) { printf("test_dit_full: SKIP (set ARIA_MODEL and ARIA_DUMPS)\n"); return 0; }

    char path[1024];
    snprintf(path, sizeof(path), "%s/model_config.json", model);
    char *cfgjson = aria_read_file(path, NULL);
    if (!cfgjson) { printf("FAIL: read model_config.json\n"); return 1; }
    aria_sa3_config cfg;
    if (aria_sa3_parse_config(cfgjson, &cfg) != 0) { printf("FAIL: parse config\n"); return 1; }
    free(cfgjson);

    snprintf(path, sizeof(path), "%s/model.safetensors", model);
    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) { printf("FAIL: open safetensors\n"); return 1; }

    aria_sa3_dit *m = aria_sa3_dit_load(sf, &cfg);
    if (!m) { printf("FAIL: dit load\n"); safetensors_close(sf); return 1; }

    aria_parity_tensor x, cross, glob, ref, tt;
    if (load_dump("full_x.atns", &x) || load_dump("full_cross.atns", &cross) ||
        load_dump("full_global.atns", &glob) || load_dump("full_out.atns", &ref) ||
        load_dump("full_t.atns", &tt)) return 1;

    int C = (int)x.shape[0], T = (int)x.shape[1];
    int n_cond = (int)cross.shape[0];
    float t = tt.data[0];

    float *out = malloc((size_t)C * T * sizeof(float));
    aria_sa3_dit_forward(m, out, x.data, T, t, cross.data, n_cond, glob.data);

    float md = aria_parity_maxabsdiff(out, ref.data, ref.numel);
    float maxref = 0.0f;
    for (int64_t i = 0; i < ref.numel; i++) { float a = fabsf(ref.data[i]); if (a > maxref) maxref = a; }
    float thresh = 2e-3f + 5e-3f * maxref;  /* 20-block deep graph */
    int fail = md > thresh;
    printf("%s full_dit: maxdiff=%.3e thresh=%.3e (max|ref|=%.3f, C=%d T=%d)\n",
           fail ? "FAIL" : "ok  ", md, thresh, maxref, C, T);

    free(out);
    aria_parity_free(&x); aria_parity_free(&cross); aria_parity_free(&glob);
    aria_parity_free(&ref); aria_parity_free(&tt);
    aria_sa3_dit_free(m);
    safetensors_close(sf);

    if (fail) { printf("test_dit_full: FAILED\n"); return 1; }
    printf("test_dit_full: OK\n");
    return 0;
}
