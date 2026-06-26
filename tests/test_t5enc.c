/* test_t5enc.c - parity: T5Gemma encoder vs the real T5GemmaConditioner.
 * Runs the C encoder on dumped token ids and compares the final [256,768]
 * conditioning. Needs ARIA_MODEL + ARIA_DUMPS. */
#include "../src/aria_safetensors.h"
#include "../src/aria_t5enc.h"
#include "../src/aria_parity.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const char *DUMPS;
static int fails = 0;

static int load(const char *name, aria_parity_tensor *t) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/t5enc/%s", DUMPS, name);
    if (aria_parity_load(path, t) != 0) { printf("FAIL: load %s\n", path); fails++; return -1; }
    return 0;
}

int main(void) {
    const char *model = getenv("ARIA_MODEL");
    DUMPS = getenv("ARIA_DUMPS");
    if (!model || !DUMPS) { printf("test_t5enc: SKIP (set ARIA_MODEL and ARIA_DUMPS)\n"); return 0; }

    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors", model);
    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) { printf("FAIL: open main safetensors\n"); return 1; }

    aria_t5enc *e = aria_t5enc_load(model, "t5gemma-b-b-ul2", sf);
    if (!e) { printf("FAIL: t5enc load\n"); safetensors_close(sf); return 1; }

    for (int idx = 0; idx < 3; idx++) {
        char nm[64];
        aria_parity_tensor ids_t, nreal_t, condref;
        snprintf(nm, sizeof(nm), "ids_%d.atns", idx);   if (load(nm, &ids_t)) break;
        snprintf(nm, sizeof(nm), "nreal_%d.atns", idx); if (load(nm, &nreal_t)) break;
        snprintf(nm, sizeof(nm), "cond_%d.atns", idx);  if (load(nm, &condref)) break;

        int seq = (int)ids_t.numel;
        int n_real = (int)nreal_t.data[0];
        int *ids = malloc((size_t)seq * sizeof(int));
        for (int i = 0; i < seq; i++) ids[i] = (int)(ids_t.data[i] + 0.5f);

        float *cond = malloc((size_t)seq * 768 * sizeof(float));
        aria_t5enc_encode(e, cond, ids, seq, n_real);

        float md = aria_parity_maxabsdiff(cond, condref.data, condref.numel), maxref = 0.0f;
        for (int64_t i = 0; i < condref.numel; i++) { float v = fabsf(condref.data[i]); if (v > maxref) maxref = v; }
        float thr = 2e-3f + 5e-3f * maxref;
        if (md > thr) { printf("FAIL t5enc[%d] (n=%d): maxdiff=%.3e thr=%.3e (max|ref|=%.3f)\n", idx, n_real, md, thr, maxref); fails++; }
        else printf("ok   t5enc[%d] (n_real=%d): maxdiff=%.3e (max|ref|=%.3f)\n", idx, n_real, md, maxref);

        free(ids); free(cond);
        aria_parity_free(&ids_t); aria_parity_free(&nreal_t); aria_parity_free(&condref);
    }

    aria_t5enc_free(e);
    safetensors_close(sf);
    if (fails) { printf("test_t5enc: FAILED\n"); return 1; }
    printf("test_t5enc: OK\n");
    return 0;
}
