/* test_inpaint.c - parity for the inpaint conditioning path (E7.2 + E7.3):
 *   1. audio-space mask -> nearest-interp to latent (mask_lat)
 *   2. local_add_cond[T,257] = [mask | latent*mask]
 *   3. DiT velocity with a live local_add_cond (E1.6 hook, real mask)
 * Needs ARIA_MODEL + ARIA_DUMPS. */
#include "../src/aria_safetensors.h"
#include "../src/aria_sa3.h"
#include "../src/aria_sa3_dit.h"
#include "../src/aria_cond.h"
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
    snprintf(path, sizeof(path), "%s/inpaint/%s", DUMPS, name);
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
    if (!model || !DUMPS) { printf("test_inpaint: SKIP (set ARIA_MODEL and ARIA_DUMPS)\n"); return 0; }

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

    aria_parity_tensor x, cross, glob, tt, initlat, maskA, maskL, local, out;
    if (load("x.atns", &x) || load("cross.atns", &cross) || load("global.atns", &glob) ||
        load("t.atns", &tt) || load("init_latent.atns", &initlat) || load("mask_audio.atns", &maskA) ||
        load("mask_lat.atns", &maskL) || load("local.atns", &local) || load("out.atns", &out))
        return 1;

    int T = (int)x.shape[1];                 /* x[256,T] */
    int n_cond = (int)cross.shape[0];        /* 257 */
    int audio_len = (int)maskA.shape[0];     /* T*4096 */

    /* E7.2a: nearest-interp the audio mask to latent length */
    {
        float *ml = malloc((size_t)T * sizeof(float));
        aria_inpaint_mask_latent(ml, maskA.data, audio_len, T);
        compare("mask_latent", ml, maskL.data, T, 0.0f, 0.0f);
        free(ml);
    }
    /* E7.2b: build local_add_cond[T,257] = [mask | latent*mask]; ref is [257,T] (channel-major) */
    float *local_TC = malloc((size_t)T * (n_cond) * sizeof(float));  /* n_cond = 257 */
    {
        aria_inpaint_local_cond(local_TC, initlat.data, maskL.data, T);
        /* compare token-major [T,257] vs ref channel-major local[257,T] */
        float *ref_TC = malloc((size_t)T * 257 * sizeof(float));
        for (int c = 0; c < 257; c++) for (int t = 0; t < T; t++)
            ref_TC[(size_t)t * 257 + c] = local.data[(size_t)c * T + t];
        compare("local_cond", local_TC, ref_TC, (int64_t)T * 257, 1e-6f, 1e-6f);
        free(ref_TC);
    }
    /* E7.3: DiT velocity with the live local_add_cond (real mask) */
    {
        float t = tt.data[0];
        float *v = malloc((size_t)256 * T * sizeof(float));
        aria_sa3_dit_req *req = aria_sa3_dit_req_begin(m, T, cross.data, n_cond, glob.data);
        aria_sa3_dit_req_set_local(req, local_TC, T, 257);
        aria_sa3_dit_step(m, req, v, x.data, t);
        aria_sa3_dit_req_end(req);
        compare("dit_velocity", v, out.data, out.numel, 2e-3f, 1e-2f);
        free(v);
    }
    free(local_TC);

    aria_parity_free(&x); aria_parity_free(&cross); aria_parity_free(&glob);
    aria_parity_free(&tt); aria_parity_free(&initlat); aria_parity_free(&maskA);
    aria_parity_free(&maskL); aria_parity_free(&local); aria_parity_free(&out);
    aria_sa3_dit_free(m);
    safetensors_close(sf);

    if (fails) { printf("test_inpaint: FAILED\n"); return 1; }
    printf("test_inpaint: OK\n");
    return 0;
}
