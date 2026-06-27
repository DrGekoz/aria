/* test_inpaint_e2e.c - end-to-end inpaint parity (E7.4): encode init clip ->
 * build local-additive cond (keep first half) -> pingpong(DiT with local cond)
 * from pure noise + injected per-step noise -> latent -> decode -> audio.
 * Exercises the whole inpaint chain (E7.1 encoder + E7.2 mask + E7.3 live cond +
 * sampler + decoder). Needs ARIA_MODEL + ARIA_DUMPS. */
#include "../src/aria_safetensors.h"
#include "../src/aria_sa3.h"
#include "../src/aria_sa3_dit.h"
#include "../src/aria_sa3_dec.h"
#include "../src/aria_sa3_enc.h"
#include "../src/aria_cond.h"
#include "../src/aria_sampler.h"
#include "../src/aria_json.h"
#include "../src/aria_parity.h"
#include "../src/aria_wav.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const char *DUMPS;
static int fails = 0;

static int load(const char *name, aria_parity_tensor *t) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/inpaint_e2e/%s", DUMPS, name);
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

/* denoiser: DiT step reusing the request's cached local-additive cond */
typedef struct { const aria_sa3_dit *dit; aria_sa3_dit_req *req; } dctx;
static void dit_denoise(void *c, const float *x, float t, float *v, int n) {
    (void)n;
    const dctx *d = c;
    aria_sa3_dit_step(d->dit, d->req, v, x, t);
}

int main(void) {
    const char *model = getenv("ARIA_MODEL");
    DUMPS = getenv("ARIA_DUMPS");
    if (!model || !DUMPS) { printf("test_inpaint_e2e: SKIP (set ARIA_MODEL and ARIA_DUMPS)\n"); return 0; }

    char path[1024];
    snprintf(path, sizeof(path), "%s/model_config.json", model);
    char *cfgjson = aria_read_file(path, NULL);
    aria_sa3_config cfg;
    if (!cfgjson || aria_sa3_parse_config(cfgjson, &cfg) != 0) { printf("FAIL: config\n"); return 1; }
    free(cfgjson);

    snprintf(path, sizeof(path), "%s/model.safetensors", model);
    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) { printf("FAIL: open safetensors\n"); return 1; }
    aria_sa3_dit *dit = aria_sa3_dit_load(sf, &cfg);
    aria_sa3_dec *dec = aria_sa3_dec_load(sf);
    aria_sa3_enc *enc = aria_sa3_enc_load(sf);
    if (!dit || !dec || !enc) { printf("FAIL: model load\n"); return 1; }

    aria_parity_tensor initA, initLat, maskA, cross, glob, init_noise, step_noise, sched, latref, audref;
    if (load("init_audio.atns", &initA) || load("init_latent.atns", &initLat) ||
        load("mask_audio.atns", &maskA) || load("cross.atns", &cross) || load("global.atns", &glob) ||
        load("init_noise.atns", &init_noise) || load("step_noise.atns", &step_noise) ||
        load("sched.atns", &sched) || load("latent.atns", &latref) || load("audio.atns", &audref)) return 1;

    int T = (int)init_noise.shape[1];
    int audio_len = (int)maskA.shape[0];
    int n = 256 * T;
    int steps = (int)sched.numel - 1;

    /* 1. encode the init clip (E7.1) and check vs the dumped latent */
    float *initlat_c = malloc((size_t)256 * T * sizeof(float));
    int tl = aria_sa3_enc_forward(enc, initlat_c, initA.data, audio_len);
    if (tl != T) { printf("FAIL: enc T=%d expected %d\n", tl, T); fails++; }
    compare("encode_init", initlat_c, initLat.data, initLat.numel, 1e-3f, 5e-3f);

    /* 2. mask -> latent (E7.2) and local-additive cond [T,257] (E7.2) */
    float *mask_lat = malloc((size_t)T * sizeof(float));
    aria_inpaint_mask_latent(mask_lat, maskA.data, audio_len, T);
    float *local_TC = malloc((size_t)T * 257 * sizeof(float));
    aria_inpaint_local_cond(local_TC, initlat_c, mask_lat, T);

    /* 3. pingpong from pure noise with the live local cond + injected per-step noise */
    aria_sa3_dit_req *req = aria_sa3_dit_req_begin(dit, T, cross.data, 257, glob.data);
    aria_sa3_dit_req_set_local(req, local_TC, T, 257);
    dctx ctx = { dit, req };
    float *x = malloc((size_t)n * sizeof(float));
    memcpy(x, init_noise.data, (size_t)n * sizeof(float));
    aria_pingpong(x, n, sched.data, steps, dit_denoise, &ctx, NULL, step_noise.data);
    aria_sa3_dit_req_end(req);
    compare("latent", x, latref.data, latref.numel, 2e-3f, 1e-2f);

    /* 4. decode -> audio */
    float *audio = malloc((size_t)2 * T * 4096 * sizeof(float));
    aria_sa3_dec_forward(dec, audio, x, T);
    compare("audio", audio, audref.data, audref.numel, 2e-3f, 1e-2f);

    {
        aria_audio *a = aria_audio_alloc(44100, 2, (int64_t)T * 4096);
        for (int64_t i = 0; i < (int64_t)T * 4096; i++) {
            a->data[i * 2 + 0] = audio[i];
            a->data[i * 2 + 1] = audio[(int64_t)T * 4096 + i];
        }
        aria_wav_write("build/inpaint_e2e_out.wav", a, 32);
        aria_audio_free(a);
    }

    free(initlat_c); free(mask_lat); free(local_TC); free(x); free(audio);
    aria_parity_free(&initA); aria_parity_free(&initLat); aria_parity_free(&maskA);
    aria_parity_free(&cross); aria_parity_free(&glob); aria_parity_free(&init_noise);
    aria_parity_free(&step_noise); aria_parity_free(&sched); aria_parity_free(&latref); aria_parity_free(&audref);
    aria_sa3_dit_free(dit); aria_sa3_dec_free(dec); aria_sa3_enc_free(enc); safetensors_close(sf);

    if (fails) { printf("test_inpaint_e2e: FAILED\n"); return 1; }
    printf("test_inpaint_e2e: OK (wrote build/inpaint_e2e_out.wav)\n");
    return 0;
}
