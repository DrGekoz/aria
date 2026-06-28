/* test_taae_med.c - parity for the medium decoder block kernel (E10.2):
 * differential DyT attention + SLIDING-WINDOW (banded) mask + SiLU/Sin GLU FF.
 * Loads a synthetic block (random weights, dim 512) dumped by Python. ARIA_DUMPS only. */
#include "../src/aria_taae.h"
#include "../src/aria_ops.h"
#include "../src/aria_arena.h"
#include "../src/aria_parity.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const char *DUMPS;
static int fails = 0;

static float *L(const char *tag, const char *fn, aria_parity_tensor *t) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/taae_med/%s_%s.atns", DUMPS, tag, fn);
    if (aria_parity_load(path, t) != 0) { printf("FAIL: load %s\n", path); fails++; return NULL; }
    return t->data;
}
static float Ls(const char *tag, const char *fn, aria_parity_tensor *t) {
    float *p = L(tag, fn, t); return p ? p[0] : 0.0f;
}

static void run(const char *tag, int sinusoidal) {
    const int dim = 512, H = 8, hd = 64, inner = 1536, rot = 32;
    aria_parity_tensor T[24]; int nt = 0;
    taae_block_w w;
    w.pre_alpha = Ls(tag, "pre_alpha", &T[nt++]); w.pre_gamma = L(tag, "pre_gamma", &T[nt++]); w.pre_beta = L(tag, "pre_beta", &T[nt++]);
    w.to_qkv = L(tag, "to_qkv", &T[nt++]);
    w.qn_alpha = Ls(tag, "qn_alpha", &T[nt++]); w.qn_gamma = L(tag, "qn_gamma", &T[nt++]); w.qn_beta = L(tag, "qn_beta", &T[nt++]);
    w.kn_alpha = Ls(tag, "kn_alpha", &T[nt++]); w.kn_gamma = L(tag, "kn_gamma", &T[nt++]); w.kn_beta = L(tag, "kn_beta", &T[nt++]);
    w.to_out = L(tag, "to_out", &T[nt++]);
    w.ff_alpha = Ls(tag, "ff_alpha", &T[nt++]); w.ff_gamma = L(tag, "ff_gamma", &T[nt++]); w.ff_beta = L(tag, "ff_beta", &T[nt++]);
    w.ff_in_w = L(tag, "ff_in_w", &T[nt++]); w.ff_in_b = L(tag, "ff_in_b", &T[nt++]);
    w.ff_out_w = L(tag, "ff_out_w", &T[nt++]); w.ff_out_b = L(tag, "ff_out_b", &T[nt++]);

    aria_parity_tensor X, ref;
    float *x = L(tag, "x", &X), *r = L(tag, "out", &ref);
    if (fails || !x || !r) return;
    int N = (int)X.shape[0];

    float *rc = malloc((size_t)N * (rot / 2) * sizeof(float));
    float *rs = malloc((size_t)N * (rot / 2) * sizeof(float));
    aria_rope_freqs(rc, rs, N, rot, 10000.0f);

    /* sliding-window attention, half-width 17 (query i attends to keys [i-17,i+17]) */
    const int W = 17;

    aria_arena ar; aria_arena_init(&ar, taae_med_block_floats(N, dim, inner) * sizeof(float));
    float *out = malloc((size_t)N * dim * sizeof(float));
    memcpy(out, x, (size_t)N * dim * sizeof(float));
    taae_med_block_forward(out, N, dim, H, hd, inner, &w, rc, rs, W, sinusoidal, &ar);
    aria_arena_free(&ar);

    float md = aria_parity_maxabsdiff(out, r, ref.numel), maxref = 0;
    for (int64_t i = 0; i < ref.numel; i++) { float a = fabsf(r[i]); if (a > maxref) maxref = a; }
    float thr = 1e-3f + 3e-3f * maxref;
    if (md > thr) { printf("FAIL med_block[%s]: maxdiff=%.3e thr=%.3e (max|ref|=%.3f)\n", tag, md, thr, maxref); fails++; }
    else printf("ok   med_block[%s]: maxdiff=%.3e (max|ref|=%.3f)\n", tag, md, maxref);

    free(out); free(rc); free(rs);
    for (int i = 0; i < nt; i++) aria_parity_free(&T[i]);
    aria_parity_free(&X); aria_parity_free(&ref);
}

int main(void) {
    DUMPS = getenv("ARIA_DUMPS");
    if (!DUMPS) { printf("test_taae_med: SKIP (set ARIA_DUMPS)\n"); return 0; }
    run("silu", 0);
    run("sin", 1);
    if (fails) { printf("test_taae_med: FAILED\n"); return 1; }
    printf("test_taae_med: OK\n");
    return 0;
}
