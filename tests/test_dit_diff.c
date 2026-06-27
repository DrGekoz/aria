/* test_dit_diff.c - parity for the DIFFERENTIAL DiT block (medium's attention).
 * Loads a synthetic differential block (random weights) dumped by Python and
 * checks aria's differential path: out = attn(q,k,v) - attn(q_diff,k_diff,v) for
 * both self- and cross-attention. Needs ARIA_DUMPS only (weights are in the dump). */
#include "../src/aria_sa3_dit.h"
#include "../src/aria_ops.h"
#include "../src/aria_parity.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const char *DUMPS;
static int fails = 0;

static float *load(const char *name, aria_parity_tensor *t) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/dit_diff/%s.atns", DUMPS, name);
    if (aria_parity_load(path, t) != 0) { printf("FAIL: cannot load %s\n", path); fails++; return NULL; }
    return t->data;
}

int main(void) {
    DUMPS = getenv("ARIA_DUMPS");
    if (!DUMPS) { printf("test_dit_diff: SKIP (set ARIA_DUMPS)\n"); return 0; }

    const int dim = 256, num_heads = 4, head_dim = 64, inner = 1024, rot_dim = 32;

    aria_parity_tensor T[20]; int nt = 0;
    aria_dit_block_w w;
    #define W(field, name) do { w.field = load(name, &T[nt++]); } while (0)
    W(pre_norm, "pre_norm"); W(cross_norm, "cross_norm"); W(ff_norm, "ff_norm");
    W(sa_to_qkv, "sa_to_qkv"); W(sa_q_norm, "sa_q_norm"); W(sa_k_norm, "sa_k_norm"); W(sa_to_out, "sa_to_out");
    W(ca_to_q, "ca_to_q"); W(ca_to_kv, "ca_to_kv"); W(ca_q_norm, "ca_q_norm"); W(ca_k_norm, "ca_k_norm");
    W(ca_to_out, "ca_to_out"); W(ff_in_w, "ff_in_w"); W(ff_in_b, "ff_in_b");
    W(ff_out_w, "ff_out_w"); W(ff_out_b, "ff_out_b"); W(to_scale_shift_gate, "ssg");
    #undef W
    if (fails) return 1;

    aria_parity_tensor x, ctx, glob, ref;
    if (!load("x", &x) || !load("context", &ctx) || !load("global", &glob) || !load("out", &ref)) return 1;
    int S = (int)x.shape[0], Sc = (int)ctx.shape[0];

    float *cosb = malloc((size_t)S * (rot_dim / 2) * sizeof(float));
    float *sinb = malloc((size_t)S * (rot_dim / 2) * sizeof(float));
    aria_rope_freqs(cosb, sinb, S, rot_dim, 10000.0f);

    float *out = malloc((size_t)S * dim * sizeof(float));
    memcpy(out, x.data, (size_t)S * dim * sizeof(float));
    aria_dit_block_forward(out, S, dim, num_heads, head_dim, inner,
                           ctx.data, Sc, dim, glob.data, cosb, sinb, rot_dim, &w, 1);  /* differential */

    float md = aria_parity_maxabsdiff(out, ref.data, ref.numel), maxref = 0.0f;
    for (int64_t i = 0; i < ref.numel; i++) { float a = fabsf(ref.data[i]); if (a > maxref) maxref = a; }
    float thresh = 1e-3f + 3e-3f * maxref;
    if (md > thresh) { printf("FAIL diff_block: maxdiff=%.3e thresh=%.3e (max|ref|=%.3f)\n", md, thresh, maxref); fails++; }
    else printf("ok   diff_block: maxdiff=%.3e (max|ref|=%.3f)\n", md, maxref);

    free(out); free(cosb); free(sinb);
    for (int i = 0; i < nt; i++) aria_parity_free(&T[i]);
    aria_parity_free(&x); aria_parity_free(&ctx); aria_parity_free(&glob); aria_parity_free(&ref);

    if (fails) { printf("test_dit_diff: FAILED\n"); return 1; }
    printf("test_dit_diff: OK\n");
    return 0;
}
