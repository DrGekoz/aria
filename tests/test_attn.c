/* test_attn.c - parity for RoPE, attention, and the GLU FFN vs PyTorch dumps.
 * Needs ARIA_DUMPS (set by `make parity`). SKIPs cleanly if unset. */
#include "../src/aria_ops.h"
#include "../src/aria_parity.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const char *DUMPS;
static int fails = 0;

static int load(const char *name, aria_parity_tensor *t) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/ops/%s", DUMPS, name);
    if (aria_parity_load(path, t) != 0) { printf("FAIL: cannot load %s\n", path); fails++; return -1; }
    return 0;
}

/* combined atol+rtol gate, like the number-cond test */
static void compare(const char *what, const float *got, const float *ref, int64_t n) {
    float md = aria_parity_maxabsdiff(got, ref, n);
    float maxref = 0.0f;
    for (int64_t i = 0; i < n; i++) { float a = fabsf(ref[i]); if (a > maxref) maxref = a; }
    float thresh = 3e-4f + 2e-3f * maxref;
    if (md > thresh) { printf("FAIL %s: maxdiff=%.3e thresh=%.3e (max|ref|=%.3f)\n", what, md, thresh, maxref); fails++; }
    else printf("ok   %s: maxdiff=%.3e (max|ref|=%.3f)\n", what, md, maxref);
}

int main(void) {
    DUMPS = getenv("ARIA_DUMPS");
    if (!DUMPS) { printf("test_attn: SKIP (set ARIA_DUMPS)\n"); return 0; }

    /* ---- RoPE: q_in [H,N,D], rot_dim=D ---- */
    {
        aria_parity_tensor qin, qout;
        if (load("rope_q_in.atns", &qin) || load("rope_q_out.atns", &qout)) return 1;
        int H = (int)qin.shape[0], N = (int)qin.shape[1], D = (int)qin.shape[2];
        int rot = D;
        float *cosb = malloc((size_t)N * (rot/2) * sizeof(float));
        float *sinb = malloc((size_t)N * (rot/2) * sizeof(float));
        aria_rope_freqs(cosb, sinb, N, rot, 10000.0f);
        float *q = malloc((size_t)qin.numel * sizeof(float));
        memcpy(q, qin.data, (size_t)qin.numel * sizeof(float));
        aria_rope_apply(q, cosb, sinb, H, N, D, rot);
        compare("rope", q, qout.data, qin.numel);
        free(q); free(cosb); free(sinb);
        aria_parity_free(&qin); aria_parity_free(&qout);
    }

    /* ---- attention: q[H,Nq,D] k/v[H,Nk,D] ---- */
    {
        aria_parity_tensor q, k, v, o;
        if (load("attn_q.atns", &q) || load("attn_k.atns", &k) ||
            load("attn_v.atns", &v) || load("attn_out.atns", &o)) return 1;
        int H = (int)q.shape[0], Nq = (int)q.shape[1], D = (int)q.shape[2];
        int Nk = (int)k.shape[1];
        float *out = malloc((size_t)o.numel * sizeof(float));
        aria_attention(out, q.data, k.data, v.data, H, Nq, Nk, D, NULL);
        compare("attention", out, o.data, o.numel);
        free(out);
        aria_parity_free(&q); aria_parity_free(&k); aria_parity_free(&v); aria_parity_free(&o);
    }

    /* ---- GLU FeedForward ---- */
    {
        aria_parity_tensor x, y, Win, bin, Wout, bout;
        if (load("ff_x.atns", &x) || load("ff_y.atns", &y) ||
            load("ff_Win.atns", &Win) || load("ff_bin.atns", &bin) ||
            load("ff_Wout.atns", &Wout) || load("ff_bout.atns", &bout)) return 1;
        int N = (int)x.shape[0], dim = (int)x.shape[1];
        int two_inner = (int)Win.shape[0], inner = two_inner / 2;
        int dim_out = (int)Wout.shape[0];
        float *out = malloc((size_t)N * dim_out * sizeof(float));
        aria_ff_glu(out, x.data, N, dim, inner, dim_out, Win.data, bin.data, Wout.data, bout.data);
        compare("ff_glu", out, y.data, (int64_t)N * dim_out);
        free(out);
        aria_parity_free(&x); aria_parity_free(&y); aria_parity_free(&Win);
        aria_parity_free(&bin); aria_parity_free(&Wout); aria_parity_free(&bout);
    }

    if (fails) { printf("test_attn: %d failures\n", fails); return 1; }
    printf("test_attn: OK\n");
    return 0;
}
