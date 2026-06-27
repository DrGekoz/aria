/* test_dit.c - keystone parity: SA3 DiT block-0 forward vs PyTorch.
 * C loads block-0's real weights from the model (validating the name mapping)
 * and reproduces the block output. Needs ARIA_MODEL + ARIA_DUMPS. */
#include "../src/aria_safetensors.h"
#include "../src/aria_sa3_dit.h"
#include "../src/aria_ops.h"
#include "../src/aria_parity.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static safetensors_file_t *SF;
static const char *DUMPS;
static int fails = 0;

/* load a block-0 tensor by suffix as f32 (caller keeps pointer until end) */
static float *load_w(const char *suffix) {
    char name[256];
    snprintf(name, sizeof(name), "model.model.transformer.layers.0.%s", suffix);
    const safetensor_t *t = safetensors_find(SF, name);
    if (!t) { printf("FAIL: missing weight %s\n", name); fails++; return NULL; }
    return safetensors_get_f32(SF, t);
}

static int load_dump(const char *name, aria_parity_tensor *t) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/dit/%s", DUMPS, name);
    if (aria_parity_load(path, t) != 0) { printf("FAIL: cannot load %s\n", path); fails++; return -1; }
    return 0;
}

int main(void) {
    const char *model = getenv("ARIA_MODEL");
    DUMPS = getenv("ARIA_DUMPS");
    if (!model || !DUMPS) { printf("test_dit: SKIP (set ARIA_MODEL and ARIA_DUMPS)\n"); return 0; }

    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors", model);
    SF = safetensors_open(path);
    if (!SF) { printf("FAIL: cannot open %s\n", path); return 1; }

    const int dim = 1024, num_heads = 16, head_dim = 64, inner = 4096, rot_dim = 32;

    aria_dit_block_w w;
    float *bufs[18]; int nb = 0;
    #define W(field, suffix) do { w.field = load_w(suffix); bufs[nb++] = (float*)w.field; } while (0)
    W(pre_norm, "pre_norm.gamma");
    W(cross_norm, "cross_attend_norm.gamma");
    W(ff_norm, "ff_norm.gamma");
    W(sa_to_qkv, "self_attn.to_qkv.weight");
    W(sa_q_norm, "self_attn.q_norm.gamma");
    W(sa_k_norm, "self_attn.k_norm.gamma");
    W(sa_to_out, "self_attn.to_out.weight");
    W(ca_to_q, "cross_attn.to_q.weight");
    W(ca_to_kv, "cross_attn.to_kv.weight");
    W(ca_q_norm, "cross_attn.q_norm.gamma");
    W(ca_k_norm, "cross_attn.k_norm.gamma");
    W(ca_to_out, "cross_attn.to_out.weight");
    W(ff_in_w, "ff.ff.0.proj.weight");
    W(ff_in_b, "ff.ff.0.proj.bias");
    W(ff_out_w, "ff.ff.2.weight");
    W(ff_out_b, "ff.ff.2.bias");
    W(to_scale_shift_gate, "to_scale_shift_gate");
    #undef W
    if (fails) return 1;

    aria_parity_tensor x, ctx, glob, ref;
    if (load_dump("block0_x.atns", &x) || load_dump("block0_context.atns", &ctx) ||
        load_dump("block0_global.atns", &glob) || load_dump("block0_out.atns", &ref)) return 1;
    int S = (int)x.shape[0], Sc = (int)ctx.shape[0];

    float *cosb = malloc((size_t)S * (rot_dim / 2) * sizeof(float));
    float *sinb = malloc((size_t)S * (rot_dim / 2) * sizeof(float));
    aria_rope_freqs(cosb, sinb, S, rot_dim, 10000.0f);

    float *out = malloc((size_t)S * dim * sizeof(float));
    memcpy(out, x.data, (size_t)S * dim * sizeof(float));
    aria_dit_block_forward(out, S, dim, num_heads, head_dim, inner,
                           ctx.data, Sc, dim, glob.data, cosb, sinb, rot_dim, &w, 0);

    float md = aria_parity_maxabsdiff(out, ref.data, ref.numel);
    float maxref = 0.0f;
    for (int64_t i = 0; i < ref.numel; i++) { float a = fabsf(ref.data[i]); if (a > maxref) maxref = a; }
    float thresh = 1e-3f + 3e-3f * maxref;  /* deeper graph -> looser than single ops */
    if (md > thresh)
        printf("FAIL block0: maxdiff=%.3e thresh=%.3e (max|ref|=%.3f)\n", md, thresh, maxref);
    else
        printf("ok   block0: maxdiff=%.3e (max|ref|=%.3f)\n", md, maxref);
    if (md > thresh) fails++;

    free(out); free(cosb); free(sinb);
    for (int i = 0; i < nb; i++) free(bufs[i]);
    aria_parity_free(&x); aria_parity_free(&ctx); aria_parity_free(&glob); aria_parity_free(&ref);
    safetensors_close(SF);

    if (fails) { printf("test_dit: FAILED\n"); return 1; }
    printf("test_dit: OK\n");
    return 0;
}
