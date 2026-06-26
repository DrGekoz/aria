/*
 * aria_t5enc.c - T5Gemma encoder (CPU). See aria_t5enc.h and the porting spec.
 */

#include "aria_t5enc.h"
#include "aria_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define T5_D    768
#define T5_H    12
#define T5_HD   64
#define T5_ROT  64     /* full-dim RoPE */
#define T5_I    2048
#define T5_L    12
#define T5_EPS  1e-6f
#define T5_SCALE   0.125f   /* query_pre_attn_scalar^-0.5 = 64^-0.5 */
#define T5_SOFTCAP 50.0f

typedef struct {
    const float *pre_attn_norm, *post_attn_norm, *pre_ff_norm, *post_ff_norm; /* [768] */
    const float *wq, *wk, *wv, *wo;   /* [768,768] */
    const float *w_gate, *w_up;       /* [2048,768] */
    const float *w_down;              /* [768,2048] */
} t5_layer;

struct aria_t5enc {
    t5_layer layers[T5_L];
    const float *final_norm;          /* [768] */
    const float *padding_embedding;   /* [768] f32, borrowed from main_sf */
    const uint16_t *embed_bf16;       /* [256000,768] bf16, borrowed from t5sf */
    safetensors_file_t *t5sf;         /* owned (kept open for embed_bf16) */
    float **owned; int n_owned, cap_owned;
    int failed;
};

static inline float bf16_to_f32(uint16_t u) {
    uint32_t v = (uint32_t)u << 16;
    float f; memcpy(&f, &v, sizeof(f)); return f;
}

/* extract [H,seq,HD] (head-major) from [seq, H*HD] */
static void extract_heads(float *dst, const float *src, int seq, int H, int hd) {
    for (int h = 0; h < H; h++)
        for (int s = 0; s < seq; s++)
            memcpy(dst + ((size_t)h * seq + s) * hd, src + (size_t)s * H * hd + (size_t)h * hd,
                   (size_t)hd * sizeof(float));
}
static void merge_heads(float *dst, const float *src, int seq, int H, int hd) {
    for (int h = 0; h < H; h++)
        for (int s = 0; s < seq; s++)
            memcpy(dst + (size_t)s * H * hd + (size_t)h * hd, src + ((size_t)h * seq + s) * hd,
                   (size_t)hd * sizeof(float));
}

void aria_t5enc_encode(const aria_t5enc *e, float *cond, const int *ids, int seq, int n_real) {
    const int D = T5_D, H = T5_H, hd = T5_HD, I = T5_I;
    float normalizer = sqrtf((float)D);

    float *h = malloc((size_t)seq * D * sizeof(float));
    for (int p = 0; p < seq; p++) {
        const uint16_t *row = e->embed_bf16 + (size_t)ids[p] * D;
        float *hr = h + (size_t)p * D;
        for (int j = 0; j < D; j++) hr[j] = bf16_to_f32(row[j]) * normalizer;
    }

    float *rcos = malloc((size_t)seq * (T5_ROT / 2) * sizeof(float));
    float *rsin = malloc((size_t)seq * (T5_ROT / 2) * sizeof(float));
    aria_rope_freqs(rcos, rsin, seq, T5_ROT, 10000.0f);

    /* padding mask bias [seq,seq] (per key, broadcast over queries) */
    float *maskmat = NULL;
    if (n_real < seq) {
        maskmat = malloc((size_t)seq * seq * sizeof(float));
        for (int q = 0; q < seq; q++)
            for (int k = 0; k < seq; k++)
                maskmat[(size_t)q * seq + k] = (k < n_real) ? 0.0f : -3.4e38f;
    }

    float *res = malloc((size_t)seq * D * sizeof(float));
    float *a = malloc((size_t)seq * D * sizeof(float));
    float *proj = malloc((size_t)seq * D * sizeof(float));
    float *qh = malloc((size_t)H * seq * hd * sizeof(float));
    float *kh = malloc((size_t)H * seq * hd * sizeof(float));
    float *vh = malloc((size_t)H * seq * hd * sizeof(float));
    float *oh = malloc((size_t)H * seq * hd * sizeof(float));
    float *merged = malloc((size_t)seq * D * sizeof(float));
    float *scores = malloc((size_t)seq * seq * sizeof(float));
    float *gate = malloc((size_t)seq * I * sizeof(float));
    float *up = malloc((size_t)seq * I * sizeof(float));

    for (int li = 0; li < T5_L; li++) {
        const t5_layer *L = &e->layers[li];

        /* ---- self-attention (sandwich norm) ---- */
        memcpy(res, h, (size_t)seq * D * sizeof(float));
        aria_gemma_rmsnorm(a, h, L->pre_attn_norm, seq, D, T5_EPS);
        aria_linear(proj, a, L->wq, NULL, seq, D, D); extract_heads(qh, proj, seq, H, hd);
        aria_linear(proj, a, L->wk, NULL, seq, D, D); extract_heads(kh, proj, seq, H, hd);
        aria_linear(proj, a, L->wv, NULL, seq, D, D); extract_heads(vh, proj, seq, H, hd);
        aria_rope_apply(qh, rcos, rsin, H, seq, hd, T5_ROT);
        aria_rope_apply(kh, rcos, rsin, H, seq, hd, T5_ROT);
        for (int hi = 0; hi < H; hi++) {
            const float *qhh = qh + (size_t)hi * seq * hd;
            const float *khh = kh + (size_t)hi * seq * hd;
            const float *vhh = vh + (size_t)hi * seq * hd;
            float *ohh = oh + (size_t)hi * seq * hd;
            aria_linear(scores, qhh, khh, NULL, seq, hd, seq);   /* q @ k^T */
            for (size_t i = 0; i < (size_t)seq * seq; i++) scores[i] *= T5_SCALE;
            aria_softcap(scores, seq * seq, T5_SOFTCAP);
            aria_softmax_inplace(scores, seq, seq, maskmat);
            aria_matmul(ohh, scores, vhh, seq, seq, hd);
        }
        merge_heads(merged, oh, seq, H, hd);
        aria_linear(a, merged, L->wo, NULL, seq, D, D);
        aria_gemma_rmsnorm(a, a, L->post_attn_norm, seq, D, T5_EPS);   /* on attn output */
        for (size_t i = 0; i < (size_t)seq * D; i++) h[i] = res[i] + a[i];

        /* ---- GeGLU feed-forward (sandwich norm) ---- */
        memcpy(res, h, (size_t)seq * D * sizeof(float));
        aria_gemma_rmsnorm(a, h, L->pre_ff_norm, seq, D, T5_EPS);
        aria_linear(gate, a, L->w_gate, NULL, seq, D, I);
        aria_linear(up, a, L->w_up, NULL, seq, D, I);
        aria_gelu_tanh(gate, seq * I);
        for (size_t i = 0; i < (size_t)seq * I; i++) gate[i] *= up[i];
        aria_linear(a, gate, L->w_down, NULL, seq, I, D);
        aria_gemma_rmsnorm(a, a, L->post_ff_norm, seq, D, T5_EPS);     /* on mlp output */
        for (size_t i = 0; i < (size_t)seq * D; i++) h[i] = res[i] + a[i];
    }

    aria_gemma_rmsnorm(h, h, e->final_norm, seq, D, T5_EPS);

    /* learned-padding overwrite */
    for (int p = 0; p < seq; p++) {
        if (p < n_real) memcpy(cond + (size_t)p * D, h + (size_t)p * D, (size_t)D * sizeof(float));
        else            memcpy(cond + (size_t)p * D, e->padding_embedding, (size_t)D * sizeof(float));
    }

    free(h); free(rcos); free(rsin); free(maskmat); free(res); free(a); free(proj);
    free(qh); free(kh); free(vh); free(oh); free(merged); free(scores); free(gate); free(up);
}

/* ---- loader ---- */
static const float *track(aria_t5enc *e, safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) { fprintf(stderr, "aria_t5enc_load: missing %s\n", name); e->failed = 1; return NULL; }
    float *p = safetensors_get_f32(sf, t);  /* dequantizes BF16 -> F32 */
    if (!p) { e->failed = 1; return NULL; }
    if (e->n_owned == e->cap_owned) {
        e->cap_owned = e->cap_owned ? e->cap_owned * 2 : 256;
        e->owned = realloc(e->owned, (size_t)e->cap_owned * sizeof(float *));
    }
    e->owned[e->n_owned++] = p;
    return p;
}

aria_t5enc *aria_t5enc_load(const char *model_dir, const char *subfolder,
                            safetensors_file_t *main_sf) {
    aria_t5enc *e = calloc(1, sizeof(*e));
    if (!e) return NULL;

    char path[1024];
    snprintf(path, sizeof(path), "%s/%s/model.safetensors", model_dir, subfolder);
    e->t5sf = safetensors_open(path);
    if (!e->t5sf) { fprintf(stderr, "aria_t5enc_load: cannot open %s\n", path); free(e); return NULL; }

    /* embedding table: borrow BF16 (gathered lazily) */
    const safetensor_t *emb = safetensors_find(e->t5sf, "model.encoder.embed_tokens.weight");
    if (!emb || emb->dtype != DTYPE_BF16) {
        fprintf(stderr, "aria_t5enc_load: embed_tokens not BF16\n");
        aria_t5enc_free(e); return NULL;
    }
    e->embed_bf16 = (const uint16_t *)safetensors_data(e->t5sf, emb);

    e->final_norm = track(e, e->t5sf, "model.encoder.norm.weight");
    char nm[256];
    #define P "model.encoder.layers.%d."
    for (int i = 0; i < T5_L; i++) {
        t5_layer *L = &e->layers[i];
        #define G(dst, suf) do { snprintf(nm, sizeof(nm), P suf, i); dst = track(e, e->t5sf, nm); } while (0)
        G(L->pre_attn_norm, "pre_self_attn_layernorm.weight");
        G(L->wq, "self_attn.q_proj.weight");
        G(L->wk, "self_attn.k_proj.weight");
        G(L->wv, "self_attn.v_proj.weight");
        G(L->wo, "self_attn.o_proj.weight");
        G(L->post_attn_norm, "post_self_attn_layernorm.weight");
        G(L->pre_ff_norm, "pre_feedforward_layernorm.weight");
        G(L->w_gate, "mlp.gate_proj.weight");
        G(L->w_up, "mlp.up_proj.weight");
        G(L->w_down, "mlp.down_proj.weight");
        G(L->post_ff_norm, "post_feedforward_layernorm.weight");
        #undef G
    }
    #undef P

    e->padding_embedding = safetensors_f32_ptr(main_sf, safetensors_find(main_sf,
        "conditioner.conditioners.prompt.padding_embedding"));
    if (!e->padding_embedding) { fprintf(stderr, "aria_t5enc_load: missing padding_embedding\n"); e->failed = 1; }

    if (e->failed) { aria_t5enc_free(e); return NULL; }
    return e;
}

void aria_t5enc_free(aria_t5enc *e) {
    if (!e) return;
    for (int i = 0; i < e->n_owned; i++) free(e->owned[i]);
    free(e->owned);
    if (e->t5sf) safetensors_close(e->t5sf);
    free(e);
}
