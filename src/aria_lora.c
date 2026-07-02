/*
 * aria_lora.c - LoRA runtime-adapter loader + op for the SA3 DiT (E12.9, CPU).
 *
 * See aria_lora.h for the tensor-name map and the math. The base weights stay
 * untouched (quantized / mmap'd); this only carries the low-rank down/up factors.
 */

#include "aria_lora.h"
#include "aria_ops.h"
#include "aria_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* substring -> projection. Ordered so the more specific cross-attn names are seen
 * first; the substrings are aria's own DiT tensor names, so the map is prefix- and
 * variant-agnostic. */
static const struct { const char *sub; aria_lora_proj proj; } LORA_PROJMAP[] = {
    { "self_attn.to_qkv",  ARIA_LORA_SA_TO_QKV },
    { "self_attn.to_out",  ARIA_LORA_SA_TO_OUT },
    { "cross_attn.to_kv",  ARIA_LORA_CA_TO_KV  },
    { "cross_attn.to_out", ARIA_LORA_CA_TO_OUT },
    { "cross_attn.to_q",   ARIA_LORA_CA_TO_Q   },
    { "ff.ff.0.proj",      ARIA_LORA_FF_IN     },
    { "ff.ff.2",           ARIA_LORA_FF_OUT    },
};

int aria_lora_match_name(const char *name, int *layer, aria_lora_proj *proj) {
    const char *lp = strstr(name, "layers.");
    if (!lp) return 0;
    int lyr = atoi(lp + 7);   /* digits right after "layers." */
    for (size_t i = 0; i < sizeof(LORA_PROJMAP) / sizeof(*LORA_PROJMAP); i++)
        if (strstr(name, LORA_PROJMAP[i].sub)) {
            if (layer) *layer = lyr;
            if (proj)  *proj = LORA_PROJMAP[i].proj;
            return 1;
        }
    return 0;
}

const aria_lora_item *aria_lora_find(const aria_lora_adapter *a, int layer, aria_lora_proj proj) {
    if (!a) return NULL;
    for (int i = 0; i < a->n; i++)
        if (a->items[i].layer == layer && a->items[i].proj == proj) return &a->items[i];
    return NULL;
}

void aria_lora_linear(float *y, const float *x, int M, int in_dim, int out_dim,
                      int rank, float scale, const float *down, const float *up,
                      float *scratch) {
    if (rank <= 0 || scale == 0.0f) return;
    float *t = scratch ? scratch : malloc((size_t)M * rank * sizeof(float));
    if (!t) return;
    aria_linear(t, x, down, NULL, M, in_dim, rank);   /* t[M,rank] = x @ downT */
    for (size_t i = 0; i < (size_t)M * rank; i++) t[i] *= scale;   /* fold in alpha/rank */
    /* y[M,out] += t[M,rank] @ upT (up is [out,rank]); rank tiny, so a plain loop is
     * far cheaper than the base GEMM and needs no [M*out] temp. */
    for (int m = 0; m < M; m++) {
        const float *tm = t + (size_t)m * rank;
        float *ym = y + (size_t)m * out_dim;
        for (int o = 0; o < out_dim; o++) {
            const float *uo = up + (size_t)o * rank;
            float acc = 0.0f;
            for (int r = 0; r < rank; r++) acc += tm[r] * uo[r];
            ym[o] += acc;
        }
    }
    if (!scratch) free(t);
}

/* ---- adapter loader ---- */

/* Tolerant scan of the safetensors header for a numeric metadata field. Handles
 * the escaped embedded JSON stable-audio-tools writes ("lora_config":"{\"alpha\":
 * 8, ...}") as well as a plain "alpha": 8, by skipping any \ " and space between
 * the key and its ':'. Returns 0 on success. */
static int lora_meta_num(const char *json, const char *key, double *out) {
    if (!json) return -1;
    size_t klen = strlen(key);
    const char *p = strstr(json, key);
    while (p) {
        const char *q = p + klen;
        while (*q == '\\' || *q == '"' || *q == ' ') q++;
        if (*q == ':') {
            q++;
            char *e = NULL;
            double d = strtod(q, &e);
            if (e != q) { *out = d; return 0; }
        }
        p = strstr(p + 1, key);
    }
    return -1;
}

/* Return the item for (layer,proj), creating it (append, growing *cap) if absent. */
static aria_lora_item *lora_item_slot(aria_lora_adapter *a, int *cap, int layer, aria_lora_proj proj) {
    for (int i = 0; i < a->n; i++)
        if (a->items[i].layer == layer && a->items[i].proj == proj) return &a->items[i];
    if (a->n == *cap) {
        int nc = *cap ? *cap * 2 : 16;
        aria_lora_item *g = realloc(a->items, (size_t)nc * sizeof(*g));
        if (!g) return NULL;
        a->items = g; *cap = nc;
    }
    aria_lora_item *it = &a->items[a->n++];
    memset(it, 0, sizeof(*it));
    it->layer = layer; it->proj = proj;
    return it;
}

aria_lora_adapter *aria_lora_load(const char *path, float alpha_override) {
    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) return NULL;   /* safetensors_open already printed the reason */

    double alpha = 0.0; int have_alpha = 0;
    if (alpha_override > 0.0f) { alpha = alpha_override; have_alpha = 1; }
    else if (lora_meta_num(sf->header_json, "alpha", &alpha) == 0 && alpha > 0.0) have_alpha = 1;

    aria_lora_adapter *a = calloc(1, sizeof(*a));
    if (!a) { safetensors_close(sf); return NULL; }
    int cap = 0, skipped = 0;

    for (int i = 0; i < sf->num_tensors; i++) {
        const safetensor_t *t = &sf->tensors[i];
        int is_down = strstr(t->name, "lora_A") != NULL;
        int is_up   = strstr(t->name, "lora_B") != NULL;
        if (!is_down && !is_up) continue;   /* magnitude / M_xs / non-LoRA tensor */
        if (t->ndim != 2) { skipped++; continue; }
        int layer; aria_lora_proj proj;
        if (!aria_lora_match_name(t->name, &layer, &proj)) { skipped++; continue; }
        aria_lora_item *it = lora_item_slot(a, &cap, layer, proj);
        if (!it) break;
        float *data = safetensors_get_f32(sf, t);   /* copies + converts F16/BF16 */
        if (!data) { skipped++; continue; }
        if (is_down) { free((void *)it->down); it->down = data; it->rank = (int)t->shape[0]; it->in_dim = (int)t->shape[1]; }
        else         { free((void *)it->up);   it->up   = data; it->out_dim = (int)t->shape[0]; it->rank = (int)t->shape[1]; }
    }
    safetensors_close(sf);

    /* drop unpaired entries, set scale = alpha/rank (or 1.0 when alpha unknown) */
    int n = 0;
    for (int i = 0; i < a->n; i++) {
        aria_lora_item *it = &a->items[i];
        if (!it->down || !it->up || it->rank <= 0) {
            free((void *)it->down); free((void *)it->up); skipped++; continue;
        }
        it->scale = have_alpha ? (float)(alpha / it->rank) : 1.0f;
        a->items[n++] = *it;
    }
    a->n = n;

    if (a->n == 0) {
        fprintf(stderr, "aria_lora_load: no DiT block-GEMM adapters in %s\n", path);
        aria_lora_free(a);
        return NULL;
    }
    fprintf(stderr, "[lora] %s: %d block GEMMs, rank=%d, scale=%.4g%s%s (%d tensor(s) skipped)\n",
            path, a->n, a->items[0].rank, a->items[0].scale,
            have_alpha ? "" : " (alpha unknown -> 1.0)",
            alpha_override > 0.0f ? " [alpha override]" : "", skipped);
    return a;
}

void aria_lora_free(aria_lora_adapter *a) {
    if (!a) return;
    for (int i = 0; i < a->n; i++) { free((void *)a->items[i].down); free((void *)a->items[i].up); }
    free(a->items);
    free(a);
}
