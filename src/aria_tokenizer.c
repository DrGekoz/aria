/*
 * aria_tokenizer.c - GemmaTokenizer BPE. See aria_tokenizer.h.
 */

#include "aria_tokenizer.h"
#include "aria_json.h"   /* aria_read_file */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

/* ---- (ptr,len) string -> int open-addressing hash map ---- */
typedef struct { const char *key; int klen; int val; } hentry;
typedef struct { hentry *e; int cap; } hmap;

static uint64_t fnv1a(const char *s, int n) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ULL; }
    return h;
}
static int hmap_init(hmap *m, int n_entries) {
    int cap = 16;
    while (cap < n_entries * 2) cap <<= 1;
    m->e = calloc((size_t)cap, sizeof(hentry));
    if (!m->e) return -1;
    m->cap = cap;
    for (int i = 0; i < cap; i++) m->e[i].klen = -1;
    return 0;
}
static void hmap_put(hmap *m, const char *key, int klen, int val) {
    uint64_t i = fnv1a(key, klen) & (uint64_t)(m->cap - 1);
    while (m->e[i].klen != -1) {
        if (m->e[i].klen == klen && memcmp(m->e[i].key, key, klen) == 0) { m->e[i].val = val; return; }
        i = (i + 1) & (uint64_t)(m->cap - 1);
    }
    m->e[i].key = key; m->e[i].klen = klen; m->e[i].val = val;
}
static int hmap_get(const hmap *m, const char *key, int klen) {
    uint64_t i = fnv1a(key, klen) & (uint64_t)(m->cap - 1);
    while (m->e[i].klen != -1) {
        if (m->e[i].klen == klen && memcmp(m->e[i].key, key, klen) == 0) return m->e[i].val;
        i = (i + 1) & (uint64_t)(m->cap - 1);
    }
    return -1;
}

struct aria_tokenizer {
    char *blob;       /* the loaded file; hash keys point into it */
    size_t blob_size;
    hmap vocab;       /* token bytes -> id */
    hmap merges;      /* "left right" -> rank */
};

static uint32_t rd_u32(const char *p) {
    return (uint32_t)(unsigned char)p[0] | ((uint32_t)(unsigned char)p[1] << 8) |
           ((uint32_t)(unsigned char)p[2] << 16) | ((uint32_t)(unsigned char)p[3] << 24);
}

aria_tokenizer *aria_tokenizer_load(const char *path) {
    size_t sz = 0;
    char *blob = aria_read_file(path, &sz);
    if (!blob) return NULL;
    if (sz < 12 || memcmp(blob, "ATOK", 4) != 0) { free(blob); return NULL; }

    aria_tokenizer *t = calloc(1, sizeof(*t));
    if (!t) { free(blob); return NULL; }
    t->blob = blob; t->blob_size = sz;

    size_t pos = 8;  /* after magic + version */
    uint32_t n_vocab = rd_u32(blob + pos); pos += 4;
    if (hmap_init(&t->vocab, (int)n_vocab) != 0) { aria_tokenizer_free(t); return NULL; }
    for (uint32_t i = 0; i < n_vocab; i++) {
        uint32_t id = rd_u32(blob + pos); pos += 4;
        uint32_t len = rd_u32(blob + pos); pos += 4;
        hmap_put(&t->vocab, blob + pos, (int)len, (int)id);
        pos += len;
    }
    uint32_t n_merges = rd_u32(blob + pos); pos += 4;
    if (hmap_init(&t->merges, (int)n_merges) != 0) { aria_tokenizer_free(t); return NULL; }
    for (uint32_t i = 0; i < n_merges; i++) {
        uint32_t len = rd_u32(blob + pos); pos += 4;
        hmap_put(&t->merges, blob + pos, (int)len, (int)i);  /* rank = order */
        pos += len;
    }
    return t;
}

void aria_tokenizer_free(aria_tokenizer *t) {
    if (!t) return;
    free(t->vocab.e); free(t->merges.e); free(t->blob); free(t);
}

/* number of bytes in the UTF-8 code point starting at lead byte c */
static int cp_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c >> 5) == 0x6) return 2;
    if ((c >> 4) == 0xE) return 3;
    if ((c >> 3) == 0x1E) return 4;
    return 1;  /* malformed -> treat as single byte */
}

int aria_tokenizer_encode(const aria_tokenizer *t, const char *text, int *ids, int seq) {
    /* normalize: replace ASCII space with U+2581 (E2 96 81) */
    size_t tlen = strlen(text);
    char *norm = malloc(tlen * 3 + 1);
    if (!norm) return -1;
    size_t nl = 0;
    for (size_t i = 0; i < tlen; i++) {
        if (text[i] == ' ') { norm[nl++] = (char)0xE2; norm[nl++] = (char)0x96; norm[nl++] = (char)0x81; }
        else norm[nl++] = text[i];
    }

    /* split into code-point spans */
    int *off = malloc((nl + 1) * sizeof(int));
    int *len = malloc((nl + 1) * sizeof(int));
    int count = 0;
    for (size_t i = 0; i < nl; ) {
        int l = cp_len((unsigned char)norm[i]);
        if (i + (size_t)l > nl) l = (int)(nl - i);
        off[count] = (int)i; len[count] = l; count++;
        i += l;
    }

    /* BPE: repeatedly merge the adjacent pair with the smallest rank */
    char keybuf[1024];
    for (;;) {
        int best_rank = INT_MAX, best_i = -1;
        for (int i = 0; i < count - 1; i++) {
            int kl = len[i] + 1 + len[i + 1];
            if (kl > (int)sizeof(keybuf)) continue;
            memcpy(keybuf, norm + off[i], len[i]);
            keybuf[len[i]] = ' ';
            memcpy(keybuf + len[i] + 1, norm + off[i + 1], len[i + 1]);
            int r = hmap_get(&t->merges, keybuf, kl);
            if (r >= 0 && r < best_rank) { best_rank = r; best_i = i; }
        }
        if (best_i < 0) break;
        len[best_i] += len[best_i + 1];           /* spans are contiguous */
        for (int j = best_i + 1; j < count - 1; j++) { off[j] = off[j + 1]; len[j] = len[j + 1]; }
        count--;
    }

    /* symbols -> ids, byte fallback */
    int n = 0;
    for (int i = 0; i < count && n < seq; i++) {
        int id = hmap_get(&t->vocab, norm + off[i], len[i]);
        if (id >= 0) {
            ids[n++] = id;
        } else {
            for (int j = 0; j < len[i] && n < seq; j++) {
                unsigned char b = (unsigned char)norm[off[i] + j];
                char bt[8];
                int bl = snprintf(bt, sizeof(bt), "<0x%02X>", b);
                int bid = hmap_get(&t->vocab, bt, bl);
                ids[n++] = (bid >= 0) ? bid : 3;  /* UNK */
            }
        }
    }
    int n_real = n;
    for (; n < seq; n++) ids[n] = 0;  /* pad */

    free(norm); free(off); free(len);
    return n_real;
}
