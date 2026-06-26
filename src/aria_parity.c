/*
 * aria_parity.c - .atns reader for parity tests.
 */

#include "aria_parity.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int aria_parity_load(const char *path, aria_parity_tensor *out) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[4];
    uint32_t version = 0, ndim = 0, dtype = 0;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "ATNS", 4) != 0) { fclose(f); return -1; }
    if (fread(&version, 4, 1, f) != 1 || version != 1) { fclose(f); return -1; }
    if (fread(&ndim, 4, 1, f) != 1 || ndim > 8) { fclose(f); return -1; }
    if (fread(&dtype, 4, 1, f) != 1 || dtype != 0) { fclose(f); return -1; }
    out->ndim = (int)ndim;
    out->numel = 1;
    for (uint32_t i = 0; i < ndim; i++) {
        int64_t s = 0;
        if (fread(&s, 8, 1, f) != 1) { fclose(f); return -1; }
        out->shape[i] = s;
        out->numel *= s;
    }
    out->data = malloc((size_t)out->numel * sizeof(float));
    if (!out->data) { fclose(f); return -1; }
    if (fread(out->data, sizeof(float), (size_t)out->numel, f) != (size_t)out->numel) {
        free(out->data); out->data = NULL; fclose(f); return -1;
    }
    fclose(f);
    return 0;
}

void aria_parity_free(aria_parity_tensor *t) {
    if (t && t->data) { free(t->data); t->data = NULL; }
}

float aria_parity_maxabsdiff(const float *a, const float *b, int64_t n) {
    float m = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}
