/*
 * aria_json.c - Tiny read-only JSON value extractor.
 */

#include "aria_json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *aria_read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return NULL; }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return NULL; }
    buf[sz] = '\0';
    fclose(f);
    if (out_len) *out_len = (size_t)sz;
    return buf;
}

/* Locate the position just after the colon following "key". Returns NULL if
 * not found. Naive first-match scan -- adequate for flat config fields. */
static const char *find_key(const char *json, const char *key) {
    size_t klen = strlen(key);
    const char *p = json;
    while ((p = strchr(p, '"')) != NULL) {
        const char *start = p + 1;
        if (strncmp(start, key, klen) == 0 && start[klen] == '"') {
            const char *q = start + klen + 1;
            while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
            if (*q == ':') {
                q++;
                while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
                return q;
            }
        }
        p = start;
    }
    return NULL;
}

int aria_json_get_string(const char *json, const char *key, char *out, size_t outlen) {
    const char *v = find_key(json, key);
    if (!v || *v != '"') return -1;
    v++;
    size_t i = 0;
    while (*v && *v != '"' && i < outlen - 1) {
        if (*v == '\\' && v[1]) v++;
        out[i++] = *v++;
    }
    out[i] = '\0';
    return 0;
}

int aria_json_get_number(const char *json, const char *key, double *out) {
    const char *v = find_key(json, key);
    if (!v) return -1;
    char *end = NULL;
    double d = strtod(v, &end);
    if (end == v) return -1;
    *out = d;
    return 0;
}
