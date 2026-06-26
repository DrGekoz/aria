/*
 * aria_parity.h - Reader for the .atns tensor-dump format used to compare C
 * outputs against PyTorch references.
 *
 * .atns layout (little-endian):
 *   "ATNS" (4) | u32 version=1 | u32 ndim | u32 dtype(0=f32) |
 *   i64 shape[ndim] | raw f32 data
 */

#ifndef ARIA_PARITY_H
#define ARIA_PARITY_H

#include <stdint.h>

typedef struct {
    int ndim;
    int64_t shape[8];
    int64_t numel;
    float *data;     /* heap; free with aria_parity_free */
} aria_parity_tensor;

int  aria_parity_load(const char *path, aria_parity_tensor *out);  /* 0 on success */
void aria_parity_free(aria_parity_tensor *t);

/* max |a-b| over n elements */
float aria_parity_maxabsdiff(const float *a, const float *b, int64_t n);

#endif /* ARIA_PARITY_H */
