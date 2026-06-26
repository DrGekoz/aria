/*
 * aria_arena.h - bump-pointer scratch arena for the forward path.
 *
 * One arena per request, allocated once and reused across all denoising steps,
 * so the hot path (DiT block x20 x steps) stops malloc/free-ing and re-faulting
 * large temporaries. Stack-discipline: aria_arena_save() before a scope, allocate
 * with aria_arena_alloc(), aria_arena_restore() on exit to free everything at once.
 * Capacity must be sized up front (pointers are never relocated).
 */

#ifndef ARIA_ARENA_H
#define ARIA_ARENA_H

#include <stddef.h>

typedef struct {
    char  *base;
    size_t cap;
    size_t used;
    size_t peak;   /* high-water mark (for sizing/diagnostics) */
} aria_arena;

int   aria_arena_init(aria_arena *a, size_t cap);   /* 0 on success */
void  aria_arena_free(aria_arena *a);

/* 64-byte-aligned bump allocation. Returns NULL if it would exceed cap. */
void *aria_arena_alloc(aria_arena *a, size_t bytes);
/* convenience: n floats */
float *aria_arena_floats(aria_arena *a, size_t n);

size_t aria_arena_save(const aria_arena *a);          /* current offset */
void   aria_arena_restore(aria_arena *a, size_t mark); /* free back to offset */

#endif /* ARIA_ARENA_H */
