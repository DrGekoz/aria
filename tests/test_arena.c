/* test_arena.c - hermetic: bump allocation, alignment, save/restore, overflow. */
#include "../src/aria_arena.h"
#include <stdio.h>

int main(void) {
    aria_arena a;
    if (aria_arena_init(&a, 1u << 20) != 0) { printf("FAIL init\n"); return 1; }
    int fails = 0;

    float *p = aria_arena_floats(&a, 10);
    if (((size_t)p & 63) != 0) { printf("FAIL: not 64-aligned\n"); fails++; }

    size_t mark = aria_arena_save(&a);
    float *q = aria_arena_floats(&a, 100);
    if (((size_t)q & 63) != 0) { printf("FAIL: q not aligned\n"); fails++; }
    if (q <= p) { printf("FAIL: q not bumped past p\n"); fails++; }

    aria_arena_restore(&a, mark);
    float *r = aria_arena_floats(&a, 50);
    if (r != q) { printf("FAIL: restore did not reuse the freed region\n"); fails++; }

    if (aria_arena_alloc(&a, (size_t)2u << 20) != NULL) { printf("FAIL: overflow not rejected\n"); fails++; }

    aria_arena_free(&a);
    if (fails) { printf("test_arena: %d failures\n", fails); return 1; }
    printf("test_arena: OK\n");
    return 0;
}
