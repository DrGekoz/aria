/* test_schedule.c - parity: aria_logsnr_schedule vs build_schedule+LogSNRShift.
 * Needs ARIA_DUMPS. */
#include "../src/aria_sampler.h"
#include "../src/aria_parity.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const char *DUMPS;
static int fails = 0;

static void run(int steps, int seq_len) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/sampler/sched_s%d_L%d.atns", DUMPS, steps, seq_len);
    aria_parity_tensor ref;
    if (aria_parity_load(path, &ref) != 0) { printf("FAIL: load %s\n", path); fails++; return; }
    float *got = malloc((size_t)(steps + 1) * sizeof(float));
    aria_logsnr_schedule(got, steps, 1.0f, -6.2f, 2000.0f, 1.0f, 2.0f, (float)seq_len);
    float md = aria_parity_maxabsdiff(got, ref.data, steps + 1);
    if (md > 1e-5f) { printf("FAIL sched s%d L%d: maxdiff=%.3e\n", steps, seq_len, md); fails++; }
    else printf("ok   sched s%d L%d: maxdiff=%.3e\n", steps, seq_len, md);
    free(got);
    aria_parity_free(&ref);
}

int main(void) {
    DUMPS = getenv("ARIA_DUMPS");
    if (!DUMPS) { printf("test_schedule: SKIP (set ARIA_DUMPS)\n"); return 0; }
    run(8, 323);
    run(8, 1293);
    run(16, 512);
    if (fails) { printf("test_schedule: FAILED\n"); return 1; }
    printf("test_schedule: OK\n");
    return 0;
}
