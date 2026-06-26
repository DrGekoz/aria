/* test_tokenizer.c - parity: C BPE token ids == HF GemmaTokenizer ids.
 * Reuses the token-id dumps from the encoder reference (t5enc/ids_*.atns).
 * Needs ARIA_TOKENIZER (the exported .bin) + ARIA_DUMPS. */
#include "../src/aria_tokenizer.h"
#include "../src/aria_parity.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* keep in sync with dump_phase1.dump_t5enc prompts */
static const char *PROMPTS[3] = {
    "warm romantic piano",
    "Amen break 174 BPM",
    "lofi house loop",
};

int main(void) {
    const char *DUMPS = getenv("ARIA_DUMPS");
    const char *tokpath = getenv("ARIA_TOKENIZER");
    if (!DUMPS || !tokpath) { printf("test_tokenizer: SKIP (set ARIA_TOKENIZER and ARIA_DUMPS)\n"); return 0; }

    aria_tokenizer *t = aria_tokenizer_load(tokpath);
    if (!t) { printf("FAIL: cannot load tokenizer %s\n", tokpath); return 1; }

    int fails = 0;
    for (int idx = 0; idx < 3; idx++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/t5enc/ids_%d.atns", DUMPS, idx);
        aria_parity_tensor ref;
        if (aria_parity_load(path, &ref) != 0) { printf("FAIL: load %s\n", path); fails++; continue; }
        int seq = (int)ref.numel;

        int *ids = malloc((size_t)seq * sizeof(int));
        int n = aria_tokenizer_encode(t, PROMPTS[idx], ids, seq);

        int mismatch = 0, first_bad = -1;
        for (int i = 0; i < seq; i++) {
            int want = (int)(ref.data[i] + 0.5f);
            if (ids[i] != want) { mismatch++; if (first_bad < 0) first_bad = i; }
        }
        if (mismatch) {
            printf("FAIL tok[%d] \"%s\": %d/%d mismatch (first at %d: got %d want %d)\n",
                   idx, PROMPTS[idx], mismatch, seq, first_bad, ids[first_bad], (int)(ref.data[first_bad] + 0.5f));
            fails++;
        } else {
            printf("ok   tok[%d] \"%s\": n_real=%d, all %d ids match\n", idx, PROMPTS[idx], n, seq);
        }
        free(ids);
        aria_parity_free(&ref);
    }

    aria_tokenizer_free(t);
    if (fails) { printf("test_tokenizer: FAILED\n"); return 1; }
    printf("test_tokenizer: OK\n");
    return 0;
}
