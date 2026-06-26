/*
 * aria_tokenizer.h - GemmaTokenizer (BPE, byte-fallback) for the T5Gemma encoder.
 *
 * Loads the compact vocab+merges binary produced by scripts/export_tokenizer.py.
 * Algorithm (verified parity): normalize (space -> U+2581), split into Unicode
 * code points, rank-ordered BPE merge of adjacent pairs, then symbol -> id with
 * byte fallback (<0xNN>). No BOS/EOS; right-pad with id 0; UNK = 3.
 */

#ifndef ARIA_TOKENIZER_H
#define ARIA_TOKENIZER_H

typedef struct aria_tokenizer aria_tokenizer;

aria_tokenizer *aria_tokenizer_load(const char *path);
void aria_tokenizer_free(aria_tokenizer *t);

/* Encode text -> ids[seq] (right-padded with 0). Returns the number of real
 * (non-pad) tokens, or -1 on error. */
int aria_tokenizer_encode(const aria_tokenizer *t, const char *text, int *ids, int seq);

#endif /* ARIA_TOKENIZER_H */
