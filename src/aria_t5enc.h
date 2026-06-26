/*
 * aria_t5enc.h - T5Gemma text encoder (google/t5gemma-b-b-ul2, encoder only).
 *
 * Maps token ids -> [seq, 768] cross-attention conditioning. Gemma2-style:
 * embed*sqrt(768) -> 12 layers (sandwich RMSNorm with (1+w), MHA with logit
 * softcap 50 + full RoPE, GeGLU MLP) -> final norm -> learned-padding overwrite.
 * Weights are BF16 (dequantized to f32 at load; the 256000-row embedding table
 * is row-gathered lazily). The learned padding vector lives in the main SA3
 * checkpoint (F32).
 */

#ifndef ARIA_T5ENC_H
#define ARIA_T5ENC_H

#include "aria_safetensors.h"

typedef struct aria_t5enc aria_t5enc;

/* Load the encoder from <model_dir>/<subfolder>/model.safetensors and the
 * learned padding embedding (conditioner.conditioners.prompt.padding_embedding)
 * from main_sf. Returns NULL on error. */
aria_t5enc *aria_t5enc_load(const char *model_dir, const char *subfolder,
                            safetensors_file_t *main_sf);
void aria_t5enc_free(aria_t5enc *e);

/* ids[seq] right-padded with 0 (n_real real tokens). Writes cond[seq*768]. */
void aria_t5enc_encode(const aria_t5enc *e, float *cond, const int *ids, int seq, int n_real);

#endif /* ARIA_T5ENC_H */
