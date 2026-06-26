/*
 * aria_sa3_dec.h - Stable Audio 3 taae_v2 autoencoder DECODER (latent -> audio).
 *
 * Forward (latent[256,T] -> audio[2, T*4096]):
 *   1. softnorm inverse:  z = latent * running_std
 *   2-3. transpose + latent->768 proj (decoder.layers.1)
 *   4-13. TransformerResamplingBlock (decoder.layers.3): pad T even, insert 16
 *         learned new_tokens per latent token, two chunked (S=34) transformer
 *         halves (blocks 0-2 unshifted, 3-5 midpoint-shift halo), extract last
 *         16 of each group, WNConv1d mapping 768->512, trim
 *   14. unpatch 512 -> stereo (patch_size 256)
 * Per-block: DyT norms, differential self-attention (o_base - o_diff), SwiGLU FF.
 */

#ifndef ARIA_SA3_DEC_H
#define ARIA_SA3_DEC_H

#include "aria_safetensors.h"

typedef struct aria_sa3_dec aria_sa3_dec;

aria_sa3_dec *aria_sa3_dec_load(safetensors_file_t *sf);
void aria_sa3_dec_free(aria_sa3_dec *m);

/* full decode: latent[256,T] -> out_audio[2, T*4096] (caller allocates). */
void aria_sa3_dec_forward(const aria_sa3_dec *m, float *out_audio, const float *latent, int T);

/* ---- stage entry points (staged parity) ---- */
void aria_sa3_softnorm_decode(const aria_sa3_dec *m, float *z, const float *latent, int T); /* z[256,T] */
void aria_sa3_same_decode(const aria_sa3_dec *m, float *dec, const float *z, int T);        /* dec[512,T*16] */
void aria_unpatch_stereo(float *audio, const float *dec, int L);                            /* audio[2,L*256] */

/* test hook: run transformers[idx] on xc[N,768] in place (rope recomputed for N). */
void aria_sa3_dec_block_test(const aria_sa3_dec *m, int idx, float *xc, int N);

#endif /* ARIA_SA3_DEC_H */
