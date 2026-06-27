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

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aria_sa3_dec aria_sa3_dec;

aria_sa3_dec *aria_sa3_dec_load(safetensors_file_t *sf);
void aria_sa3_dec_free(aria_sa3_dec *m);

/* ---- MEDIUM model decoder (taae_v2 dim 1536/depth 12, sliding-window + sin FF) ----
 * Separate path because medium's resampling differs structurally from small-music's
 * (full-sequence banded attention, no chunk halo). See aria_sa3_dec_medium.c. */
typedef struct aria_sa3_dec_medium aria_sa3_dec_medium;
aria_sa3_dec_medium *aria_sa3_dec_medium_load(safetensors_file_t *sf);
void aria_sa3_dec_medium_free(aria_sa3_dec_medium *m);
void aria_sa3_dec_medium_forward(const aria_sa3_dec_medium *m, float *audio, const float *latent, int T);
/* (medium view + CUDA medium decoder declared after aria_taae_block_view, below) */

/* full decode: latent[256,T] -> out_audio[2, T*4096] (caller allocates). */
void aria_sa3_dec_forward(const aria_sa3_dec *m, float *out_audio, const float *latent, int T);

/* ---- stage entry points (staged parity) ---- */
void aria_sa3_softnorm_decode(const aria_sa3_dec *m, float *z, const float *latent, int T); /* z[256,T] */
void aria_sa3_same_decode(const aria_sa3_dec *m, float *dec, const float *z, int T);        /* dec[512,T*16] */
void aria_unpatch_stereo(float *audio, const float *dec, int L);                            /* audio[2,L*256] */

/* test hook: run transformers[idx] on xc[N,768] in place (rope recomputed for N). */
void aria_sa3_dec_block_test(const aria_sa3_dec *m, int idx, float *xc, int N);

/* ---- read-only weight view (so the CUDA backend can upload) ---- */
typedef struct {
    float pre_alpha, qn_alpha, kn_alpha, ff_alpha;
    const float *pre_gamma, *pre_beta, *qn_gamma, *qn_beta, *kn_gamma, *kn_beta, *ff_gamma, *ff_beta;
    const float *to_qkv, *to_out, *ff_in_w, *ff_in_b, *ff_out_w, *ff_out_b;
} aria_taae_block_view;
typedef struct {
    float running_std;
    const float *proj_w, *proj_b, *new_tokens, *mapping_w, *mapping_b;
    aria_taae_block_view blocks[6];
} aria_sa3_dec_view;
void aria_sa3_dec_get_view(const aria_sa3_dec *m, aria_sa3_dec_view *v);

/* read-only view + CUDA device-resident medium decoder (GPU). Blocks where
 * (12-i) < 8 use a sinusoidal FF (the runtime decides from the block index). */
typedef struct {
    float running_std;
    const float *proj_w, *proj_b, *new_tokens, *mapping_w, *mapping_b;
    aria_taae_block_view blocks[12];
} aria_sa3_dec_medium_view;
void aria_sa3_dec_medium_get_view(const aria_sa3_dec_medium *m, aria_sa3_dec_medium_view *v);

typedef struct aria_cuda_dec_medium aria_cuda_dec_medium;
aria_cuda_dec_medium *aria_cuda_dec_medium_create(const aria_sa3_dec_medium_view *v);   /* NULL if no device/fit */
void aria_cuda_dec_medium_free(aria_cuda_dec_medium *h);
void aria_cuda_dec_medium_forward(aria_cuda_dec_medium *h, float *audio, const float *latent, int T);

/* ---- CUDA device-resident decoder (E8.5c; implemented in aria_cuda.cu) ----
 * Uploads weights (fp16 GEMMs, fp32 norms/conv) once; runs the whole latent->audio
 * decode on the device, batching all S=34 chunks of a pass. All host pointers. */
typedef struct aria_cuda_dec aria_cuda_dec;
aria_cuda_dec *aria_cuda_dec_create(const aria_sa3_dec_view *v);   /* NULL if no device / no fit */
void aria_cuda_dec_free(aria_cuda_dec *h);
void aria_cuda_dec_forward(aria_cuda_dec *h, float *audio, const float *latent, int T); /* audio[2,T*4096] */

#ifdef __cplusplus
}
#endif

#endif /* ARIA_SA3_DEC_H */
