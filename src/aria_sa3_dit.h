/*
 * aria_sa3_dit.h - Stable Audio 3 DiT: conditioning embedders + transformer block.
 *
 * Reference: stable_audio_tools models/dit.py (DiffusionTransformer) and
 * models/transformer.py (TransformerBlock, ContinuousTransformer).
 *
 * Block forward (adaLN), batch=1, operating on x[S, dim]:
 *   mod = to_scale_shift_gate[6*dim] + global_cond[6*dim]; chunk into
 *     (scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff) each [dim]
 *   self-attn: r=x; h=rmsnorm(x); h=h*(1+scale_self)+shift_self;
 *              h=selfattn(h, qk-rms, rope); h*=sigmoid(1-gate_self); x=r+h
 *   cross-attn: r=x; h=rmsnorm(x); h=crossattn(h, context, qk-rms, NO rope); x=r+h
 *   ffn:       r=x; h=rmsnorm(x); h=h*(1+scale_ff)+shift_ff;
 *              h=glu_ff(h); h*=sigmoid(1-gate_ff); x=r+h
 */

#ifndef ARIA_SA3_DIT_H
#define ARIA_SA3_DIT_H

#include "aria_safetensors.h"
#include "aria_sa3.h"

/* Two-layer MLP: Linear(in_dim->mid, W0,b0) -> SiLU -> Linear(mid->out_dim, W2,b2),
 * applied to N rows. b0/b2 may be NULL (no bias). out [N, out_dim]. */
void aria_sa3_mlp2(float *out, const float *in, int N, int in_dim, int mid, int out_dim,
                   const float *W0, const float *b0, const float *W2, const float *b2);

/* Timestep embedding: ExpoFourier(t, feat_dim, 0.5, 10000) -> mlp2(feat->ed->ed).
 * out [ed]. (t is the raw diffusion timestep.) */
void aria_sa3_timestep_embed(float *out, float t, int feat_dim, int ed,
                             const float *W0, const float *b0, const float *W2, const float *b2);

typedef struct {
    const float *pre_norm, *cross_norm, *ff_norm;   /* RMSNorm gammas [dim] */
    const float *sa_to_qkv, *sa_q_norm, *sa_k_norm, *sa_to_out;
    const float *ca_to_q, *ca_to_kv, *ca_q_norm, *ca_k_norm, *ca_to_out;
    const float *ff_in_w, *ff_in_b, *ff_out_w, *ff_out_b;
    const float *to_scale_shift_gate;               /* [6*dim] */
} aria_dit_block_w;

/* In-place adaLN block forward on x[S, dim].
 *   context [Sc, dim_ctx] (cross-attn kv), global_cond [6*dim] (post embedder),
 *   rope_cos/rope_sin [S, rot_dim/2] for self-attn (cross-attn has no rope).
 *   inner = ff inner dim (dim * ff_mult). */
void aria_dit_block_forward(float *x, int S, int dim, int num_heads, int head_dim, int inner,
                            const float *context, int Sc, int dim_ctx,
                            const float *global_cond,
                            const float *rope_cos, const float *rope_sin, int rot_dim,
                            const aria_dit_block_w *w);

/* ---- full DiT model ---- */

typedef struct aria_sa3_dit aria_sa3_dit;

/* Load all DiT weights (model.model.*) from sf using cfg. Returns NULL on a
 * missing tensor. Weights are copied to f32 buffers owned by the model. */
aria_sa3_dit *aria_sa3_dit_load(safetensors_file_t *sf, const aria_sa3_config *cfg);
void aria_sa3_dit_free(aria_sa3_dit *m);

/* Denoiser forward (matches DiffusionTransformer._forward, batch=1).
 *   x_CT, out_CT: latent [io_channels, T] (channel-major).
 *   t: diffusion timestep scalar.
 *   cross_768: [n_cond, cond_token_dim] raw cross-attn cond (pre to_cond_embed).
 *   global_768: [global_cond_dim] raw global cond (seconds, pre to_global_embed).
 * Returns velocity in out_CT. */
void aria_sa3_dit_forward(const aria_sa3_dit *m, float *out_CT, const float *x_CT, int T,
                          float t, const float *cross_768, int n_cond, const float *global_768);

#endif /* ARIA_SA3_DIT_H */
