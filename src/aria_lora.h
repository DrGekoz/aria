/*
 * aria_lora.h - LoRA runtime-adapter arm for the SA3 DiT (E12.9, CPU-first).
 *
 * A LoRA adapter is a set of low-rank residuals applied on top of the frozen base
 * weights at runtime. For a hooked linear y = x W^T, the adapter adds
 *   y += scale * (x downT) upT
 * with down [rank,in] (PEFT lora_A), up [out,rank] (PEFT lora_B), scale = alpha/rank.
 * The base weight W is never touched -- it stays quantized / mmap'd -- so this
 * composes with the Q8/Q4 overlay and costs only two tiny GEMMs per hooked site
 * (the same "keep the base captured" trick the steering arm uses for the CUDA graph).
 * Standard LoRA only (LoHa/LoKr/DoRA magnitude tensors are ignored).
 *
 * -- Name mapping (loader) --------------------------------------------------
 * The loader maps each adapter tensor onto one of the SA3 DiT block GEMMs by the
 * layer index parsed from the "layers.<i>." segment plus a projection substring:
 *
 *   accepted projection substring          aria block GEMM
 *   -----------------------------          ---------------
 *   "self_attn.to_qkv"                      sa_to_qkv
 *   "self_attn.to_out"                      sa_to_out
 *   "cross_attn.to_q"   (not to_kv)         ca_to_q
 *   "cross_attn.to_kv"                      ca_to_kv
 *   "cross_attn.to_out"                     ca_to_out
 *   "ff.ff.0.proj"                          ff_in   (GLU input, [2*inner,ed])
 *   "ff.ff.2"                               ff_out  (GLU output, [ed,inner])
 *
 * These substrings are exactly aria's own DiT tensor names (aria_sa3_dit_load), so
 * the map is prefix-agnostic. PEFT tensor-name conventions, both accepted:
 *   <module>.parametrizations.weight.0.lora_A   [rank,in]   (down)   <- stable-audio-tools
 *   <module>.parametrizations.weight.0.lora_B   [out,rank]  (up)        export_lora_safetensors
 *   <module>.lora_A.weight / <module>.lora_B.weight                  <- plain PEFT
 * e.g. model.transformer.layers.7.self_attn.to_qkv.parametrizations.weight.0.lora_A.
 * Adapters on layers aria does not hook (project_in/out, embedders,
 * to_scale_shift_gate, conv1d) are counted and skipped -- only the 7 GEMMs apply.
 *
 * scale = alpha/rank. alpha comes from aria_lora_load's alpha_override (>0 wins),
 * else the file's embedded "lora_config" metadata; rank is read from tensor shape.
 * When no alpha is available scale defaults to 1.0 (== alpha==rank).
 *
 * E12.9b (OUT of scope here): the GPU LoRA path (device-resident scale folded into
 * the CUDA graph) is not implemented; --lora runs the DiT on the CPU backend.
 */

#ifndef ARIA_LORA_H
#define ARIA_LORA_H

#ifdef __cplusplus
extern "C" {
#endif

/* the 7 hooked DiT block GEMMs, in the order the block forward computes them. */
typedef enum {
    ARIA_LORA_SA_TO_QKV = 0, ARIA_LORA_SA_TO_OUT,
    ARIA_LORA_CA_TO_Q, ARIA_LORA_CA_TO_KV, ARIA_LORA_CA_TO_OUT,
    ARIA_LORA_FF_IN, ARIA_LORA_FF_OUT, ARIA_LORA_NPROJ
} aria_lora_proj;

typedef struct {
    int   layer;          /* DiT block index */
    aria_lora_proj proj;  /* which block GEMM this adapter patches */
    int   rank, in_dim, out_dim;
    float scale;          /* alpha/rank (or override/rank); 0 = no-op */
    const float *down;    /* [rank, in_dim]  (lora_A) */
    const float *up;      /* [out_dim, rank] (lora_B) */
} aria_lora_item;

/* Public (the hermetic test builds one on the stack). aria_lora_load allocates a
 * heap adapter that owns every items[i].down/up buffer; aria_lora_free releases it.
 * Do NOT call aria_lora_free on a hand-built adapter whose items point at borrowed
 * memory. */
typedef struct {
    int n;
    aria_lora_item *items;   /* [n] */
} aria_lora_adapter;

/* Load a PEFT/stable-audio-tools LoRA safetensors adapter (F32/F16/BF16), mapping
 * its tensors onto the DiT block GEMMs above. alpha_override>0 forces alpha; <=0
 * reads it from the file metadata (else scale defaults to 1.0). Returns NULL on a
 * missing/unreadable file or when no block-GEMM adapter is found. */
aria_lora_adapter *aria_lora_load(const char *path, float alpha_override);
void aria_lora_free(aria_lora_adapter *a);

/* Find the adapter entry for (layer, proj), or NULL. Linear scan (n is small). */
const aria_lora_item *aria_lora_find(const aria_lora_adapter *a, int layer, aria_lora_proj proj);

/* Map an adapter tensor name to (layer, proj). Returns 1 on a match (fills *layer
 * and *proj, which may be NULL), 0 otherwise. */
int aria_lora_match_name(const char *name, int *layer, aria_lora_proj *proj);

/* Low-rank residual on top of an already-computed base GEMM output:
 *   y[M,out_dim] += scale * (x[M,in_dim] @ downT) @ upT
 * down [rank,in_dim], up [out_dim,rank]. scratch is [M*rank] floats, or NULL to
 * malloc internally. No-op when rank<=0 or scale==0 (y left bit-identical). */
void aria_lora_linear(float *y, const float *x, int M, int in_dim, int out_dim,
                      int rank, float scale, const float *down, const float *up,
                      float *scratch);

#ifdef __cplusplus
}
#endif

#endif /* ARIA_LORA_H */
