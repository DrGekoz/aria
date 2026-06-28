/*
 * aria_ops.h - Backend-agnostic primitive op surface
 *
 * Every model module (DiT, taae_v2 autoencoder, T5Gemma encoder) is built
 * from these primitives. CPU implementations live in aria_cpu.c; CUDA
 * implementations (Phase 3) live in aria_cuda.cu behind #ifdef ARIA_CUDA.
 *
 * Convention: CPU ops operate on flat row-major float arrays with explicit
 * dimensions (iris.c style) -- simple to test and to parity-check against
 * PyTorch. Weight tensors follow PyTorch's nn.Linear layout: W is [out, in].
 */

#ifndef ARIA_OPS_H
#define ARIA_OPS_H

/* ---- linear algebra ---- */

/* y[M,N] = x[M,K] @ W^T + b ; W is [N,K] (PyTorch Linear weight), b is [N] or NULL. */
void aria_linear(float *y, const float *x, const float *W, const float *b,
                 int M, int K, int N);

/* C[M,N] = A[M,K] @ B[K,N] (both row-major). */
void aria_matmul(float *C, const float *A, const float *B, int M, int K, int N);

/* ---- normalization ---- */

/* RMSNorm over the last dim. y and x are [rows, dim]. weight is [dim] or NULL.
 * y = x / sqrt(mean(x^2) + eps) * (weight ? weight : 1). Computed in fp32. */
void aria_rmsnorm(float *y, const float *x, const float *weight,
                  int rows, int dim, float eps);

/* DynamicTanh (taae_v2): y = tanh(alpha * x) * weight + bias, per-channel
 * weight/bias of length dim. alpha is a learned scalar. */
void aria_dynamic_tanh(float *y, const float *x, float alpha,
                       const float *weight, const float *bias,
                       int rows, int dim);

/* Gemma RMSNorm: y = x / sqrt(mean(x^2)+eps) * (1 + weight). Note the (1+weight):
 * Gemma stores zero-centered norm gains. Computed in fp32. */
void aria_gemma_rmsnorm(float *y, const float *x, const float *weight,
                        int rows, int dim, float eps);

/* Attention-logit softcapping: s[i] = cap * tanh(s[i] / cap), n elements. */
void aria_softcap(float *s, int n, float cap);

/* ---- activations (in-place over n elements) ---- */
void aria_silu(float *x, int n);          /* x * sigmoid(x) */
void aria_gelu_tanh(float *x, int n);     /* gelu_pytorch_tanh */

/* SiLU-gated FFN combine: out[i] = silu(gate[i]) * up[i], n elements. */
void aria_silu_gate(float *out, const float *gate, const float *up, int n);

/* ---- rotary position embedding (rotate-half, GPT-NeoX style) ---- */

/* Precompute cos/sin tables [n, rot_dim/2] for integer positions 0..n-1.
 * inv_freq[i] = 1 / base^(2i/rot_dim). Matches stable_audio_tools RotaryEmbedding
 * (base 10000, interpolation_factor 1). cos_t/sin_t hold n*(rot_dim/2) floats. */
void aria_rope_freqs(float *cos_t, float *sin_t, int n, int rot_dim, float base);

/* Apply RoPE in place to x laid out [H, N, D] (head-major). Rotates the first
 * rot_dim (<= D) dims with pairing (i, i+rot_dim/2); dims >= rot_dim untouched. */
void aria_rope_apply(float *x, const float *cos_t, const float *sin_t,
                     int H, int N, int D, int rot_dim);

/* ---- attention ---- */

/* Multi-head scaled-dot-product attention, bidirectional (non-causal).
 * q [H,Nq,D], k/v [H,Nk,D], out [H,Nq,D]. scale = 1/sqrt(D) (SDPA default).
 * Optional additive mask [Nq,Nk] (e.g. -inf for padding) or NULL.
 * scratch: [Nq*Nk] floats, or NULL to malloc internally. */
void aria_attention(float *out, const float *q, const float *k, const float *v,
                    int H, int Nq, int Nk, int D, const float *mask, float *scratch);

/* Sliding-window (banded) self-attention: query i attends to keys [i-W, i+W] only.
 * Same result as aria_attention with a [-W,W] band mask, but O(N*(2W+1)*D) instead
 * of O(N^2*D). q/k/v/out [H,N,D]. */
void aria_attention_band(float *out, const float *q, const float *k, const float *v,
                         int H, int N, int D, int W);

/* ---- conv1d (stride 1, pre-folded weights) ---- */

/* out[Cout,L] = conv1d(in[Cin,L]) with weight w[Cout,Cin,K], zero-padded `pad`
 * each side, stride 1 (length preserving when 2*pad == K-1). bias [Cout] or NULL.
 * Weight-normalized convs must have weight_g/weight_v folded before calling. */
void aria_conv1d(float *out, const float *in, const float *w, const float *bias,
                 int Cin, int Cout, int K, int pad, int L);

/* ---- GLU/SwiGLU feed-forward ---- */

/* FeedForward: proj x[N,dim] -> [N,2*inner] via W_in[2*inner,dim](+b_in),
 * GLU combine value*SiLU(gate) (value = first half), then W_out[dim_out,inner]
 * (+b_out) -> out[N,dim_out]. scratch: [N*3*inner] floats, or NULL to malloc. */
void aria_ff_glu(float *out, const float *x, int N, int dim, int inner, int dim_out,
                 const float *W_in, const float *b_in,
                 const float *W_out, const float *b_out, float *scratch);

/* ---- softmax over rows ---- */
/* In-place row softmax. x is [rows, cols]. Optional additive mask [rows,cols]
 * (e.g. -inf for padding) when mask != NULL. */
void aria_softmax_inplace(float *x, int rows, int cols, const float *mask);

/* ---- elementwise ---- */
void aria_add(float *y, const float *a, const float *b, int n);   /* y = a + b */
void aria_addto(float *y, const float *a, int n);                 /* y += a */
void aria_scale(float *y, const float *x, float s, int n);        /* y = x * s */
void aria_axpy(float *y, const float *x, float s, int n);         /* y += s * x */

#endif /* ARIA_OPS_H */
