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

/* ---- activations (in-place over n elements) ---- */
void aria_silu(float *x, int n);          /* x * sigmoid(x) */
void aria_gelu_tanh(float *x, int n);     /* gelu_pytorch_tanh */

/* SiLU-gated FFN combine: out[i] = silu(gate[i]) * up[i], n elements. */
void aria_silu_gate(float *out, const float *gate, const float *up, int n);

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
