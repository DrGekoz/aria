# aria

**aria** is a small native inference engine for **audio diffusion models**,
written from scratch in C with no third-party dependencies. Its first target is
**Stable Audio 3** (small-music and medium). It is intentionally narrow: not a
generic model runner, not a wrapper around PyTorch or ONNX — it is completely
self-contained, and it is built to run on **budget hardware**, CPU-first, with
an optional CUDA backend for old and cheap NVIDIA cards.

It is the audio sibling of two existing native runtimes, and it borrows
deliberately from both:

- **`iris.c`** (FLUX.2 / Z-Image image diffusion) — the diffusion architecture
  and the modularity: a DiT, an autoencoder, a text encoder, a sampler, mmap
  safetensors, one source file per model.
- **`ds4`** (DeepSeek LLMs) — the CUDA backend, int8/q4 quantization with
  dequant-on-use, mmap + SSD weight streaming, and compile-time backend
  selection.

aria is the union of the two, pointed at audio: **iris.c's shape with ds4's
budget-hardware muscle.**

We support (or are building toward) the following backends:

- **CPU** — the primary, highest-quality target. AVX2/FMA + OpenMP, no BLAS, no
  external libraries. It must always build and pass with `make`.
- **NVIDIA CUDA** — secondary accelerator. The realistic GPU target is an RTX
  3070 (sm_86); a 2 GB Pascal GT 1030 (sm_61) is supported as an extreme
  low-VRAM edge case via fp16-storage and SSD weight streaming.

## Motivations

- Capable open-weight **audio** diffusion models now exist and are small enough
  to run locally. Stable Audio 3 small-music is ~460M params of DiT plus a
  ~108M autoencoder — that fits on hardware most people already own.
- The Python stack (PyTorch + `stable-audio-tools`) is wonderful for research
  and miserable for deployment on a 2 GB GPU or an old laptop. A self-contained
  native runtime turns "can I run this at all?" into "how fast does it run?"
- We want this to be the efficient inference vehicle for **taste-steering**
  research on Stable Audio 3 (see [Steering](#steering)). A from-scratch C
  runtime exposes every intermediate tensor, which makes training-free
  activation steering a one-line tensor add instead of a framework fight.
- New audio diffusion models appear constantly. aria is **opportunistic and
  modular**: a model is one module file behind a small vtable. When a better
  open model lands (AceStep 1.5, whatever comes next), it slots in without
  touching the kernels, sampler, quantization, or I/O.

## Status

**Working end-to-end (small-music, CPU).** `aria` loads Stable Audio 3 and
generates audio from a text prompt, entirely in C — every stage parity-checked
against the `stable-audio-tools` PyTorch reference:

- **text → tokens** (GemmaTokenizer BPE, exact id parity) →
- **T5Gemma encoder** (12-layer Gemma2 encoder, ~0.15% rel) →
- **DiT denoiser** (20 blocks, ~1e-5) over the **pingpong sampler** (LogSNR
  schedule, ~6e-8) → **taae_v2 decoder** (latent→audio, ~1e-4) → stereo WAV.

CPU-only, AVX2/FMA + OpenMP, zero-copy mmap weights. ~9 s for a 2 s/8-step clip.

- **Not yet:** CUDA backend, int8/q4 quantization, continue/inpaint, the medium
  model. See [ROADMAP.md](ROADMAP.md) for the atomic, individually-verifiable
  task list with priorities and dependencies. This software
is developed with **strong assistance from large language models**, with a human
leading the ideas, testing, and debugging. We say so openly because it shaped how
the project was built.

## Documentation

- [AGENTS.md](AGENTS.md) — conventions and ground rules for anyone (human or AI)
  working in this repo. **Read this before changing code.**
- [CONTRIBUTING.md](CONTRIBUTING.md) — the parity and build regression tracks.
  **Read this before sending a pull request.**
- [ROADMAP.md](ROADMAP.md) — epics → atomic tasks to v1.0.0, with milestones,
  dependencies, and what can be done in parallel right now.

## Model weights

Stable Audio 3 weights live on the Hugging Face Hub and are **gated**: you must
accept the license on the model page and be logged in. The download script pulls
exactly what aria needs — `model_config.json`, `model.safetensors`, and the
`t5gemma-b-b-ul2/` text-encoder subfolder — into `./models/<name>/`.

```sh
./download_model.sh small-music   # ~2.3 GB DiT+autoencoder + ~1.2 GB T5Gemma encoder
./download_model.sh small-sfx     # sound-effects variant
./download_model.sh medium        # larger; needs quantization to fit a 2 GB GPU
```

Authentication: the script uses `--token TOKEN`, the `HF_TOKEN` environment
variable, or your local Hugging Face token cache (`~/.cache/huggingface/token`),
in that order. Get a token at <https://huggingface.co/settings/tokens> after
accepting the model license. Downloads resume if interrupted — run the same
command again.

The script requires the Hugging Face CLI (`hf` or `huggingface-cli`):

```sh
python3 -m pip install -U huggingface_hub
```

## Build

```sh
make            # CPU build -> ./aria + build/libaria.a   (default)
make test       # hermetic unit tests (no model, no GPU)
make cuda CUDA_ARCH=sm_86   # CUDA build (sm_61 for a GT 1030)
make clean
```

Requirements: a C11 compiler (gcc/clang), `make`, libm, and OpenMP (a compiler
feature, not a library). No BLAS, no libsndfile, no JSON library, nothing else.
The CPU build must stay warning-clean under `-Wall -Wextra`.

## Usage

```sh
# one-time: export the tokenizer to a compact binary aria loads at runtime
python scripts/export_tokenizer.py models/small-music

# text -> audio
./aria -m models/small-music -p "warm romantic piano, slow, tender" -d 15 -s 8 --seed 0 -o out.wav

# unconditional, or from a precomputed [256,768] prompt embedding
./aria -m models/small-music --uncond -d 10 -o out.wav
./aria -m models/small-music --prompt-embed prompt.atns -d 10 -o out.wav

# inspect
./aria -m models/small-music --info
./aria -m models/small-music --list-tensors pretransform.
```

`-m <dir>` is any directory with `model_config.json` and `model.safetensors`
(what `download_model.sh` produces, or a raw Hugging Face snapshot).

## Architecture

Five layers, CPU-first, CUDA optional:

```
CLI (main.c)
 └─ Public API (aria.h)            opaque aria_ctx; load / generate / continue / inpaint
     └─ Orchestrator (aria.c)      model detect/load, sampler loop, low-VRAM load/free
         ├─ Model modules          aria_model_sa3.c (+ aria_dit_sa3 / aria_taae / aria_t5gemma)
         └─ Op library (aria_ops.h)
             ├─ CPU backend  (aria_cpu.c, AVX2/FMA + OpenMP)
             └─ CUDA backend (aria_cuda.cu, #ifdef ARIA_CUDA)
   Common: aria_safetensors, aria_quant, aria_wav, aria_cond, aria_rng, aria_parity
```

A model is an `aria_model_module` vtable (`detect / load / generate`) registered
in `aria.c`. Kernels, sampler, and I/O are model-agnostic, so adding a new
architecture is one module file plus a registry entry. See [AGENTS.md](AGENTS.md)
for the full contract.

## Backends and precision

The CPU path is the reference and the production target on machines without a
capable GPU (on a 2 GB GT 1030, plain CPU is often faster anyway). The optimized
path is fp16-storage + fp32-compute; int8/q4 quantization with dequant-on-use is
how the medium model and the 2 GB GPU are made to fit. The CUDA backend is an
accelerator, never a requirement — `make` (CPU) must always work.

## Steering

aria is designed to be **steerable**. Because a native runtime owns every
intermediate tensor, training-free activation steering is an elementwise
`x += scale * direction` at a known boundary: the 256-D diffusion latent, the
per-layer DiT residual stream, the conditioning vectors, or the pre-decode
latent. Directions are plain tensor files, and the parity harness doubles as the
activation-extraction path. This is a first-class roadmap epic (E12); see
[ROADMAP.md](ROADMAP.md).

## Parity and testing

Correctness is proven against the **real** `stable-audio-tools` PyTorch
reference, not by inspection. The `make parity` harness dumps reference tensors
from PyTorch and compares them in C with magnitude-aware tolerances. See
[CONTRIBUTING.md](CONTRIBUTING.md) for how to run and extend it.

## Acknowledgements

aria would not exist without two reference runtimes — **`iris.c`** (its diffusion
architecture and modularity) and **`ds4`** (its CUDA, quantization, and streaming
ideas) — and without **Stable Audio 3** and **`stable-audio-tools`** by Stability
AI, which provide the model and the parity ground truth. Thank you.
