# aria.c

A from-scratch, dependency-free **C inference runtime for audio diffusion models**,
starting with **Stable Audio 3** (small-music and medium) and designed to be
**modular** enough to host other audio diffusion models (e.g. AceStep 1.5) by
adding a single model module.

It is the audio sibling of two existing runtimes: `iris.c` (FLUX.2 / Z-Image
image diffusion) and `ds4` (DeepSeek LLMs). It fuses **iris.c's diffusion
architecture + modularity** with **ds4's CUDA backend + int8/q4 quantization +
weight streaming**.

## Goals

- **CPU-first**, no third-party dependencies (AVX2/FMA + OpenMP). Builds and runs
  with **and** without CUDA.
- CUDA backend targeting **RTX 3070 (sm_86)** realistically and **GT 1030 (sm_61,
  2 GB)** as an extreme low-VRAM edge case (fp16 storage + SSD weight streaming).
- **int8/q4 quantization** (dequant-on-use) for small CPU/VRAM footprint.
- A reusable **library** (`libaria.a`) plus a **CLI** (`aria`).

## Architecture

```
CLI (main.c)
 └─ Public API (aria.h)           opaque aria_ctx; load / generate / continue / inpaint
     └─ Orchestrator (aria.c)     model detect/load, sampler loop, low-VRAM load/free
         ├─ Model modules         aria_model_sa3.c (+ aria_dit_sa3.c / aria_taae.c /
         │                        aria_t5gemma.c collaborators)  ← AceStep slots in here
         └─ Op library (aria_ops.h)
             ├─ CPU backend  (aria_cpu.c, AVX2/FMA + OpenMP)
             └─ CUDA backend (aria_cuda.cu, #ifdef ARIA_CUDA)        [Phase 3]
   Common: aria_tensor, aria_safetensors, aria_quant, aria_wav, aria_spm (tokenizer), aria_rng
```

A model is described by an `aria_model_module` vtable (detect / load / generate)
in `aria_model.h`; the registry lives in `aria.c`. Kernels, sampler and I/O are
model-agnostic, so a new architecture is one module file plus a registry entry.

## Build

```sh
make            # CPU build -> ./aria + build/libaria.a
make test       # unit tests
make cuda CUDA_ARCH=sm_86   # CUDA build (Phase 3)
```

## Usage

```sh
# Phase 0 (implemented):
./aria -m <model_dir> --info
./aria -m <model_dir> --list-tensors pretransform.
./aria --wav-roundtrip in.wav out.wav

# Phase 1 (text-to-audio, in progress):
./aria -m <model_dir> -p "warm romantic piano" -d 15 -s 8 -o out.wav
```

`<model_dir>` is a directory containing `model_config.json` and
`model.safetensors` (e.g. a HuggingFace snapshot of
`stabilityai/stable-audio-3-small-music`).

## Status

Phase 0 (scaffolding) — safetensors loader, tensor/WAV I/O, op surface + CPU
kernels, model registry, CLI, unit tests. Generation lands in Phase 1.

See `../sa3-sf-api` for the Python reference (`stable-audio-tools`) used for
parity testing.
