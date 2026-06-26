# Contributing

aria changes should be tested against the failure mode they can realistically
affect. The project has two regression tracks: **correctness (parity)** and
**build/portability**. Please include the commands you ran, the machine and
backend, the model, and any notable failures in the PR or commit notes.

Do not send PRs touching a numeric op or a model component without a parity
check against the PyTorch reference. "It looks right" is not a passing test.
Read [AGENTS.md](AGENTS.md) first — it carries the golden rules (no third-party
dependencies, CPU-first, simplicity, surgical changes) that every change must
respect.

## Correctness: parity regression tests

Everything numeric in aria is validated against the real `stable-audio-tools`
PyTorch implementation. There are two suites.

### Hermetic unit tests

These need no model, no network, and no GPU. They must always pass and must stay
warning-clean.

```sh
make clean
make
make test
```

`make test` builds and runs the C test runners in `tests/` (`test_ops`,
`test_wav`, ...). They cover the CPU op kernels (linear, matmul, rmsnorm,
dyn_tanh, silu, gelu_tanh, softmax, elementwise) with known-answer checks, and
the WAV round-trip. This is the best quick check for op-level changes.

### PyTorch parity tests

These compare C output against tensors dumped from the real model. They need a
model directory and the `sa3-sf-api` reference venv, so they live behind
environment variables and **skip cleanly** when those are unset (which is why
they do not break `make test`).

```sh
# Download a model first (see README), e.g. to ./models/small-music
make parity ARIA_MODEL=models/small-music
```

`make parity` first runs `scripts/dump_phase1.py` with the reference venv to
write `.atns` reference tensors under `build/parity_dumps/`, then builds and runs
the parity tests against them. Override the interpreter or the model with:

```sh
make parity ARIA_MODEL=/abs/path/to/model PYTHON=/path/to/venv/bin/python
```

What the parity suite covers (growing as the runtime grows):

- `test_number_cond`: the `seconds_total` NumberConditioner (ExpoFourier +
  Linear) against `NumberConditioner.forward`.
- (planned) per-op parity for attention/RoPE/QK-RMSNorm/conv1d, the DiT block and
  full DiT forward, the taae_v2 decoder, and the sampler loop — each landing with
  its own parity test before it is trusted.

### How parity is judged

Float32 matmul accumulation order legitimately differs from numpy/BLAS, so a
fixed tiny tolerance will false-fail. Gate with **atol + rtol**:

```
pass if maxabsdiff <= atol + rtol * max|reference|
```

Typical values are `atol ≈ 3e-4`, `rtol ≈ 2e-3` for linear-heavy ops; tighten for
pure elementwise ops. Where a computation is ill-conditioned in float32 (for
example the Fourier duration/timestep embeddings, whose argument reaches ~6e4),
compute it in double in C and gate against an f64-canonical reference rather than
bit-matching PyTorch's float32 noise — and log the deviation from PyTorch so it
is visible. For RNG-sensitive paths (the sampler), inject identical noise from
Python into the C path so you validate the math independently of the RNG.

When you add an op or a component, add a `.atns` dump to `scripts/dump_phase1.py`
(prefer driving the **real** `stable-audio-tools` module, not a reimplementation)
and a parity test under `tests/`, and wire it into the `parity` target in the
`Makefile`.

## CUDA changes

The CUDA backend is optional and must never be required. For CUDA work, verify
the CPU build is unaffected, then build and parity-check on a CUDA machine:

```sh
make            # CPU still builds and passes
make cuda CUDA_ARCH=sm_86   # or sm_61 for a GT 1030
```

Every CUDA kernel is validated CUDA-vs-CPU within tolerance — the CPU path is the
reference. If you optimize one backend, confirm the other did not regress.

## Adding a model

aria is modular on purpose. To add an architecture (e.g. AceStep 1.5): write a
`aria_dit_<model>.c` (plus autoencoder / encoder modules as needed) built from
the shared op surface in `aria_ops.h`, expose an `aria_module_<model>`
descriptor, and append it to `g_registry[]` in `aria.c`. **Do not** modify the
kernels, sampler, quantization, or I/O to fit a new model — extend the op surface
generically if a primitive is genuinely missing. See the model-module contract in
[AGENTS.md](AGENTS.md).

## Style and hygiene

- No third-party dependencies. Standard C11 + libm + OpenMP only; CUDA only
  inside `#ifdef ARIA_CUDA`.
- Keep `make` and `make test` green and warning-clean (`-Wall -Wextra`).
- Read dims from `model_config.json`; do not hardcode values the config carries.
- Surgical changes: touch only what the task needs, match surrounding style,
  remove only the orphans your change created.
- Do not commit model weights, `.safetensors`, `.atns` dumps, audio, or `build/`.
- Do not download gated weights in CI or without explicit approval.

## Reporting bugs

Include the exact commands, the machine and backend (`cpu` / `cuda sm_XX`), the
model and any quantization, the failing parity values (max diff, tolerance,
`max|ref|`), and — for a failing generation — the prompt, seed, and steps so it
can be reproduced deterministically.
