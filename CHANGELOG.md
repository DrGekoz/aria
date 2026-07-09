# Changelog

All notable changes to *aria* are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-08

First public release: a dependency-free C/CUDA runtime for Stable Audio 3.

### Added

- Native runtime for the full Stable Audio 3 text-to-music pipeline — tokenizer, T5Gemma text
  encoder, diffusion-transformer denoiser, and audio autoencoder — with no BLAS, deep-learning
  framework, or third-party dependency beyond a C math library and OpenMP.
- Support for both released model variants: `small-music` and the 1.2B-parameter `medium`.
- Two backends behind one op surface: a vectorized, multi-threaded CPU path (AVX2/FMA, ARM/NEON)
  and a CUDA path with half-precision tensor-core kernels and a per-step denoise loop captured and
  replayed as a CUDA graph. Byte-identical output where a path is shared.
- Precision options: fp16 / bf16 / 8-bit (q8) / 4-bit (q4) weight storage, plus an int8-activation
  mode (W8A8) on GPU tensor cores and the equivalent ARM instruction. A source-releasing overlay
  frees the full-precision weights once packed, so a lower precision reduces resident memory rather
  than adding to it.
- Offline quantizer (`aria-quantize`) and `--load-quant` for pre-packed overlays.
- Built-in activation steering: three injection sites (residual stream, latent, text conditioning),
  additive and projection operators, optional step windows, and `.atns` direction files. Live
  steering ramps during streaming; output is bit-identical to the base model at zero strength.
- Per-axis LoRA adapters via `--lora`.
- Long-form and streaming generation: bounded-memory windowed decoding, `--stream` sliding-window
  generation, `--continue` / `--inpaint`, `--anchor` re-anchoring, and `--evolve` seed control.
- Serving: a resident batch mode (`--batch`) and an HTTP server (`aria-server`) that keep the model
  loaded across requests.
- CLI utilities: `--fast` preset, `--bench`, `--info`, `--list-tensors`, and WAV round-trip tools.
- Runs on NVIDIA GPUs, x86 CPUs, and the Raspberry Pi 5, including the medium model at 4-bit within
  an 8 GB memory budget.
- Weight downloader (`download_model.sh`), runnable examples, and benchmark/design notes
  (`README.md`, `BENCHMARKS.md`, `PROFILING.md`).

[Unreleased]: https://github.com/matteospanio/aria/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/matteospanio/aria/releases/tag/v0.1.0
