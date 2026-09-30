# Changelog

All notable changes to *aria* are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

Native Windows build support: the same runtime, compiled for Windows x64 with MSVC + `nvcc`.

### Added

- Native Windows x64 build path: the C sources compile with MSVC and are linked by `nvcc` against
  the CUDA toolkit already used on POSIX, with no second ML stack, Python, or PyTorch involved.
- Windows compatibility headers for what the sources previously assumed from the platform:
  64-bit `CreateFileMapping`/`MapViewOfFile` safetensors mapping, `clock_gettime` via
  `QueryPerformanceCounter`, `MAP_*`/`madvise`/`sysconf` shims, `strtok_r`/`strcasestr`, and CRT
  differences (`isatty`, `setenv`).
- Windows-safe large-file seeking for model, config, and text-encoder assets
  (`_fseeki64`/`_ftelli64`/`_fstat64` instead of 32-bit `long` offsets).
- A `pthread` shim for MSVC built on `SRWLOCK` + `SleepConditionVariableSRW` -- both genuinely
  zero-initializable, so `PTHREAD_MUTEX_INITIALIZER` is a real initializer and no lazy
  `InitializeCriticalSection` is needed -- with `_beginthreadex` threads.
- README: native Windows CUDA build instructions, `--precision q8` usage, and how to read the
  performance numbers.

### Performance

- Measured on the Windows build (RTX 3070, Stable Audio 3 Medium, `--precision q8`): a 180-second
  progressive-house generation in 131.77 s of generation time, 135.395 s wall clock
  (1.37x / 1.33x realtime). The q8 path itself is unchanged by this PR; the numbers document that
  it reaches the MSVC build intact.

### Compatibility

- The POSIX/CPU build path is unchanged.
- Live prompt re-steering during `--stream` stays POSIX-only: it relies on `select()` over fd 0,
  which has no portable Windows console/pipe equivalent. Windows stubs it out and does not
  advertise the hint.
- The Windows runtime needs only the built binary plus the installed NVIDIA driver/CUDA runtime.

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
[0.2.0]: https://github.com/matteospanio/aria/releases/tag/v0.2.0
[0.1.0]: https://github.com/matteospanio/aria/releases/tag/v0.1.0
