# Changelog

## 0.2.0 - Windows CUDA and Stable Audio 3 Medium

### Added

- Native Windows x64 build path using MSVC and NVIDIA `nvcc`.
- Windows compatibility layer for large-file safetensors mapping, timing, allocation, threading, and CRT differences.
- CUDA W8A8 8-bit inference with `ARIA_W8A8=1` and `--precision q8`.
- Stable Audio 3 Medium model support on the CUDA path, including the device-resident medium decoder.
- Exported compact Aria tokenizer workflow for text prompts.
- Windows-safe large-file seeking for model and text-encoder assets.

### Performance

- Verified on an RTX 3070: 180-second Stable Audio 3 Medium progressive-house generation in 131.77 seconds of generation time and 135.395 seconds wall clock.
- Runtime log verifies W8A8 int8 tensor-core GEMMs and GPU-resident DiT and decoder.

### Compatibility

- Existing POSIX/CPU build path remains available.
- Windows runtime requires only the compiled Aria binary and the installed NVIDIA driver/CUDA runtime. Python and PyTorch are not runtime dependencies.
