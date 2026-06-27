# aria.c - audio diffusion inference runtime
#
#   make help       # list all targets + key variables
#
# CPU-first per design: AVX2/FMA + OpenMP, no third-party deps.

CC      ?= gcc
AR      ?= ar
CSTD    ?= -std=c11
DEFS    ?= -D_GNU_SOURCE
WARN    ?= -Wall -Wextra -Wno-unused-parameter
OPT     ?= -O3 -march=native -mavx2 -mfma -fopenmp
CFLAGS  ?= $(CSTD) $(DEFS) $(WARN) $(OPT)
LDFLAGS ?= -fopenmp -lm

SRC   := src
BUILD := build

LIB_SRCS := \
  $(SRC)/aria_safetensors.c \
  $(SRC)/aria_wav.c \
  $(SRC)/aria_json.c \
  $(SRC)/aria_cpu.c \
  $(SRC)/aria_quant.c \
  $(SRC)/aria_arena.c \
  $(SRC)/aria_cond.c \
  $(SRC)/aria_parity.c \
  $(SRC)/aria_sa3_config.c \
  $(SRC)/aria_sa3_dit.c \
  $(SRC)/aria_taae.c \
  $(SRC)/aria_sa3_dec.c \
  $(SRC)/aria_sa3_enc.c \
  $(SRC)/aria_t5enc.c \
  $(SRC)/aria_tokenizer.c \
  $(SRC)/aria_sampler.c \
  $(SRC)/aria.c \
  $(SRC)/aria_model_sa3.c

EXTRA_LIB_OBJS ?=
LIB_OBJS := $(patsubst $(SRC)/%.c,$(BUILD)/%.o,$(LIB_SRCS)) $(EXTRA_LIB_OBJS)
LIB := $(BUILD)/libaria.a

.PHONY: all cpu clean test cuda test_cuda blas bench bench-sweep profile help quantize
all: cpu
cpu: aria

# offline weight quantizer (E9.2): model.safetensors -> packed .aria DiT overlay
quantize: aria-quantize
aria-quantize: $(LIB)
	$(CC) $(CFLAGS) -I$(SRC) tools/aria_quantize.c -L$(BUILD) -laria $(LDFLAGS) -o aria-quantize

help:
	@echo "aria.c - make targets:"
	@echo ""
	@echo "  Build"
	@echo "    make / make cpu    CPU build (default): build/libaria.a + ./aria"
	@echo "    make cuda          CUDA build (CUDA_ARCH=sm_86 for RTX 3070; default sm_61."
	@echo "                       also NVCC, CUDA_HOME, CUDA_CCBIN)"
	@echo "    make blas          CPU build routing GEMM to a BLAS (BLAS_LIB=-lopenblas)"
	@echo "    make quantize      offline DiT quantizer -> ./aria-quantize (E9.2)"
	@echo "    make clean         remove build/ and ./aria"
	@echo ""
	@echo "  Test"
	@echo "    make test          hermetic unit tests (no model needed)"
	@echo "    make test_cuda     CUDA op parity vs CPU (skips cleanly with no device)"
	@echo "    make parity        full parity vs PyTorch (ARIA_MODEL=<dir>; needs the venv)"
	@echo ""
	@echo "  Benchmark / profile"
	@echo "    make bench         GEMM microbenchmark on the dominant SA3 shapes"
	@echo "    make bench-sweep   sweep the AVX2 register tile (MR x NR)"
	@echo "    make profile ARGS=\"-m <model> --uncond -d 10 -s 8 --device cuda\""
	@echo "                       stages + memory, plus per-kernel GPU time (nsys). See PROFILING.md"
	@echo ""
	@echo "  Key vars: CC, CUDA_ARCH, NVCC, CUDA_HOME, ARIA_MODEL, ARGS, BLAS_LIB"

# GEMM microbenchmark on the dominant SA3 shapes (no model load)
bench: $(LIB)
	$(CC) $(CFLAGS) -I$(SRC) tests/bench_gemm.c -L$(BUILD) -laria $(LDFLAGS) -o $(BUILD)/bench_gemm
	$(BUILD)/bench_gemm

# end-to-end profile of a generation: built-in stage timing + peak RSS/VRAM, plus
# per-kernel GPU time (nsys) on a CUDA build, or perf (CPU). Uses the built ./aria.
#   make profile ARGS="-m <model> --uncond -d 10 -s 8 --device cuda"
ARGS ?=
profile:
	@bash scripts/profile.sh ./aria $(ARGS)

# sweep the AVX2 register-tile (MR x NR) to pick the best default, then restore it
bench-sweep: $(LIB)
	@for c in 3:3 4:3 4:2 5:2 6:2 2:4; do \
	  mr=$${c%%:*}; nr=$${c##*:}; \
	  $(CC) $(CFLAGS) -DARIA_MR=$$mr -DARIA_NR=$$nr -I$(SRC) -c $(SRC)/aria_cpu.c -o $(BUILD)/aria_cpu.o 2>/dev/null; \
	  $(AR) rs $(BUILD)/libaria.a $(BUILD)/aria_cpu.o 2>/dev/null; \
	  $(CC) $(CFLAGS) -I$(SRC) tests/bench_gemm.c -L$(BUILD) -laria $(LDFLAGS) -o $(BUILD)/bench_gemm 2>/dev/null; \
	  printf "MR=%s NR=%s: " $$mr $$nr; \
	  $(BUILD)/bench_gemm | awk '/GFLOP/{s+=$$(NF-1);n++} END{printf "%.0f GFLOP/s avg\n", s/n}'; \
	done; \
	$(CC) $(CFLAGS) -I$(SRC) -c $(SRC)/aria_cpu.c -o $(BUILD)/aria_cpu.o 2>/dev/null; $(AR) rs $(BUILD)/libaria.a $(BUILD)/aria_cpu.o 2>/dev/null

# Optional CPU acceleration: route GEMM to a BLAS (OpenBLAS here; on macOS use
# LDLIBS="-framework Accelerate"). Keeps the pure-C build as the default.
BLAS_LIB ?= -lopenblas
blas:
	rm -f $(BUILD)/*.o $(BUILD)/libaria.a aria   # rebuild objects (CFLAGS change); keep dumps
	$(MAKE) all CFLAGS="$(CSTD) $(DEFS) $(WARN) $(OPT) -DARIA_BLAS" \
	            LDFLAGS="-fopenmp -lm $(BLAS_LIB)"

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/%.o: $(SRC)/%.c | $(BUILD)
	$(CC) $(CFLAGS) -I$(SRC) -c $< -o $@

$(LIB): $(LIB_OBJS)
	$(AR) rcs $@ $^

aria: $(BUILD)/main.o $(LIB)
	$(CC) $(CFLAGS) -I$(SRC) $< -L$(BUILD) -laria $(LDFLAGS) -o $@

# ---- tests ----
TESTS := test_ops test_wav test_config test_sampler test_arena test_quant
test: $(LIB)
	@set -e; \
	for t in $(TESTS); do \
	  echo "==> build $$t"; \
	  $(CC) $(CFLAGS) -I$(SRC) tests/$$t.c -L$(BUILD) -laria $(LDFLAGS) -o $(BUILD)/$$t; \
	  echo "==> run $$t"; \
	  $(BUILD)/$$t; \
	done; \
	echo "all tests passed"

# ---- parity tests (need a model dir + the sa3-sf-api venv) ----
PYTHON    ?= ../sa3-sf-api/.venv/bin/python
ARIA_MODEL ?=
DUMPS     ?= build/parity_dumps
PARITY_TESTS := test_number_cond test_attn test_dit test_dit_full test_quant_dit test_schedule test_dec test_enc test_inpaint test_inpaint_e2e test_e2e test_t5enc test_tokenizer
parity: $(LIB)
	@test -n "$(ARIA_MODEL)" || { echo "usage: make parity ARIA_MODEL=<model dir>"; exit 1; }
	$(PYTHON) scripts/dump_phase1.py "$(ARIA_MODEL)" "$(DUMPS)"
	$(PYTHON) scripts/export_tokenizer.py "$(ARIA_MODEL)" $(BUILD)/aria_tokenizer.bin
	@set -e; for t in $(PARITY_TESTS); do \
	  echo "==> build $$t"; \
	  $(CC) $(CFLAGS) -I$(SRC) tests/$$t.c -L$(BUILD) -laria $(LDFLAGS) -o $(BUILD)/$$t; \
	  echo "==> run $$t"; \
	  ARIA_MODEL="$(ARIA_MODEL)" ARIA_DUMPS="$(DUMPS)" ARIA_TOKENIZER="$(BUILD)/aria_tokenizer.bin" $(BUILD)/$$t; \
	done; \
	echo "parity passed"

# ---- CUDA backend (E8) ----
# Local dev box: GT 1030 (sm_61), CUDA 11.2 at /usr/lib/cuda (needs host gcc <= 10).
# RTX 3070: make cuda CUDA_ARCH=sm_86. The pure-C build stays the default.
NVCC       ?= /usr/lib/cuda/bin/nvcc
CUDA_HOME  ?= /usr/lib/cuda
CUDA_CCBIN ?= gcc-9
CUDA_ARCH  ?= sm_61
CUDA_CFLAGS  := $(CSTD) $(DEFS) $(WARN) -O3 -march=native -mavx2 -mfma -fopenmp -DARIA_CUDA
CUDA_LDFLAGS := -fopenmp -lm -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart -lcublas -lstdc++

# build the lib (+ aria_cuda.o) and CLI with the CUDA backend linked in
cuda:
	rm -f $(BUILD)/*.o $(BUILD)/libaria.a aria   # CFLAGS change; keep dumps
	$(NVCC) -ccbin $(CUDA_CCBIN) -arch=$(CUDA_ARCH) -O3 -I$(SRC) -c $(SRC)/aria_cuda.cu -o $(BUILD)/aria_cuda.o
	$(MAKE) aria CFLAGS="$(CUDA_CFLAGS)" LDFLAGS="$(CUDA_LDFLAGS)" EXTRA_LIB_OBJS="$(BUILD)/aria_cuda.o"

# CUDA op parity vs CPU (skips cleanly if no device)
test_cuda:
	rm -f $(BUILD)/*.o $(BUILD)/libaria.a
	$(NVCC) -ccbin $(CUDA_CCBIN) -arch=$(CUDA_ARCH) -O3 -I$(SRC) -c $(SRC)/aria_cuda.cu -o $(BUILD)/aria_cuda.o
	$(MAKE) $(LIB) CFLAGS="$(CUDA_CFLAGS)" EXTRA_LIB_OBJS="$(BUILD)/aria_cuda.o"
	$(CC) $(CUDA_CFLAGS) -I$(SRC) tests/test_cuda.c -L$(BUILD) -laria $(CUDA_LDFLAGS) -o $(BUILD)/test_cuda
	$(BUILD)/test_cuda

clean:
	rm -rf $(BUILD) aria aria-quantize
