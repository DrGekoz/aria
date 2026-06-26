# aria.c - audio diffusion inference runtime
#
# Targets:
#   make            # CPU build (default): libaria.a + aria CLI
#   make test       # build & run unit tests
#   make cuda CUDA_ARCH=sm_86   # CUDA build (Phase 3; sm_61 for GT 1030)
#   make clean
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
  $(SRC)/aria_cond.c \
  $(SRC)/aria_parity.c \
  $(SRC)/aria_sa3_config.c \
  $(SRC)/aria_sa3_dit.c \
  $(SRC)/aria_sa3_dec.c \
  $(SRC)/aria_sampler.c \
  $(SRC)/aria.c \
  $(SRC)/aria_model_sa3.c

LIB_OBJS := $(patsubst $(SRC)/%.c,$(BUILD)/%.o,$(LIB_SRCS))
LIB := $(BUILD)/libaria.a

.PHONY: all cpu clean test cuda
all: cpu
cpu: aria

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/%.o: $(SRC)/%.c | $(BUILD)
	$(CC) $(CFLAGS) -I$(SRC) -c $< -o $@

$(LIB): $(LIB_OBJS)
	$(AR) rcs $@ $^

aria: $(BUILD)/main.o $(LIB)
	$(CC) $(CFLAGS) -I$(SRC) $< -L$(BUILD) -laria $(LDFLAGS) -o $@

# ---- tests ----
TESTS := test_ops test_wav test_config test_sampler
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
PARITY_TESTS := test_number_cond test_attn test_dit test_dit_full test_schedule test_dec test_e2e
parity: $(LIB)
	@test -n "$(ARIA_MODEL)" || { echo "usage: make parity ARIA_MODEL=<model dir>"; exit 1; }
	$(PYTHON) scripts/dump_phase1.py "$(ARIA_MODEL)" "$(DUMPS)"
	@set -e; for t in $(PARITY_TESTS); do \
	  echo "==> build $$t"; \
	  $(CC) $(CFLAGS) -I$(SRC) tests/$$t.c -L$(BUILD) -laria $(LDFLAGS) -o $(BUILD)/$$t; \
	  echo "==> run $$t"; \
	  ARIA_MODEL="$(ARIA_MODEL)" ARIA_DUMPS="$(DUMPS)" $(BUILD)/$$t; \
	done; \
	echo "parity passed"

# ---- CUDA (Phase 3) ----
CUDA_ARCH ?= sm_86
cuda:
	@if [ ! -f $(SRC)/aria_cuda.cu ]; then \
	  echo "CUDA backend (aria_cuda.cu) lands in Phase 3. CPU build: 'make'."; exit 1; \
	fi
	$(MAKE) all CFLAGS="$(CSTD) $(WARN) -O3 -fopenmp -DARIA_CUDA" \
	  EXTRA_OBJS="$(BUILD)/aria_cuda.o" CUDA_ARCH=$(CUDA_ARCH)

clean:
	rm -rf $(BUILD) aria
