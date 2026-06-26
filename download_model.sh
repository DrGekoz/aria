#!/bin/sh
# download_model.sh - fetch Stable Audio 3 weights for aria.
#
# Pulls exactly what aria needs (model_config.json, model.safetensors, and the
# t5gemma-b-b-ul2/ text-encoder subfolder) into ./models/<name>/.
#
# Stable Audio 3 is GATED: accept the license on the model page and be logged in.
set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT_BASE=${ARIA_MODELS_DIR:-"$ROOT/models"}
case "$OUT_BASE" in
    /*) ;;
    *) OUT_BASE="$ROOT/$OUT_BASE" ;;
esac
TOKEN=${HF_TOKEN:-}
DOWNLOAD_ALL=0

usage() {
    cat <<EOF
Stable Audio 3 weight downloader for aria

Usage:
  ./download_model.sh small-music [--token TOKEN] [--all]
  ./download_model.sh small-sfx   [--token TOKEN] [--all]
  ./download_model.sh medium      [--token TOKEN] [--all]

Targets:

  small-music   stabilityai/stable-audio-3-small-music
                ~2.3 GB DiT + autoencoder, plus ~1.2 GB T5Gemma encoder.
                Recommended starting point; runs on CPU and small GPUs.

  small-sfx     stabilityai/stable-audio-3-small-sfx
                Sound-effects variant, same size class as small-music.

  medium        stabilityai/stable-audio-3-medium
                Larger DiT (differential attention) + SAME-L autoencoder.
                Needs int8/q4 quantization to fit a 2 GB GPU.

Options:
  --token TOKEN  Hugging Face token. Falls back to HF_TOKEN, then to the local
                 token cache (~/.cache/huggingface/token).
  --all          Download the entire repo instead of only the files aria needs
                 (use if a model stores its text encoder under a different name).

Output goes to ./models/<name>/ (override the base with ARIA_MODELS_DIR).
Downloads resume if interrupted: run the same command again.
EOF
}

[ $# -ge 1 ] || { usage >&2; exit 1; }
MODEL=$1
shift

case "$MODEL" in
    small-music) REPO="stabilityai/stable-audio-3-small-music" ;;
    small-sfx)   REPO="stabilityai/stable-audio-3-small-sfx" ;;
    medium)      REPO="stabilityai/stable-audio-3-medium" ;;
    -h|--help|help) usage; exit 0 ;;
    *)
        echo "Unknown model: $MODEL" >&2
        echo >&2
        usage >&2
        exit 1
        ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --token)
            shift
            [ $# -gt 0 ] || { echo "Missing value after --token" >&2; exit 1; }
            TOKEN=$1
            ;;
        --all) DOWNLOAD_ALL=1 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

if [ -z "$TOKEN" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

# Prefer the new `hf` CLI, fall back to `huggingface-cli`.
HF_CMD=
if command -v hf >/dev/null 2>&1; then
    HF_CMD=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CMD=huggingface-cli
fi
if [ -z "$HF_CMD" ]; then
    echo "This script needs the Hugging Face CLI." >&2
    echo "Install it with:  python3 -m pip install -U huggingface_hub" >&2
    exit 1
fi

OUT="$OUT_BASE/$MODEL"
mkdir -p "$OUT"

echo "Downloading $REPO"
echo "  into $OUT"
echo "  using '$HF_CMD download'"
echo
echo "Stable Audio 3 is gated: accept the license at"
echo "  https://huggingface.co/$REPO"
echo "and authenticate (token via --token, HF_TOKEN, or ~/.cache/huggingface/token)."
echo "If the download stops, run the same command again to resume it."
echo

set -- download "$REPO" --repo-type model --local-dir "$OUT"
if [ "$DOWNLOAD_ALL" -eq 0 ]; then
    set -- "$@" --include "model_config.json" "model.safetensors" "t5gemma-b-b-ul2/*"
fi
if [ -n "$TOKEN" ]; then
    set -- "$@" --token "$TOKEN"
fi

"$HF_CMD" "$@"

if [ ! -s "$OUT/model.safetensors" ]; then
    echo "Download finished but $OUT/model.safetensors is missing." >&2
    echo "If this model stores its weights differently, retry with --all." >&2
    exit 1
fi

echo
echo "Done. Inspect it with:"
echo "  ./aria -m \"$OUT\" --info"
