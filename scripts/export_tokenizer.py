"""export_tokenizer.py - export the GemmaTokenizer vocab + merges to a compact
binary aria can load (parsing the 34 MB tokenizer.json at runtime is slow).

Usage:
  python export_tokenizer.py <model_dir> [out.bin]
default out: <model_dir>/t5gemma-b-b-ul2/aria_tokenizer.bin

Format (little-endian):
  "ATOK" u32 version=1
  u32 n_vocab ; repeated: u32 id, u32 len, <len> UTF-8 bytes (token)
  u32 n_merges ; repeated: u32 len, <len> bytes ("left right", rank = order)
"""
import json
import os
import struct
import sys


def main(model_dir, out_path=None):
    sub = os.path.join(model_dir, "t5gemma-b-b-ul2")
    tok_json = os.path.join(sub, "tokenizer.json")
    if out_path is None:
        out_path = os.path.join(sub, "aria_tokenizer.bin")
    d = json.load(open(tok_json))
    vocab = d["model"]["vocab"]      # {token: id}
    merges = d["model"]["merges"]    # ["left right", ...] or [[left,right], ...]

    with open(out_path, "wb") as f:
        f.write(b"ATOK")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", len(vocab)))
        for tok, i in vocab.items():
            b = tok.encode("utf-8")
            f.write(struct.pack("<I", int(i)))
            f.write(struct.pack("<I", len(b)))
            f.write(b)
        f.write(struct.pack("<I", len(merges)))
        for m in merges:
            if isinstance(m, list):
                joined = m[0] + " " + m[1]
            else:
                joined = m  # already "left right"
            b = joined.encode("utf-8")
            f.write(struct.pack("<I", len(b)))
            f.write(b)
    print(f"wrote {out_path}: {len(vocab)} vocab, {len(merges)} merges")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
