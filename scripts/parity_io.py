"""parity_io.py - write/read .atns tensor dumps shared with the C parity tests.

.atns layout (little-endian):
  b"ATNS" | u32 version=1 | u32 ndim | u32 dtype(0=f32) | i64 shape[ndim] | f32 data
"""
import struct
import numpy as np


def save_atns(path, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    with open(path, "wb") as f:
        f.write(b"ATNS")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", arr.ndim))
        f.write(struct.pack("<I", 0))
        for s in arr.shape:
            f.write(struct.pack("<q", int(s)))
        f.write(arr.tobytes())


def load_atns(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"ATNS"
        (version,) = struct.unpack("<I", f.read(4))
        assert version == 1
        (ndim,) = struct.unpack("<I", f.read(4))
        (dtype,) = struct.unpack("<I", f.read(4))
        assert dtype == 0
        shape = [struct.unpack("<q", f.read(8))[0] for _ in range(ndim)]
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(shape)
