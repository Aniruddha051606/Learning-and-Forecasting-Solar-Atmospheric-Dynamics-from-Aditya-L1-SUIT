"""Crash-safe writes: write to a temporary file in the same folder, then rename over the target."""
import json
import os
from pathlib import Path

import numpy as np


def _tmp(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, path.with_name(path.name + f".tmp{os.getpid()}")


def write_text(path, text):
    path, tmp = _tmp(path)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def write_json(path, obj):
    return write_text(path, json.dumps(obj, indent=1, default=str))


def to_parquet(df, path, **kw):
    path, tmp = _tmp(path)
    df.to_parquet(tmp, index=False, **kw)
    os.replace(tmp, path)
    return path


def save_npy(path, arr):
    path, tmp = _tmp(path)
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)
    return path


def savez(path, **arrays):
    path, tmp = _tmp(path)
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)
    return path
