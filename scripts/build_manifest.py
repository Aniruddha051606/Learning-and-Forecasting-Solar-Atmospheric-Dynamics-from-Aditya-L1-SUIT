"""Build the raw-data manifest (headers + SHA-256 of every FITS file).

    python scripts/build_manifest.py
Writes <out>/manifest.parquet and <out>/manifest_meta.json.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402

from suitdyn import config, manifest  # noqa: E402


def main():
    cfg = config.load()
    out = config.out_dir(cfg)
    t0 = time.time()
    prev_path = out / "manifest.parquet"
    prev = pd.read_parquet(prev_path) if prev_path.exists() else None
    df = manifest.build(cfg["paths"]["raw_root"], workers=cfg["run"]["workers"] + 2, previous=prev)
    df.to_parquet(out / "manifest.parquet", index=False)
    meta = {"files": len(df), "read_errors": int(df.read_error.notna().sum()),
            "filter_name_mismatches": int((~df.filter_name_matches_header.fillna(True)).sum()),
            "gigabytes": round(df.bytes.sum() / 1e9, 3), "span": [str(df.t.min()), str(df.t.max())],
            "rows_reused": df.attrs.get("reused", 0), "files_read": df.attrs.get("read", len(df)),
            # identifies the exact set of raw files: the dataset version everything downstream refers to
            "manifest_sha256": hashlib.sha256("\n".join(sorted(df.sha256.dropna())).encode()).hexdigest(),
            "seconds": round(time.time() - t0, 1), **cfg["_meta"]}
    (out / "manifest_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
