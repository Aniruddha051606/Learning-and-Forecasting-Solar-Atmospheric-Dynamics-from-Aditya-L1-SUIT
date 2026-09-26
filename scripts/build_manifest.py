"""Build the raw-data manifest (headers + SHA-256 of every FITS file).

    python scripts/build_manifest.py
Writes <out>/manifest.parquet and <out>/manifest_meta.json.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import config, manifest  # noqa: E402


def main():
    cfg = config.load()
    out = config.out_dir(cfg)
    t0 = time.time()
    df = manifest.build(cfg["paths"]["raw_root"], workers=cfg["run"]["workers"] + 2)
    df.to_parquet(out / "manifest.parquet", index=False)
    meta = {"files": len(df), "read_errors": int(df.read_error.notna().sum()),
            "filter_name_mismatches": int((~df.filter_name_matches_header.fillna(True)).sum()),
            "gigabytes": round(df.bytes.sum() / 1e9, 3), "span": [str(df.t.min()), str(df.t.max())],
            "seconds": round(time.time() - t0, 1), **cfg["_meta"]}
    (out / "manifest_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
