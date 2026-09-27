"""Build the raw-data manifest (headers + SHA-256 of every FITS file).

    python scripts/build_manifest.py
Writes <out>/manifest.parquet and <out>/manifest_meta.json.

Source of each row (column `source`):
  archive     read from raw_root, the network share (the archive of record)
  local_copy  a file not (yet) on the share, taken from the manifest of the untouched local copy
              ([paths] local_copy_manifest) and still present on disk. The share is being downloaded
              into; without this fallback, rebuilding from a partial share would silently drop frames
              of existing data sets. Files present in both are taken from the share (they were
              checked byte-identical: 10,977 of 10,977 on 2026-09-27).
"""
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402

from suitdyn import atomic, config, manifest  # noqa: E402


def main():
    cfg = config.load()
    out = config.out_dir(cfg)
    t0 = time.time()
    prev_path = out / "manifest.parquet"
    prev = pd.read_parquet(prev_path) if prev_path.exists() else None
    if prev is not None and "source" in prev:
        prev = prev[prev.source == "archive"]
    if prev is not None:
        # only complete rows are reused; rows damaged by the 2026-09-27 reuse bug (DATE-OBS lost) are re-read
        prev = prev[prev["DATE-OBS"].notna() & prev["FTR_NAME"].notna()]
    df = manifest.build(cfg["paths"]["raw_root"], workers=cfg["run"]["workers"] + 2, previous=prev)
    df["source"] = "archive"
    n_archive, n_local, conflicts = len(df), 0, 0
    lcm = cfg["paths"].get("local_copy_manifest")
    if lcm and (config.ROOT / lcm).exists():
        loc = pd.read_parquet(config.ROOT / lcm)
        both = loc.merge(df[["file", "sha256"]], on="file", suffixes=("", "_archive"))
        conflicts = int((both.sha256 != both.sha256_archive).sum())
        extra = loc[~loc.file.isin(df.file)]
        extra = extra[[Path(p).exists() for p in extra.path]].assign(source="local_copy")
        n_local = len(extra)
        df = pd.concat([df, extra[[c for c in extra.columns if c in df.columns]]], ignore_index=True)
        df = df.sort_values("t", na_position="last").reset_index(drop=True)
    atomic.to_parquet(df, out / "manifest.parquet")
    meta = {"files": len(df), "read_errors": int(df.read_error.notna().sum()),
            "filter_name_mismatches": int((~df.filter_name_matches_header.fillna(True)).sum()),
            "gigabytes": round(df.bytes.sum() / 1e9, 3), "span": [str(df.t.min()), str(df.t.max())],
            "rows_from_archive": n_archive, "rows_from_local_copy": n_local,
            "archive_vs_local_copy_checksum_conflicts": conflicts,
            # identifies the exact set of raw files: the dataset version everything downstream refers to
            "manifest_sha256": hashlib.sha256("\n".join(sorted(df.sha256.dropna())).encode()).hexdigest(),
            "seconds": round(time.time() - t0, 1), **cfg["_meta"]}
    atomic.write_json(out / "manifest_meta.json", meta)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
