"""Check that every raw frame of a data set is present, byte-identical, in the current raw archive.

    python scripts/verify_dataset.py [--dataset outputs/phase2/sequences/frames.parquet]

Compares the data set's frame list (file name + SHA-256 recorded when it was built) with the current
manifest (outputs/phase1/manifest.parquet, rebuilt from raw_root = the network share). Reports, per
split and overall: identical, different content (same name, other SHA-256), missing. Nothing is read
from the archive here: the manifest already holds the checksums. A data set is reproducible from the
share only when every frame is identical; until then the local copy (local_copy_root) is kept.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import config  # noqa: E402

CFG = config.load()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(config.seq_dir() / "frames.parquet"))
    a = ap.parse_args()
    ds = pd.read_parquet(a.dataset)
    man = pd.read_parquet(config.out_dir(CFG) / "manifest.parquet", columns=["file", "sha256", "path"])
    m = ds[["frame_id", "sha256", "split"]].merge(man.rename(columns={"file": "frame_id", "sha256": "sha_now"}),
                                                  on="frame_id", how="left")
    m["status"] = "identical"
    m.loc[m.sha_now.isna(), "status"] = "missing"
    m.loc[m.sha_now.notna() & (m.sha_now != m.sha256), "status"] = "different"
    by = m.groupby(["split", "status"]).size().unstack(fill_value=0)
    roots = sorted({str(Path(p).parents[2]) for p in man.path.dropna().head(1000)})
    out = {"dataset": a.dataset, "manifest_roots_sample": roots, "frames": len(m),
           "status": m.status.value_counts().to_dict(), "by_split": by.to_dict("index"),
           "reproducible_from_archive": bool((m.status == "identical").all()),
           "missing_examples": m.loc[m.status == "missing", "frame_id"].head(5).tolist(),
           "different_examples": m.loc[m.status == "different", "frame_id"].head(5).tolist(), **CFG["_meta"]}
    dst = Path(a.dataset).parent / "verify_against_archive.json"
    dst.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("frames", "status", "by_split", "reproducible_from_archive",
                                         "missing_examples", "different_examples")}, indent=1, default=str))


if __name__ == "__main__":
    main()
