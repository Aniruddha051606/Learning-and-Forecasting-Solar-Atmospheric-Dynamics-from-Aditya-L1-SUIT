"""Check that re-running Phase 1 on more data leaves an existing data set's registration and QC unchanged.

    python scripts/compare_registration.py --before outputs/phase1/snapshot_before_c0/registration.parquet
        [--dataset outputs/phase2/sequences/frames.parquet]

For every frame of the data set: registration (reg_x0, reg_y0, reg_R), QC decision (qc_usable) and
reasons, before vs now (outputs/phase1/registration.parquet). Reports the largest differences. A data set
is only reproducible from the new Phase 1 tables if nothing that selects or places its frames changed.
Writes <dataset folder>/registration_check.json.
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
    ap.add_argument("--before", required=True)
    ap.add_argument("--dataset", default=str(config.ROOT / "outputs" / "phase2" / "sequences" / "frames.parquet"))
    a = ap.parse_args()
    ds = pd.read_parquet(a.dataset, columns=["frame_id"])
    cols = ["file", "reg_x0", "reg_y0", "reg_R", "qc_usable", "qc_reasons", "segment"]
    b = pd.read_parquet(a.before, columns=cols).set_index("file").loc[ds.frame_id]
    n = pd.read_parquet(config.out_dir(CFG) / "registration.parquet", columns=cols).set_index("file")
    missing = int((~ds.frame_id.isin(n.index)).sum())
    n = n.reindex(ds.frame_id)
    out = {"frames": len(ds), "missing_now": missing}
    for c in ("reg_x0", "reg_y0", "reg_R"):
        d = (n[c] - b[c]).abs()
        out[f"max_abs_diff_{c}"] = float(d.max())
        out[f"frames_diff_gt_0.01px_{c}"] = int((d > 0.01).sum())
    out["qc_usable_changed"] = int((n.qc_usable.astype(bool) != b.qc_usable.astype(bool)).sum())
    out["qc_reasons_changed"] = int((n.qc_reasons.fillna("") != b.qc_reasons.fillna("")).sum())
    out["unchanged"] = bool(missing == 0 and out["qc_usable_changed"] == 0 and out["qc_reasons_changed"] == 0
                            and all(out[f"max_abs_diff_{c}"] < 0.01 for c in ("reg_x0", "reg_y0", "reg_R")))
    out.update(CFG["_meta"])
    (Path(a.dataset).parent / "registration_check.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k not in ("config_path", "git")}, indent=1, default=str))


if __name__ == "__main__":
    main()
