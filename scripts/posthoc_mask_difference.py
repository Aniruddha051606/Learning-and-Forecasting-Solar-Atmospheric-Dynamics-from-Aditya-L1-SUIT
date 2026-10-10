"""Size of the target-validity leak (docs/PREREGISTRATION.md, Addenda H and I): which pixels did the original
mask channel isfinite(target) & isfinite(context) switch off that the input-only mask isfinite(context)
keeps?

    python scripts/posthoc_mask_difference.py --dataset final_offset [--windows 600]
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--windows", type=int, default=600, help="windows per split, evenly spaced")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.environ["SUITDYN_DATASET"] = a.dataset
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from suitdyn import atomic, config, paths
    from suitdyn.ml import data

    P3 = config.load_phase3()
    bank = data.Bank(paths.phase3("cache"), a.device)
    data.load_background(bank, paths.phase3("background", f"static_bg_{bank.G}.npz"))
    inner = bank.mu > float(np.sqrt(1 - P3["eval"]["disk_rho_max"] ** 2))
    res = {}
    for split in ("train", "holdout", "val"):
        ids = bank.ids(split)
        ids = ids[np.linspace(0, len(ids) - 1, min(a.windows, len(ids))).astype(int)]
        disk_ctx = disk_off = frame_ctx = frame_off = hit = 0
        for n0 in range(0, len(ids), 8):
            b = bank.batch(ids[n0:n0 + 8], P3["train"]["inputs"])
            ctx = data.input_mask(b["x"])[:, 0]
            off = ctx & ~b["valid"][:, 0]
            disk_ctx += int((ctx & inner).sum())
            disk_off += int((off & inner).sum())
            frame_ctx += int(ctx.sum())
            frame_off += int(off.sum())
            hit += int(((off & inner).flatten(1).sum(1) > 0).sum())
        res[split] = {"windows": int(len(ids)), "disk_pixels_removed_pct": 100 * disk_off / max(disk_ctx, 1),
                      "frame_pixels_removed_pct": 100 * frame_off / max(frame_ctx, 1),
                      "windows_affected_pct": 100 * hit / len(ids)}
    out = {"dataset": a.dataset, "registration": "docs/PREREGISTRATION.md, Addenda H and I", "splits": res}
    atomic.write_json(paths.phase3("posthoc", "mask_difference.json"), out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
