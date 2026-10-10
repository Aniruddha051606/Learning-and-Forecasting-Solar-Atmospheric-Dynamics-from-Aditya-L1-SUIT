"""Draft a data-set config per pointing cluster from the pointing-mode scan (scripts/pointing_modes.py).

    python scripts/propose_splits.py [--train 0.70 --val 0.15] [--min-days 2] [--write]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import config, paths  # noqa: E402

TEMPLATE = """# Final data set: pointing cluster {cluster} ({n} NB03 full-disk frames, {t0} -> {t1} UT).
# Drafted by scripts/propose_splits.py from the pointing-mode scan; chronological splits with each boundary in an
# observing gap. Rules as data set v0. Products: outputs/datasets/{name}/.

[dataset]
pointing_mode = "{mode}"
# Pointing cluster from the header scan: {cluster}. Not pinned by name (registration labels clusters from the
# fitted limb centre, which can round differently); the time span selects it, and build_sequences.py refuses
# the data set if another pointing appears inside it.
exclude_flags = ["pointing_mode_change", "limb_outlier", "spike_rate"]
drop_program_first_frame = true

[scope]
margin_h = 3

[split]
embargo_h = 4.0
train = ["{a}", "{b}"]
val = ["{b}", "{c}"]
test = ["{c}", "{d}"]

[sequences]
max_gap_s = 300
contexts = [1, 5, 20]
horizons = [1, 5, 10, 20, 40, 80, 160]
"""


def boundary(t, q):
    """Midpoint of the observing gap (> 30 min) nearest to the q-quantile of the frame times."""
    target = t.quantile(q)
    gaps = t.diff().dt.total_seconds()
    starts = t.shift(1)[gaps > 1800]
    ends = t[gaps > 1800]
    mids = starts + (ends.values - starts.values) / 2
    if not len(mids):
        return target.floor("min")
    k = int(np.argmin(np.abs((mids - target).dt.total_seconds().values)))
    return mids.iloc[k].floor("min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--min-days", type=float, default=2.0)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    nb = pd.read_parquet(paths.archive("pointing_modes.parquet")).sort_values("t")
    for k, (cluster, g) in enumerate(nb.groupby("cluster")):
        t = g.t.reset_index(drop=True)
        days = (t.iloc[-1] - t.iloc[0]).total_seconds() / 86400
        if days < a.min_days:
            print(f"skip {cluster}: {days:.1f} days")
            continue
        mode = cluster.split("@")[0]
        name = f"final_{mode}" + (f"_{k}" if (nb.cluster.str.startswith(mode + "@")).groupby(nb.cluster).any().sum() > 1 else "")
        b, c = boundary(t, a.train), boundary(t, a.train + a.val)
        text = TEMPLATE.format(cluster=cluster, n=len(t), t0=t.iloc[0], t1=t.iloc[-1], name=name, mode=mode,
                               a=(t.iloc[0] - pd.Timedelta(minutes=1)).floor("min"), b=b, c=c,
                               d=(t.iloc[-1] + pd.Timedelta(minutes=1)).ceil("min"))
        f = {s: int(((t >= lo) & (t < hi)).sum()) for s, (lo, hi) in
             {"train": (t.iloc[0], b), "val": (b, c), "test": (c, t.iloc[-1] + pd.Timedelta(minutes=1))}.items()}
        print(f"===== {name}: {cluster}, {days:.1f} days, frames {f}\n{text}")
        if a.write:
            (config.ROOT / "configs" / "datasets" / f"{name}.toml").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
