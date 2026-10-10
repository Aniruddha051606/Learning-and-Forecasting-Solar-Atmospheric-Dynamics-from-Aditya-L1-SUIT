"""Phase 0 data audit of a SUIT Level-1 archive."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

KEYS = ["DATE-OBS", "FTR_NAME", "NAXIS1", "NAXIS2", "IMG_TYPE", "OBS_MODE", "ROI_FF", "BIN_EN", "ROI_ID",
        "CMD_EXPT", "MEAS_EXP", "IMGTRID", "QVAL", "QDESC", "F_LEVEL", "F_VER", "CRPIX1", "CRPIX2", "R_SUN",
        "CDELT1", "CROTA2", "X1", "Y1", "X2", "Y2", "SOLX1TR", "SOLX2TR", "HELIOSTR", "NORM_FLR", "PROM_FLR",
        "FLR_ED", "FLAT_CF", "SCAT_CF", "PRNU_CF", "GAINCOR", "FIR_ROT"]


def read_inventory(root):
    rows, bad = [], []
    for f in sorted(Path(root).rglob("*.fits")):
        try:
            h = fits.getheader(f, 0)
        except Exception as e:  # corrupt or partial file
            bad.append({"file": str(f), "error": str(e)[:200]})
            continue
        r = {k: h.get(k) for k in KEYS}
        r.update(file=f.name, obsid=f.parent.name, path=str(f), bytes=f.stat().st_size)
        rows.append(r)
    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["DATE-OBS"])
    df["frame"] = np.select([df.ROI_FF.str.strip() == "ROI", df.NAXIS1 == 2048], ["roi", "full_binned"], "full")
    return df.sort_values("t").reset_index(drop=True), bad


def segments(t, max_gap_s):
    """Contiguous runs of a time series: a gap longer than max_gap_s starts a new run."""
    t = t.sort_values().reset_index(drop=True)
    new = t.diff().dt.total_seconds().fillna(np.inf) > max_gap_s
    seg = new.cumsum()
    g = t.groupby(seg)
    return pd.DataFrame({"start": g.min(), "end": g.max(), "frames": g.size()}).assign(
        hours=lambda s: (s.end - s.start).dt.total_seconds() / 3600).reset_index(drop=True)


def pixel_sample(nb, n):
    out = []
    for i in np.linspace(0, len(nb) - 1, min(n, len(nb))).astype(int):
        r = nb.iloc[i]
        d = fits.getdata(r.path).astype(np.float32)
        yy, xx = np.indices(d.shape)
        rr = np.hypot(xx - (r.CRPIX1 - 1), yy - (r.CRPIX2 - 1)) / r.R_SUN
        disk, off = d[rr < 0.9], d[(rr > 1.1) & (rr < 1.4)]
        # share of the geometric disk that falls on the detector
        on_ccd = (rr <= 1).sum() / (np.pi * r.R_SUN ** 2)
        out.append(dict(t=r.t, disk_median=float(np.median(disk)), disk_p999=float(np.percentile(disk, 99.9)),
                        max=float(d.max()), min=float(d.min()), n_at_min=int((d == d.min()).sum()),
                        offlimb_median=float(np.median(off)), disk_on_ccd=float(on_ccd)))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="outputs/phase0")
    ap.add_argument("--max-gap-s", type=float, default=300)
    ap.add_argument("--pixel-sample", type=int, default=24)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df, bad = read_inventory(a.root)
    df.drop(columns="path").to_csv(f"{a.out}/inventory.csv", index=False)
    nb = df[(df.FTR_NAME == "NB03") & (df.frame == "full_binned")]
    seg = segments(nb.t, a.max_gap_s)
    seg.to_csv(f"{a.out}/segments.csv", index=False)
    px = pixel_sample(nb, a.pixel_sample)
    px.to_csv(f"{a.out}/pixel_sample.csv", index=False)

    cad = nb.t.diff().dt.total_seconds().dropna()
    summary = {
        "files": len(df), "unreadable": bad, "gigabytes": round(df.bytes.sum() / 1e9, 2),
        "span": [str(df.t.min()), str(df.t.max())], "obsids": df.obsid.value_counts().to_dict(),
        "frames_by_type_filter": df.groupby(["frame", "FTR_NAME"]).size().unstack(fill_value=0).to_dict(),
        "nb03_full_binned": {
            "frames": len(nb), "cadence_s_median": float(cad.median()), "cadence_s_p90": float(cad.quantile(.9)),
            "segments": len(seg), "longest_segment_h": float(seg.hours.max()) if len(seg) else 0,
            "crpix1_range": [float(nb.CRPIX1.min()), float(nb.CRPIX1.max())],
            "crpix2_range": [float(nb.CRPIX2.min()), float(nb.CRPIX2.max())],
            "r_sun_px_range": [float(nb.R_SUN.min()), float(nb.R_SUN.max())],
            "exposure_ms": nb.CMD_EXPT.value_counts().to_dict(),
        },
        "pixels": {"disk_median_rel_std": float(px.disk_median.std() / px.disk_median.mean()),
                   "offlimb_median_range": [float(px.offlimb_median.min()), float(px.offlimb_median.max())],
                   "disk_on_ccd_range": [float(px.disk_on_ccd.min()), float(px.disk_on_ccd.max())],
                   "max_range": [float(px["max"].min()), float(px["max"].max())]},
        "flags_nonzero": {k: int((df[k].fillna(0) != 0).sum()) for k in
                          ["SOLX1TR", "SOLX2TR", "HELIOSTR", "NORM_FLR", "PROM_FLR"]},
        "qval_unique": sorted(df.QVAL.dropna().unique().tolist()),
        "calibration": {"flat": df.FLAT_CF.value_counts().to_dict(), "level": df.F_LEVEL.value_counts().to_dict(),
                        "version": df.F_VER.value_counts().to_dict()},
        "roi_programs": df[df.frame == "roi"].groupby("ROI_ID").agg(
            frames=("file", "size"), x1_min=("X1", "min"), x1_max=("X1", "max"), obsid=("obsid", "first")
        ).reset_index().to_dict("records"),
    }
    with open(f"{a.out}/audit.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print(json.dumps({k: summary[k] for k in ["files", "gigabytes", "span", "nb03_full_binned", "pixels"]},
                     indent=1, default=str))
    print(seg.to_string())


if __name__ == "__main__":
    main()
