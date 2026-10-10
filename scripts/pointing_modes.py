"""Which pointings are in the archive, day by day?

    python scripts/pointing_modes.py                          # from the archive manifest (indexed spans only)
    python scripts/pointing_modes.py --scan --from 2026-09-10 --to 2026-09-28   # read headers on the share
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import atomic, config, manifest, paths, pointing  # noqa: E402

CFG = config.load()


def header_row(path):
    try:
        h = fits.getheader(path, 0)
        return {"file": os.path.basename(path), "t": pd.Timestamp(h.get("DATE-OBS")), "FTR_NAME": h.get("FTR_NAME"),
                "NAXIS1": h.get("NAXIS1"), "ROI_FF": h.get("ROI_FF"), "CRPIX1": h.get("CRPIX1"), "CRPIX2": h.get("CRPIX2")}
    except Exception as e:
        return {"file": os.path.basename(path), "error": str(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true", help="read headers from the share instead of the manifest")
    ap.add_argument("--from", dest="t0", default=None)
    ap.add_argument("--to", dest="t1", default=None)
    a = ap.parse_args()
    lo = pd.Timestamp(a.t0) if a.t0 else pd.Timestamp.min
    hi = pd.Timestamp(a.t1) if a.t1 else pd.Timestamp.max
    if a.scan:
        files = []
        for dp, _, fn in os.walk(CFG["paths"]["raw_root"]):
            for f in fn:
                t = manifest.name_time(f) if f.endswith("NB03.fits") else None
                if t is not None and lo <= t <= hi:
                    files.append(os.path.join(dp, f))
        print(f"reading {len(files)} NB03 headers ...", flush=True)
        with ThreadPoolExecutor(8) as ex:
            df = pd.DataFrame(list(ex.map(header_row, files)))
    else:
        df = pd.read_parquet(paths.archive("manifest.parquet"),
                            columns=["file", "t", "FTR_NAME", "NAXIS1", "ROI_FF", "CRPIX1", "CRPIX2"])
        df = df[(df.t >= lo) & (df.t <= hi)]
    nb = df[(df.FTR_NAME == "NB03") & (df.NAXIS1 == 2048) & df.ROI_FF.astype(str).str.contains("Full")].dropna(
        subset=["CRPIX1", "CRPIX2"]).sort_values("t")
    nb["cluster"] = pointing.clusters(nb.CRPIX1.astype(float).values, nb.CRPIX2.astype(float).values, 2048)
    nb["day"] = nb.t.dt.strftime("%Y-%m-%d")
    tab = nb.groupby(["day", "cluster"]).agg(frames=("file", "size"), first=("t", "min"), last=("t", "max")).reset_index()
    atomic.to_parquet(nb[["file", "t", "day", "cluster", "CRPIX1", "CRPIX2"]], paths.archive("pointing_modes.parquet"))
    tab.to_csv(paths.archive("pointing_modes.csv"), index=False)
    pd.set_option("display.width", 200)
    print(tab.to_string(index=False))
    print("\nclusters overall:\n" + nb.groupby("cluster").agg(frames=("file", "size"), first=("t", "min"),
                                                            last=("t", "max")).to_string())


if __name__ == "__main__":
    main()
