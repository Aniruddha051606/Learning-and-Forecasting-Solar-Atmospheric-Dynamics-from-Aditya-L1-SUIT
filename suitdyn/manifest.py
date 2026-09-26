"""Raw-data manifest: one row per SUIT FITS file, with its full header, checksum and derived labels.

The manifest is what makes the processed data reproducible: every derived product points back to a
file name and SHA-256, so the exact raw input can be re-downloaded from PRADAN and verified.
"""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from .filters import FILTERS

# Header keys promoted to typed columns; the complete header is kept too, as JSON.
COLUMNS = ["DATE-OBS", "FTR_NAME", "NAXIS1", "NAXIS2", "BITPIX", "BZERO", "BSCALE", "IMG_TYPE", "OBS_MODE",
           "ROI_FF", "BIN_EN", "ROI_ID", "CMD_EXPT", "MEAS_EXP", "IMGTRID", "QVAL", "QDESC", "F_LEVEL", "F_VER",
           "CTYPE1", "CDELT1", "CDELT2", "CRPIX1", "CRPIX2", "CROTA2", "P_ANGLE", "ROLL", "R_SUN", "RSUN_OBS",
           "DSUN_OBS", "HGLT_OBS", "HGLN_OBS", "CRLT_OBS", "X1", "Y1", "X2", "Y2", "SOLX1TR", "SOLX2TR",
           "HELIOSTR", "FLR_TRIG", "NORM_FLR", "PROM_FLR", "FLR_ED", "FLAT_CF", "SCAT_CF", "PRNU_CF", "DIST_CF",
           "PSF_CF", "NSPIKES", "GAINCOR", "FIR_ROT"]


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def parse_name(name):
    """The last 8 characters of a SUIT file name: image type, program ID, size code, filter (User Manual §1.3)."""
    code = Path(name).stem[-8:]
    return {"name_img_type": code[0], "name_prog_id": code[1:3], "name_size_code": code[3], "name_filter": code[4:]}


def read_row(path):
    path = Path(path)
    try:
        h = fits.getheader(path, 0)
    except Exception as e:
        return {"file": path.name, "path": str(path), "read_error": str(e)[:300]}
    row = {k: h.get(k) for k in COLUMNS}
    row.update(file=path.name, path=str(path), obsid=path.parent.name, bytes=path.stat().st_size,
               sha256=sha256(path), read_error=None,
               header_json=json.dumps({k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                                       for k, v in h.items() if k not in ("COMMENT", "HISTORY", "")}))
    row.update(parse_name(path.name))
    return row


def frame_type(df):
    roi = df["ROI_FF"].astype(str).str.strip().str.upper() == "ROI"
    binned = df["BIN_EN"].astype(str).str.strip().str.lower() == "enable"
    return np.select([roi, binned], ["roi", "full_binned"], "full")


def build(raw_root, workers=8):
    files = sorted(Path(raw_root).rglob("*.fits"))
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(read_row, files))
    df = pd.DataFrame(rows)
    ok = df["read_error"].isna()
    df.loc[ok, "t"] = pd.to_datetime(df.loc[ok, "DATE-OBS"])
    df["frame"] = None
    df.loc[ok, "frame"] = frame_type(df[ok])
    df["wavelength_nm"] = df["FTR_NAME"].map(lambda f: FILTERS.get(f, (np.nan,))[0])
    # Clipping limits of the integer encoding: a pixel at either value was not measured, it was clipped.
    df["clip_lo"] = df["BZERO"] + df["BSCALE"] * -32768
    df["clip_hi"] = df["BZERO"] + df["BSCALE"] * 32767
    df["filter_name_matches_header"] = df["name_filter"] == df["FTR_NAME"]
    return df.sort_values("t", na_position="last").reset_index(drop=True)
