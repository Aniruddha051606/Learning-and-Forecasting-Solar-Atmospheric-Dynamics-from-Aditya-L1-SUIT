"""Training store: calibrated, registered NB03 frames in Zarr, with per-frame provenance.

Layout (docs/PHASE1.md §6):
  <store>.zarr/nb03/image   float16 [T, G, G]  chunks (1, G, G), zstd   calibrated counts at the nominal exposure
  <store>.zarr/nb03/mask    uint8   [T, G, G]  chunks (1, G, G), zstd   QC bits after resampling (255 = no source)
  <store>.frames.parquet    one row per T: frame_id (raw file), t, split, run, registration, calibration
  attrs                     grid, r_ref, pipeline commit, config hashes, manifest hash, calibration files

Values are counts scaled to the nominal exposure (EXPOSURE_REF_MS), not counts per second: float16
holds up to 65504 and the encoding ceiling of the raw data is 62767, so no finite value overflows. The exposure is the commanded one (CMD_EXPT);
see calibrate().
"""
import hashlib
import json

import numpy as np

from . import flat, geometry, io, qc, register

EXPOSURE_REF_MS = 300.0


def calibrate(im, row, pattern, pattern_mode, seam_fn=None):
    """Raw frame → calibrated counts at the nominal exposure. Steps are applied in this order and each is
    optional so the effect of each can be measured: fixed pattern, seam, exposure."""
    out = im
    if pattern is not None:
        out = flat.correct(out, pattern, pattern_mode)
    if seam_fn is not None:
        out = seam_fn(out)
    # Commanded exposure, not MEAS_EXP: within runs the NB03 pixel data do not follow MEAS_EXP's ±1 %
    # quantised values (correlation −0.005; dividing by it raises frame-to-frame scatter 0.24 → 0.37 %).
    return out * (EXPOSURE_REF_MS / float(row["CMD_EXPT"]))


def native_mask(im, row, fit, cfg_qc, edge_px):
    r = geometry.r_map_model(im.shape, fit)
    spk, _ = qc.spikes(im, r < cfg_qc["core_r"], cfg_qc["spike_k"], cfg_qc["spike_rel"], cfg_qc["spike_max_area"])
    return qc.pixel_mask(im, r, row["clip_lo"], row["clip_hi"], spk, cfg_qc["seam_px"], cfg_qc["seam_halfwidth"], edge_px)


def process(row, grid, r_ref, cfg_qc, edge_px, pattern=None, pattern_mode="multiplicative", seam_fn=None):
    """One frame → (registered calibrated image float16, registered mask uint8)."""
    raw, _ = io.read(row["path"])
    fit = {"x0": row["reg_x0"], "y0": row["reg_y0"], "R": row["reg_R"], "harm": []}
    mask = native_mask(raw, row, fit, cfg_qc, edge_px)
    im = calibrate(raw, row, pattern, pattern_mode, seam_fn)
    im = np.where((mask & (qc.CLIP_LO | qc.CLIP_HI | qc.SPIKE)) > 0, np.nan, im)
    A, b = register.transform(row["reg_x0"], row["reg_y0"], row["reg_R"], row["CROTA2"], grid, r_ref)
    reg = register.apply(np.nan_to_num(im, nan=0.0), A, b, grid)
    bad = register.apply(np.isnan(im).astype(np.float32), A, b, grid, order=1, cval=1.0) > 0.01
    reg[bad] = np.nan
    m = register.apply_mask(mask, A, b, grid)
    return reg.astype(np.float16), m


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def provenance(cfg, extra):
    return json.loads(json.dumps({"config_meta": cfg["_meta"], **extra}, default=str))
