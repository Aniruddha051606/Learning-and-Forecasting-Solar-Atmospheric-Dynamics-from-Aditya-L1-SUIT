"""Where every product lives. Scripts ask here; none builds its own output path.

outputs/
  archive/                  FRAME-LOCAL products, valid for any data set; they only grow as data arrive
      manifest.parquet, manifest_meta.json            one row per raw FITS file (header, SHA-256)
      frames_full.parquet, frames_roi.parquet, ...    per-frame measurements (limb fit, QC stats, motion)
  datasets/<name>/          everything that depends on WHICH frames form a data set; built only from
                            frames inside the data set's time span, so adding new days never changes it
      phase1/               registration, frame QC decision, fixed pattern for motion
      calibration/          detector fixed pattern estimated from the training split
      sequences/            frame list, splits, windows, test seal
      stores/               calibrated, registered frames (Zarr)
      phase2/               noise maps, pointing-response correction
      phase3/               frame cache, sample index, background, runs, evaluation, controls
  pipeline/<name>/          runner state (one JSON per stage) and per-stage logs
  logs/progress/            live heartbeats for the dashboard

The data set is config.DATASET (environment variable SUITDYN_DATASET) unless a name is passed.
"""
from pathlib import Path

from . import config

OUT = config.ROOT / "outputs"


def _p(base, parts, make=True):
    """base/parts; with make=True the folder is created (make=False: a pure path, e.g. for planning)."""
    p = base.joinpath(*parts) if parts else base
    if make:
        target = p.parent if p.suffix and p.suffix != ".zarr" else p
        target.mkdir(parents=True, exist_ok=True)
    return p


def archive(*parts, make=True):
    return _p(OUT / "archive", parts, make)


def dataset(*parts, name=None, make=True):
    return _p(OUT / "datasets" / (name or config.DATASET), parts, make)


def phase1(*parts, name=None, make=True):
    return dataset("phase1", *parts, name=name, make=make)


def calibration(*parts, name=None, make=True):
    return dataset("calibration", *parts, name=name, make=make)


def sequences(*parts, name=None, make=True):
    return dataset("sequences", *parts, name=name, make=make)


def stores(*parts, name=None, make=True):
    return dataset("stores", *parts, name=name, make=make)


def phase2(*parts, name=None, make=True):
    return dataset("phase2", *parts, name=name, make=make)


def phase3(*parts, name=None, make=True):
    return dataset("phase3", *parts, name=name, make=make)


def pipeline(*parts, name=None, make=True):
    return _p(OUT / "pipeline" / (name or config.DATASET), parts, make)


def progress():
    return _p(OUT / "logs" / "progress", ())


def smoke():
    """A quick end-to-end check (python -m suitdyn run --smoke): its runs and evaluation are kept apart."""
    import os
    return os.environ.get("SUITDYN_SMOKE") == "1"


def runs(*parts, name=None, make=True):
    return phase3("runs_smoke" if smoke() else "runs", *parts, name=name, make=make)


def evals(*parts, name=None, make=True):
    return phase3("eval_smoke" if smoke() else "eval", *parts, name=name, make=make)
