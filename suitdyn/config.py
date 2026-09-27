"""Configuration: settings files, the current data set, and provenance recorded with every output.

  configs/phase1.toml            archive paths, limb fit, QC, calibration and registration settings
  configs/datasets/<name>.toml   one data set: pointing mode, exclusions, time splits, windows
  configs/phase3.toml            the learning stage: samples, models, training, evaluation, controls

The current data set is the environment variable SUITDYN_DATASET (default v0). Where its products
live is suitdyn/paths.py.
"""
import hashlib
import os
import subprocess
import tomllib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET = os.environ.get("SUITDYN_DATASET", "v0")


def _read(path):
    p = ROOT / path
    raw = p.read_bytes()
    return tomllib.loads(raw.decode()), p, hashlib.sha256(raw).hexdigest()


def load(path="configs/phase1.toml"):
    cfg, p, sha = _read(path)
    cfg["_meta"] = {"config_path": str(p), "config_sha256": sha, "git": git_state()}
    return cfg


def dataset_config_path(name=None):
    return ROOT / "configs" / "datasets" / f"{name or DATASET}.toml"


def load_dataset(name=None):
    """Phase 1 settings plus the data set's settings, with both hashes recorded."""
    name = name or DATASET
    cfg = load()
    ds, p, sha = _read(dataset_config_path(name).relative_to(ROOT))
    cfg.update(ds)
    cfg["_meta"].update(dataset=name, dataset_config_path=str(p), dataset_config_sha256=sha)
    return cfg


load_phase2 = load_dataset  # earlier name


def load_phase3():
    cfg, p, sha = _read("configs/phase3.toml")
    cfg["_meta"] = {"phase3_config_path": str(p), "phase3_config_sha256": sha}
    return cfg


def dataset_span(cfg, margin_h=None):
    """Time span of a data set (earliest split start, latest split end), widened by a margin so that
    registration smoothing and QC statistics near the edges see their neighbours."""
    margin = pd.Timedelta(hours=cfg.get("scope", {}).get("margin_h", 3) if margin_h is None else margin_h)
    starts = [pd.Timestamp(v[0]) for v in cfg["split"].values()]
    ends = [pd.Timestamp(v[1]) for v in cfg["split"].values()]
    return min(starts) - margin, max(ends) + margin


def git_state():
    """Commit hash and whether the tree has uncommitted changes, recorded with every output."""
    try:
        r = subprocess.run(["git", "rev-parse", "--verify", "-q", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        commit = r.stdout.strip() if r.returncode == 0 else ""
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    except OSError:
        return {"commit": None, "dirty": None}
    return {"commit": commit or None, "dirty": dirty}


# ---- earlier helpers, now thin wrappers over suitdyn/paths.py ----------------------------------------
def out_dir(cfg, *parts):
    """The frame-local archive folder (manifest, per-frame measurements)."""
    from . import paths
    return paths.archive(*parts)


def seq_dir():
    from . import paths
    return paths.sequences()


def phase3_dir(*parts):
    from . import paths
    return paths.phase3(*parts)
