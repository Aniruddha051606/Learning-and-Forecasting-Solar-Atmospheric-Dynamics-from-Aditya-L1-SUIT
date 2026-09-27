import hashlib
import os
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Which data set the Phase 2/3 scripts work on (environment variable SUITDYN_DATASET, default v0).
# v0 keeps its original paths; another name <n> uses configs/phase2_<n>.toml, outputs/phase2/sequences_<n>,
# the store <n>, and outputs/phase3_<n>/, so data sets never overwrite each other.
DATASET = os.environ.get("SUITDYN_DATASET", "v0")


def load(path="configs/phase1.toml"):
    p = ROOT / path
    raw = p.read_bytes()
    cfg = tomllib.loads(raw.decode())
    cfg["_meta"] = {"config_path": str(p), "config_sha256": hashlib.sha256(raw).hexdigest(), "git": git_state()}
    return cfg


def load_phase2():
    """Phase 1 settings (paths, limb, qc, register) plus configs/phase2.toml, with both hashes recorded."""
    cfg = load()
    p = ROOT / ("configs/phase2.toml" if DATASET == "v0" else f"configs/phase2_{DATASET}.toml")
    raw = p.read_bytes()
    cfg.update(tomllib.loads(raw.decode()))
    cfg["_meta"]["phase2_config_sha256"] = hashlib.sha256(raw).hexdigest()
    cfg["_meta"]["phase2_config_path"] = str(p)
    cfg["_meta"]["dataset"] = DATASET
    return cfg


def seq_dir():
    """Frame list, splits and windows of the current data set."""
    return ROOT / "outputs" / "phase2" / ("sequences" if DATASET == "v0" else f"sequences_{DATASET}")


def phase3_dir(*parts):
    """Phase 3 products of the current data set (cache, runs, diagnostics, background)."""
    return ROOT / "outputs" / ("phase3" if DATASET == "v0" else f"phase3_{DATASET}") / Path(*parts)


def git_state():
    """Commit hash and whether the tree has uncommitted changes, recorded with every output."""
    try:
        r = subprocess.run(["git", "rev-parse", "--verify", "-q", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        commit = r.stdout.strip() if r.returncode == 0 else ""
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    except OSError:
        return {"commit": None, "dirty": None}
    return {"commit": commit or None, "dirty": dirty}


def out_dir(cfg, *parts):
    d = ROOT / cfg["paths"]["out"] / Path(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d
