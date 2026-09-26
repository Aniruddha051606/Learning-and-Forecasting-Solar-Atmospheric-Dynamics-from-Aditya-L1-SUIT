import hashlib
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path="configs/phase1.toml"):
    p = ROOT / path
    raw = p.read_bytes()
    cfg = tomllib.loads(raw.decode())
    cfg["_meta"] = {"config_path": str(p), "config_sha256": hashlib.sha256(raw).hexdigest(), "git": git_state()}
    return cfg


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
