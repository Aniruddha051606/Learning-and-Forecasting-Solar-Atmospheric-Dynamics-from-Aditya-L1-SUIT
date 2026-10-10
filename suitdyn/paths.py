"""Where every product lives."""

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
    """The model folder: phase3/runs, or phase3/<SUITDYN_RUNS_DIR> to train or score another model set while
    phase3/runs is in use (the v2 retraining, docs/PREREGISTRATION.md Addendum I).
    """
    import os
    return phase3("runs_smoke" if smoke() else os.environ.get("SUITDYN_RUNS_DIR", "runs"), *parts, name=name, make=make)


def evals(*parts, name=None, make=True):
    return phase3("eval_smoke" if smoke() else "eval", *parts, name=name, make=make)
