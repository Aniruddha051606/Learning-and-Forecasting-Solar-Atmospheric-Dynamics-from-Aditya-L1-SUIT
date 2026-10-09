"""The pre-registered verdict logic (scripts/posthoc_report.py) on synthetic evaluation outputs."""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from suitdyn import paths

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def _summary(skills):
    rows = []
    for (m, h), (s, lo, hi) in skills.items():
        rows.append({"region": "disk", "horizon": h, "minutes": h * 1.45, "method": m, "n": 500, "rel_mae": 0.02,
                     "skill_vs_B1avg": s, "lo": lo, "hi": hi, "strongest_baseline": "B1-avg-bgS",
                     "skill_vs_strongest": s, "lo_strongest": lo, "hi_strongest": hi, "rmse_skill_vs_B1avg": s,
                     "rmse_skill_vs_strongest": s, "gradient_corr": 0.9, "is_baseline": False})
    return pd.DataFrame(rows)


def _errors(gain_by_day):
    """Per-sample disk MAE: the ensemble beats B1-avg-bgS by gain_by_day[d] on day d."""
    rng = np.random.default_rng(0)
    rows = []
    for d, gain in enumerate(gain_by_day):
        for k in range(60):
            base = 0.02 * (1 + 0.05 * rng.standard_normal())
            rows.append({"sample": len(rows), "horizon": 20, "run": d, "minutes": 29.0,
                         "t_last": pd.Timestamp("2026-09-26") + pd.Timedelta(days=d, minutes=k),
                         "disk|B1-avg-bgS": base, "disk|unet_bg-ens": base * (1 - gain),
                         "disk|convlstm_bg-ens": base * (1 - gain / 2)})
    return pd.DataFrame(rows)


def test_verdicts(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUT", tmp_path / "outputs")
    rep_mod = importlib.import_module("posthoc_report")
    for ds, test_skill in (("dsA", 0.03), ("dsB", -0.01)):
        p3 = paths.phase3(name=ds)
        (p3 / "eval").mkdir(parents=True)
        (p3 / "eval_test").mkdir(parents=True)
        sk = {(m, h): (test_skill, test_skill - 0.01, test_skill + 0.01) for m in rep_mod.ENSEMBLES for h in (20, 40, 80, 160)}
        _summary(sk).to_csv(p3 / "eval" / "summary.csv", index=False)
        _summary(sk).to_csv(p3 / "eval_test" / "summary.csv", index=False)
        _errors([test_skill, test_skill * 1.2, test_skill * 0.8]).to_parquet(p3 / "eval_test" / "errors.parquet")
        pd.DataFrame([{"control": "frozen", "model_run": "unet_bg_s0", "n": 100, "skill_vs_own_B1": -0.02, "lo": -0.03,
                       "hi": -0.005 if ds == "dsA" else 0.05},
                      {"control": "shuffle", "model_run": "unet_bg_s0", "n": 100, "skill_vs_own_B1": 0.01, "lo": 0.0, "hi": 0.02}]
                     ).to_csv(p3 / "eval" / "controls.csv", index=False)
        pd.DataFrame([{"model_run": "unet_bg_s0", "horizon": h, "fixed_share": 0.05 if ds == "dsA" else 0.4, "corr_with_M": 0.1}
                      for h in (20, 40, 80, 160)]).to_csv(p3 / "eval" / "corotation.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["posthoc_report.py", "--datasets", "dsA,dsB"])
    rep_mod.main()
    rep = json.loads((tmp_path / "outputs" / "tests" / "report.json").read_text())
    a, b = rep["datasets"]["dsA"], rep["datasets"]["dsB"]
    assert all(r["P1"] for r in a["test"]) and not any(r["P1"] for r in b["test"])
    h20 = [r for r in a["test"] if r["horizon"] == 20 and r["method"] == "unet_bg-ens"][0]
    assert h20["P2"] is True and h20["lo_day"] > 0                     # day blocks: still above zero
    assert a["controls"]["C1"] is True and b["controls"]["C1"] is False  # frozen control
    assert a["controls"]["C2"] is True and b["controls"]["C2"] is False  # corotation control
    assert rep["solar"] and not any(r["P3"] for r in rep["solar"])     # skill in one mode only: not solar
    assert (tmp_path / "outputs" / "tests" / "REPORT.md").exists()


def test_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUT", tmp_path / "outputs")
    rep_mod = importlib.import_module("posthoc_report")
    monkeypatch.setattr(sys, "argv", ["posthoc_report.py", "--datasets", "nothing_yet"])
    rep_mod.main()
    rep = json.loads((tmp_path / "outputs" / "tests" / "report.json").read_text())
    assert rep["datasets"]["nothing_yet"]["status"] == {"val": False, "test": False, "flow": False, "cross": False}
    assert "no evaluation yet" in rep["headline"][0]
