"""End to end on the CPU: the real training and evaluation scripts on a tiny synthetic data set (64² frames with a
planted static background). Catches wiring errors in phase3_train.py / phase3_evaluate.py before a long run."""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from suitdyn import config, paths

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from test_ml_data_background import G, _synthetic_cache  # noqa: E402


def _dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUT", tmp_path / "outputs")
    monkeypatch.setattr(config, "DATASET", "smoke")
    monkeypatch.delenv("SUITDYN_SMOKE", raising=False)
    cache = paths.phase3("cache")
    c = (G - 1) / 2
    v, u = np.indices((G, G))
    S_true = (0.5 * (u - c) / 28.75).astype(np.float32)
    _synthetic_cache(cache, S_true)
    idx = pd.read_parquet(cache / f"samples_{G}.parquet")
    idx["set"] = np.where(idx.run <= 3, "train", np.where(idx.run == 4, "holdout", "val"))
    idx["px"], idx["py"] = 1280.0, 600.0
    idx.to_parquet(cache / f"samples_{G}.parquet")
    bg = paths.phase3("background")
    np.savez(bg / f"static_bg_{G}.npz", S=S_true, q_mu=np.linspace(0.3, 1, 10), q=np.ones(10),
             M_H2=np.zeros((G, G), np.float32), M_H4=np.zeros((G, G), np.float32))


def _run(module, argv, monkeypatch):
    mod = importlib.import_module(module)
    monkeypatch.setattr(sys, "argv", [module] + argv)
    mod.main()


def test_train_then_evaluate(tmp_path, monkeypatch):
    _dataset(tmp_path, monkeypatch)
    _run("phase3_train", ["--model", "unet", "--seed", "0", "--inputs", "bg", "--epochs", "3", "--max-batches", "2",
                          "--device", "cpu"], monkeypatch)
    _run("phase3_train", ["--model", "convlstm", "--seed", "0", "--inputs", "plain", "--epochs", "2", "--max-batches", "2",
                          "--device", "cpu"], monkeypatch)
    run = json.loads((paths.runs("unet_bg_s0") / "run.json").read_text())
    assert run["epochs_run"] == 3 and run["best_epoch"] in (1, 2)          # checked every 2nd epoch and the last
    assert run["env"]["numpy"] and run["inputs"] == "bg"
    log = pd.read_csv(paths.runs("unet_bg_s0") / "log.csv")
    assert log.holdout_l1.isna().tolist() == [True, False, False]
    _run("phase3_evaluate", ["--device", "cpu"], monkeypatch)
    ev = paths.evals()
    summ = pd.read_csv(ev / "summary.csv")
    assert {"unet_bg_s0", "convlstm_plain_s0", "B1-avg-bgS", "B1-avg-LDadd"} <= set(summ.method)
    assert {"disk", "ring_inner", "ring_mid"} <= set(summ.region)
    assert summ[["rmse_skill_vs_B1avg", "gradient_corr"]].notna().any().all()
    # the planted static background: the background-aware baseline beats the plain context mean
    d = summ[(summ.region == "disk") & (summ.method == "B1-avg-bgS")]
    assert (d.skill_vs_B1avg > 0).all()
    ctrl = pd.read_csv(ev / "controls.csv")
    assert set(ctrl.control) == {"shuffle", "frozen"}
    rot = pd.read_csv(ev / "corotation.csv")
    assert set(rot.model_run) == {"unet_bg_s0", "convlstm_plain_s0"} and rot.fixed_share.between(0, 1).all()
