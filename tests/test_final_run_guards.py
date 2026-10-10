"""Guards added before the final run: pointing clusters, split embargo, sample index, one-time test reads,
pointing-grouped backgrounds, evaluation metrics, environment record.
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from suitdyn import config, pointing, sequences
from suitdyn.ml import background as bgm
from suitdyn.ml import evalguard, samples


def test_pointing_clusters_separate_pointings_but_keep_the_oscillation_together():
    rng = np.random.default_rng(0)
    t = np.arange(400)
    x = np.r_[1024 + 10 * np.sin(t[:200] / 30), 1280 + 10 * np.sin(t[200:] / 30)] + rng.normal(0, 1.5, 400)
    y = np.r_[1000 + 10 * np.cos(t[:200] / 30), 600 + 10 * np.cos(t[200:] / 30)]
    lab = pointing.clusters(x, y, 2048)
    assert len(set(lab)) == 2
    assert lab[0].startswith("centred@") and lab[-1].startswith("offset@1280,600")
    assert len(set(lab[:200])) == 1 and len(set(lab[200:])) == 1
    assert list(pointing.mode([1024, 1300], [1024, 600], 2048)) == ["centred", "offset"]


def test_embargo_drops_the_start_of_each_later_split():
    t = pd.date_range("2026-09-20 00:00", periods=48, freq="1h")
    split = np.array(["train"] * 20 + ["val"] * 14 + ["test"] * 14)
    drop = sequences.embargo(t, split, 4)
    # val starts 1 h after the last train frame: its first 3 frames fall inside 4 h; same for test
    assert drop.tolist() == [False] * 20 + [True] * 3 + [False] * 11 + [True] * 3 + [False] * 11
    assert not sequences.embargo(t, split, 0).any()


def _toy(n=40):
    seqf = pd.DataFrame({"frame_id": [f"f{i}" for i in range(n)],
                         "t": pd.date_range("2026-09-20", periods=n, freq="87s"), "run": [1] * (n // 2) + [2] * (n // 2)})
    store = pd.DataFrame({"frame_id": seqf.frame_id, "store_index": np.arange(n)[::-1], "reg_x0": 1280.0 + np.arange(n),
                          "reg_y0": 600.0})
    man = pd.Series(7.0, index=seqf.frame_id)
    return seqf, store, man


def test_sample_index_and_holdout_embargo():
    seqf, store, man = _toy()
    win = pd.DataFrame({"first": [0, 1, 25], "last": [4, 5, 29], "target": [9, 10, 34], "horizon": [5, 5, 5]})
    idx = samples.build(win, seqf, store, man, ["train", "train", "holdout"])
    assert list(idx.ctx.iloc[0]) == [39, 38, 37, 36, 35] and idx.tgt.iloc[0] == 30   # store order, not position
    assert np.allclose(idx.dt_context_s.iloc[0], [9 * 87, 8 * 87, 7 * 87, 6 * 87, 5 * 87])
    assert idx.px.iloc[0] == 1289.0
    kept = samples.embargo_before(idx, "holdout", "train", hours=0.5)
    assert len(kept) == 1 and kept.set.iloc[0] == "holdout"   # both train targets are < 30 min before the hold-out
    assert len(samples.embargo_before(idx, "holdout", "train", hours=0.0)) == 3


def test_test_reads_are_tied_to_the_model_set(tmp_path):
    rec = tmp_path / "test_reads.json"
    fp1 = evalguard.models_fingerprint([("unet_bg_s0", "aa")], {"phase3": "x"})
    fp2 = evalguard.models_fingerprint([("unet_bg_s0", "bb")], {"phase3": "x"})
    assert evalguard.authorize(rec, fp1) is False
    evalguard.record(rec, fp1, [("unet_bg_s0", "aa")], "final", False)
    assert evalguard.authorize(rec, fp1) is False                       # same models again: allowed
    with pytest.raises(PermissionError):
        evalguard.authorize(rec, fp2)                                   # other models: refused
    assert evalguard.authorize(rec, fp2, allow_new_models=True) is True
    evalguard.record(rec, fp2, [("unet_bg_s0", "bb")], "override", True)
    log = json.loads(rec.read_text())
    assert [r["models_changed_after_first_read"] for r in log["reads"]] == [False, True]


def test_pointing_groups_and_shrinkage():
    px = np.r_[np.full(300, 1280.0), np.full(300, 1330.0), np.full(20, 1400.0)]
    py = np.full(len(px), 600.0)
    g, cen = bgm.pointing_groups(px, py, bin_px=12, min_pairs=150)
    assert len(cen) == 2 and set(g[-20:]) == {1}                         # the small bin joins the nearest big one
    g1, cen1 = bgm.pointing_groups(np.full(50, 1280.0) + np.arange(50) * 0.1, np.full(50, 600.0), 12, 10)
    assert len(cen1) == 1                                                # spread below one bin: one group
    # the shrinkage prior: with no data rows the solution is the prior
    G = 16
    pos = bgm.disk_index(G, 7.0, 0.95)
    prior = np.random.default_rng(1).normal(size=(G, G)).astype(np.float32)
    S, _, _ = bgm.solve({}, {}, pos, lam=1e-9, prior=prior, shrink=100.0)
    m = pos >= 0
    assert np.abs(S[m] - prior[m]).max() < 1e-3


def test_bank_uses_the_background_of_each_samples_pointing_group(tmp_path):
    sys.path.insert(0, str(Path(__file__).parent))
    from test_ml_data_background import G, _synthetic_cache
    from suitdyn.ml import data
    _synthetic_cache(tmp_path, np.zeros((G, G)))
    idx = pd.read_parquet(tmp_path / f"samples_{G}.parquet")
    idx["px"] = np.where(np.arange(len(idx)) % 2 == 0, 1280.0, 1330.0)
    idx["py"] = 600.0
    idx.to_parquet(tmp_path / f"samples_{G}.parquet")
    bank = data.Bank(tmp_path, "cpu")
    S2 = np.stack([np.zeros((G, G)), np.full((G, G), 0.5)]).astype(np.float32)
    bank.set_background_groups(S2, np.array([[1280.0, 600.0], [1330.0, 600.0]]))
    assert list(bank.group[:4]) == [0, 1, 0, 1]
    b = bank.batch(np.array([0, 1]), "bg")
    assert float(b["S"][0].abs().max()) == 0.0 and abs(float(b["S"][1].mean()) - 0.5) < 1e-6
    extra = idx.iloc[:3].assign(set="test")
    ids = bank.add_samples(extra)
    assert list(ids) == [len(idx), len(idx) + 1, len(idx) + 2] and len(bank.group) == len(idx) + 3


def test_evaluation_metrics():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    ev = importlib.import_module("phase3_evaluate")
    y = torch.rand(2, 16, 16)
    m = torch.ones(2, 16, 16, dtype=torch.bool)
    mae, rmse, gc, n = ev.masked_metrics(y + 0.1, y, m)
    assert torch.allclose(mae, torch.full((2,), 0.1), atol=1e-5) and torch.allclose(rmse, torch.full((2,), 0.1), atol=1e-5)
    assert torch.allclose(gc, torch.ones(2), atol=1e-5)                 # a constant offset keeps the structure
    _, _, gc_neg, _ = ev.masked_metrics(-y, y, m)
    assert torch.allclose(gc_neg, -torch.ones(2), atol=1e-5)
    assert int(n[0]) == 256


def test_environment_is_recorded():
    env = config.load()["_meta"]["env"]
    assert env["python"] and env["numpy"] and "torch" in env
