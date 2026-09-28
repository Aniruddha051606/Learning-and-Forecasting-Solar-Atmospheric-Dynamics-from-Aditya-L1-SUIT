"""The pipeline runner's planning logic (suitdyn/pipeline.py), in a temporary outputs folder."""
import json

import pytest

from suitdyn import config, paths, pipeline


@pytest.fixture()
def tmp_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUT", tmp_path / "outputs")
    monkeypatch.setattr(config, "DATASET", "c0")
    return tmp_path


def _row(rows, stage):
    return next(r for r in rows if r["stage"] == stage)


def test_fresh_project_runs_everything(tmp_outputs):
    st, fps, rows = pipeline.plan("c0")
    assert all(r["action"] == "run" for r in rows)
    assert _row(rows, "registration")["why"] == "never run"
    assert list(st)[0] == "manifest" and list(st)[-1] == "evaluate"


def test_up_to_date_stage_is_skipped_and_change_reruns(tmp_outputs):
    st, fps, _ = pipeline.plan("c0")
    s = st["registration"]
    for o in s.outputs:
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text("x")
    state = pipeline.state_path("c0", "registration")
    state.write_text(json.dumps({"status": "done", "fingerprint": fps["registration"]}))
    _, _, rows = pipeline.plan("c0")
    assert _row(rows, "registration")["action"] == "skip"
    state.write_text(json.dumps({"status": "done", "fingerprint": "something else"}))
    _, _, rows = pipeline.plan("c0")
    assert _row(rows, "registration")["action"] == "run"
    assert "changed" in _row(rows, "registration")["why"]


def test_upstream_change_propagates(tmp_outputs):
    _, fps_a, _ = pipeline.plan("c0")
    st = pipeline.stages("c0")
    fps_b = {}
    for n, s in st.items():  # recompute with a different upstream fingerprint for 'frames'
        fps_b[n] = "changed" if n == "frames" else pipeline.fingerprint(s, fps_b, "c0", False)
    assert fps_b["registration"] != fps_a["registration"] and fps_b["evaluate"] != fps_a["evaluate"]


def test_selection_and_code_dependencies(tmp_outputs):
    _, _, rows = pipeline.plan("c0", until="sequences")
    sel = [r["stage"] for r in rows if r["action"] == "run"]
    assert sel == ["manifest", "frames", "registration", "sequences"]
    _, _, rows = pipeline.plan("c0", only="train", skip_archive=True)
    assert {r["stage"] for r in rows if r["action"] == "run"} == {r["stage"] for r in rows if r["stage"].startswith("train:")}
    files = {p.name for p in pipeline.code_files("scripts/phase3_train.py")}
    assert {"data.py", "geometry.py", "models.py", "thermal.py", "solar.py"} <= files
    assert "progress.py" not in files and "atomic.py" not in files


def test_smoke_runs_are_kept_apart(tmp_outputs):
    st = pipeline.stages("c0", smoke=True)
    assert all("runs_smoke" in str(o) for n, s in st.items() if n.startswith("train:") for o in s.outputs)
    assert "eval_smoke" in str(st["evaluate"].outputs[0])


def test_only_transient_failures_are_retried(tmp_path):
    code_bug = tmp_path / "a.log"
    code_bug.write_text("Traceback (most recent call last):\n  File \"x.py\", line 1\nTypeError: unsupported operand\n")
    share = tmp_path / "b.log"
    share.write_text("Traceback (most recent call last):\n  File \"x.py\"\nOSError: [WinError 64] The specified network name is no longer available\n")
    killed = tmp_path / "c.log"
    killed.write_text("full-disk: 1274 to process\n")  # no traceback: killed, power cut, native crash
    assert not pipeline.transient(code_bug)
    assert pipeline.transient(share) and pipeline.transient(killed)
