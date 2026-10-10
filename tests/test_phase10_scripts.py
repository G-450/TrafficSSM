"""Tests for the Phase 10 tuning and seed-run scripts: re-running them must skip
finished work and resume unfinished runs instead of starting over."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from st_dssm.result_schema import RunManifest, save_run_manifest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_manifest(path: Path, run_id: str, nll: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest(run_id=run_id, model_name="st_dssm", split_evaluated="val")
    manifest.overall_metrics = {"NLL": nll}
    manifest.mark_complete()
    save_run_manifest(manifest, str(path))


def test_tune_skips_finished_grid_points_and_resumes_the_rest(tmp_path, monkeypatch) -> None:
    tune = _load_script("tune_phase10")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "train.yaml").write_text(
        yaml.safe_dump({"training": {}, "model": {}}), encoding="utf-8"
    )
    out = tmp_path / "experiments" / "tuning"
    # Grid point 0 finished in an earlier invocation with the best NLL.
    _write_manifest(out / "run_0" / "st_dssm-val-seed42-x_manifest.json", "st_dssm-val-seed42-x", nll=1.0)

    calls = []

    def fake_train(**kwargs):
        calls.append(kwargs)
        return {"manifest": SimpleNamespace(overall_metrics={"NLL": 2.0 + len(calls)})}

    monkeypatch.setattr(tune, "train_st_dssm", fake_train)
    monkeypatch.setattr(sys, "argv", ["tune_phase10.py", "--output-dir", str(out)])
    tune.main()

    assert [os.path.basename(c["output_dir"]) for c in calls] == ["run_1", "run_2"]
    assert all(c["resume_path"] == "auto" for c in calls)
    # The finished run's NLL still takes part in selection, and it wins.
    tuned = yaml.safe_load((tmp_path / "configs" / "tuned.yaml").read_text(encoding="utf-8"))
    assert tuned["training"]["learning_rate"] == tune.GRID[0]["learning_rate"]
    assert "seed" not in tuned and "split" not in tuned


def test_run_phase10_skips_finished_seeds_and_resumes(tmp_path, monkeypatch) -> None:
    run_phase10 = _load_script("run_phase10")
    config_path = tmp_path / "tuned.yaml"
    config_path.write_text(yaml.safe_dump({"seeds": [2026, 2027]}), encoding="utf-8")
    out = tmp_path / "normal"
    _write_manifest(out / "st_dssm-test-seed2026-x_manifest.json", "st_dssm-test-seed2026-x", nll=1.0)

    monkeypatch.setattr(
        sys, "argv",
        ["run_phase10.py", "--config", str(config_path), "--output-dir", str(out),
         "--checkpoint-dir", str(tmp_path / "ckpt")],
    )
    with patch.object(run_phase10.subprocess, "run") as mock_run:
        run_phase10.main()

    assert mock_run.call_count == 1
    cmd = mock_run.call_args.args[0]
    assert cmd[:3] == [sys.executable, "-m", "st_dssm.cli.train"]
    assert cmd[cmd.index("--seed") + 1] == "2027"
    assert cmd[cmd.index("--resume") + 1] == "auto"
