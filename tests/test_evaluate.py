# Fork note: Modified by Norbert Laszlo on 2026-04-17 from upstream ContextBench.
# Summary of changes: add regression coverage for symbol-only context, effective file coverage, and EditLoc integrity.

from __future__ import annotations

import pytest
from unittest.mock import patch

from contextbench import evaluate
from contextbench.metrics.compute import compute_trajectory_metrics
from contextbench.parsers.trajectory import Step


def test_tree_sitter_install_command_switches_by_python_version() -> None:
    assert (
        evaluate._tree_sitter_install_command((3, 11))
        == 'pip install "tree-sitter==0.20.4" tree-sitter-languages'
    )
    assert (
        evaluate._tree_sitter_install_command((3, 13))
        == 'pip install "tree-sitter>=0.24,<0.25" tree-sitter-language-pack'
    )


def test_main_uses_tree_sitter_install_hint(monkeypatch, capsys) -> None:
    monkeypatch.setattr("contextbench.extractors.available", lambda: False)
    monkeypatch.setattr(
        evaluate,
        "_tree_sitter_install_command",
        lambda version_info=None: "pip install tree-sitter-test-package",
    )
    monkeypatch.setattr(
        evaluate.sys,
        "argv",
        [
            "contextbench.evaluate",
            "--gold",
            "gold.parquet",
            "--pred",
            "pred.jsonl",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        evaluate.main()

    err = capsys.readouterr().err
    assert exc.value.code == 1
    assert "ERROR: Tree-sitter not available" in err
    assert "pip install tree-sitter-test-package" in err


class _DummyGold:
    def __init__(self, *, patch: str = "") -> None:
        self.repo_url = "https://github.com/example/repo.git"
        self.commit = "abc123"
        self._data = {"patch": patch}

    def files(self):
        return ["src/a.py"]

    def byte_spans(self, repo_dir: str):
        del repo_dir
        return {"src/a.py": [(0, 10)]}

    def line_spans_init(self):
        return {"src/a.py": [(1, 10)]}


def test_evaluate_instance_fails_on_repo_identity_mismatch() -> None:
    gold = _DummyGold()
    pred = {
        "instance_id": "task-1",
        "repo_url": "https://github.com/other/repo.git",
        "commit": "abc123",
        "traj_data": {"pred_steps": [], "pred_files": [], "pred_spans": {}, "pred_symbols": {}},
    }

    result = evaluate.evaluate_instance("task-1", gold, pred, "/tmp")

    assert result["error"] == "repo_identity_mismatch"
    assert result["prediction_repo_url"] == "https://github.com/other/repo.git"
    assert result["gold_repo_url"] == gold.repo_url


def test_evaluate_instance_fails_when_prediction_metadata_missing() -> None:
    gold = _DummyGold()
    pred = {
        "instance_id": "task-1",
        "traj_data": {"pred_steps": [], "pred_files": [], "pred_spans": {}, "pred_symbols": {}},
    }

    result = evaluate.evaluate_instance("task-1", gold, pred, "/tmp")

    assert result["error"] == "missing_prediction_repo_url"


def test_evaluate_instance_fails_on_commit_mismatch() -> None:
    gold = _DummyGold()
    pred = {
        "instance_id": "task-1",
        "repo_url": gold.repo_url,
        "commit": "different",
        "traj_data": {"pred_steps": [], "pred_files": [], "pred_spans": {}, "pred_symbols": {}},
    }

    result = evaluate.evaluate_instance("task-1", gold, pred, "/tmp")

    assert result["error"] == "commit_mismatch"
    assert result["prediction_commit"] == "different"
    assert result["gold_commit"] == gold.commit


def test_evaluate_instance_fails_on_invalid_predicted_context_path(tmp_path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "src").mkdir()
    (repo_dir / "src" / "a.py").write_text("print('ok')\n", encoding="utf-8")
    gold = _DummyGold()
    pred = {
        "instance_id": "task-1",
        "repo_url": gold.repo_url,
        "commit": gold.commit,
        "traj_data": {
            "pred_steps": [{"files": ["../outside.py"], "spans": {}, "symbols": {}}],
            "pred_files": ["../outside.py"],
            "pred_spans": {},
            "pred_symbols": {},
        },
    }

    with patch("contextbench.evaluate.checkout", return_value=str(repo_dir)):
        result = evaluate.evaluate_instance("task-1", gold, pred, "/tmp")

    assert result["error"] == "invalid_predicted_context_path"
    assert result["invalid_paths"] == ["../outside.py"]


def test_evaluate_instance_drops_non_repo_context_paths_without_failing(tmp_path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "src").mkdir()
    (repo_dir / "src" / "a.py").write_text("print('ok')\n", encoding="utf-8")
    gold = _DummyGold()
    pred = {
        "instance_id": "task-1",
        "repo_url": gold.repo_url,
        "commit": gold.commit,
        "traj_data": {
            "pred_steps": [{"files": [".venv/bin", "src/a.py"], "spans": {}, "symbols": {}}],
            "pred_files": [".venv/bin", "src/a.py"],
            "pred_spans": {},
            "pred_symbols": {},
        },
    }

    with patch("contextbench.evaluate.checkout", return_value=str(repo_dir)):
        result = evaluate.evaluate_instance("task-1", gold, pred, "/tmp")

    assert "error" not in result
    assert result["final"]["file"]["pred_size"] == 1
    assert result["final"]["file"]["intersection"] == 1


def test_evaluate_instance_fails_when_no_trajectory_steps(tmp_path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "src").mkdir()
    (repo_dir / "src" / "a.py").write_text("print('ok')\n", encoding="utf-8")
    gold = _DummyGold()
    pred = {
        "instance_id": "task-1",
        "repo_url": gold.repo_url,
        "commit": gold.commit,
        "traj_data": {
            "pred_steps": [],
            "pred_files": ["src/a.py"],
            "pred_spans": {"src/a.py": [{"start": 1, "end": 1}]},
            "pred_symbols": {},
        },
    }

    with patch("contextbench.evaluate.checkout", return_value=str(repo_dir)):
        result = evaluate.evaluate_instance("task-1", gold, pred, "/tmp")

    assert result["error"] == "no_trajectory_steps"


def test_evaluate_instance_accepts_symbol_only_final_context() -> None:
    gold = _DummyGold()
    pred = {
        "instance_id": "task-1",
        "repo_url": gold.repo_url,
        "commit": gold.commit,
        "traj_data": {
            "pred_steps": [{"files": [], "spans": {}, "symbols": {"src/a.py": ["func"]}}],
            "pred_files": [],
            "pred_spans": {},
            "pred_symbols": {"src/a.py": ["func"]},
        },
        "model_patch": "",
    }

    with patch("contextbench.evaluate.checkout", return_value="/tmp"), \
        patch("contextbench.evaluate._filter_step_to_repo", side_effect=lambda step, repo_dir: step), \
        patch("contextbench.evaluate.extract_def_set_in_spans", return_value=set()), \
        patch("contextbench.evaluate.extract_def_set_from_symbol_names", return_value={("src/a.py", "function", 0, 10)}), \
        patch("contextbench.evaluate._step_spans", return_value={}), \
        patch("contextbench.evaluate._step_lines", return_value={}), \
        patch("contextbench.evaluate.compute_granularity_metrics", return_value={"file": {}, "symbol": {}, "span": {}, "line": {}}), \
        patch("contextbench.evaluate.compute_trajectory_metrics", return_value={}):
        result = evaluate.evaluate_instance("task-1", gold, pred, "/tmp")

    assert result.get("error") is None
    assert result["instance_id"] == "task-1"


def test_compute_trajectory_metrics_counts_span_only_step_for_file_coverage() -> None:
    step = Step(files=[], spans=[{"file": "src/a.py", "start_line": 1, "end_line": 2}], symbols={})

    with patch("contextbench.metrics.compute._step_to_byte_spans", return_value={"src/a.py": [(0, 10)]}), \
        patch("contextbench.extractors.extract_def_set_in_spans", return_value=set()):
        result = compute_trajectory_metrics(
            [step],
            {"src/a.py"},
            set(),
            {"src/a.py": [(0, 10)]},
            "/tmp",
            gold_lines={"src/a.py": [(1, 2)]},
        )

    assert result["auc_coverage"]["file"] == pytest.approx(1.0)


def test_compute_trajectory_metrics_counts_symbol_only_step_for_file_coverage() -> None:
    step = Step(files=[], spans=[], symbols={"src/a.py": ["func"]})

    with patch("contextbench.metrics.compute._step_to_byte_spans", return_value={}), \
        patch("contextbench.extractors.extract_def_set_from_symbol_names", return_value=set()):
        result = compute_trajectory_metrics(
            [step],
            {"src/a.py"},
            set(),
            {},
            "/tmp",
            gold_lines={},
        )

    assert result["auc_coverage"]["file"] == pytest.approx(1.0)


def test_evaluate_instance_does_not_use_gold_patch_when_model_patch_missing() -> None:
    gold = _DummyGold(
        patch="""diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -2 +2 @@
-x
+y
""",
    )
    pred = {
        "instance_id": "task-1",
        "repo_url": gold.repo_url,
        "commit": gold.commit,
        "traj_data": {
            "pred_steps": [{"files": ["src/a.py"], "spans": {"src/a.py": [{"start": 1, "end": 3}]}, "symbols": {}}],
            "pred_files": ["src/a.py"],
            "pred_spans": {"src/a.py": [{"start": 1, "end": 3}]},
            "pred_symbols": {},
        },
        "model_patch": "",
    }

    with patch("contextbench.evaluate.checkout", return_value="/tmp"), \
        patch("contextbench.evaluate._filter_step_to_repo", side_effect=lambda step, repo_dir: step), \
        patch("contextbench.evaluate.extract_def_set_in_spans", return_value=set()), \
        patch("contextbench.evaluate._step_spans", return_value={}), \
        patch("contextbench.evaluate._step_lines", return_value={}), \
        patch("contextbench.evaluate.compute_granularity_metrics", return_value={"file": {}, "symbol": {}, "span": {}, "line": {}}), \
        patch("contextbench.evaluate.compute_trajectory_metrics", return_value={}):
        result = evaluate.evaluate_instance("task-1", gold, pred, "/tmp")

    assert "editloc" not in result


def test_evaluate_instance_editloc_recall_uses_gold_size() -> None:
    gold = _DummyGold()
    pred = {
        "instance_id": "task-1",
        "repo_url": gold.repo_url,
        "commit": gold.commit,
        "traj_data": {
            "pred_steps": [{"files": ["src/a.py"], "spans": {"src/a.py": [{"start": 1, "end": 3}]}, "symbols": {}}],
            "pred_files": ["src/a.py"],
            "pred_spans": {"src/a.py": [{"start": 1, "end": 3}]},
            "pred_symbols": {},
        },
        "model_patch": """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -2,2 +2,2 @@
-x
-y
+z
+w
""",
    }

    with patch("contextbench.evaluate.checkout", return_value="/tmp"), \
        patch("contextbench.evaluate._filter_step_to_repo", side_effect=lambda step, repo_dir: step), \
        patch("contextbench.evaluate.extract_def_set_in_spans", return_value=set()), \
        patch("contextbench.evaluate._step_spans", return_value={}), \
        patch("contextbench.evaluate._step_lines", return_value={}), \
        patch("contextbench.evaluate.compute_granularity_metrics", return_value={"file": {}, "symbol": {}, "span": {}, "line": {}}), \
        patch("contextbench.evaluate.compute_trajectory_metrics", return_value={}):
        result = evaluate.evaluate_instance("task-1", gold, pred, "/tmp")

    assert result["editloc"]["intersection"] == 2
    assert result["editloc"]["pred_size"] == 2
    assert result["editloc"]["gold_size"] == 10
    assert result["editloc"]["precision"] == pytest.approx(1.0)
    assert result["editloc"]["recall"] == pytest.approx(0.2)


def test_evaluate_instance_reports_patch_editloc_without_gold_fallback() -> None:
    gold = _DummyGold(
        patch="""diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -2,2 +2,2 @@
-x
-y
+a
+b
""",
    )
    pred = {
        "instance_id": "task-1",
        "repo_url": gold.repo_url,
        "commit": gold.commit,
        "traj_data": {
            "pred_steps": [{"files": ["src/a.py"], "spans": {"src/a.py": [{"start": 1, "end": 3}]}, "symbols": {}}],
            "pred_files": ["src/a.py"],
            "pred_spans": {"src/a.py": [{"start": 1, "end": 3}]},
            "pred_symbols": {},
        },
        "model_patch": """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -3 +3 @@
-y
+b
""",
    }

    with patch("contextbench.evaluate.checkout", return_value="/tmp"), \
        patch("contextbench.evaluate._filter_step_to_repo", side_effect=lambda step, repo_dir: step), \
        patch("contextbench.evaluate.extract_def_set_in_spans", return_value=set()), \
        patch("contextbench.evaluate._step_spans", return_value={}), \
        patch("contextbench.evaluate._step_lines", return_value={}), \
        patch("contextbench.evaluate.compute_granularity_metrics", return_value={"file": {}, "symbol": {}, "span": {}, "line": {}}), \
        patch("contextbench.evaluate.compute_trajectory_metrics", return_value={}):
        result = evaluate.evaluate_instance("task-1", gold, pred, "/tmp")

    assert result["patch_editloc"]["status"] == "available"
    assert result["patch_editloc"]["intersection"] == 1
    assert result["patch_editloc"]["gold_size"] == 2
    assert result["patch_editloc"]["pred_size"] == 1
    assert result["patch_editloc"]["recall"] == pytest.approx(0.5)
    assert result["patch_editloc"]["precision"] == pytest.approx(1.0)


def test_aggregate_results_micro_averages_available_patch_editloc_only() -> None:
    result = evaluate.aggregate_results(
        [
            {
                "instance_id": "a",
                "patch_editloc": {
                    "status": "available",
                    "intersection": 1,
                    "gold_size": 2,
                    "pred_size": 1,
                },
            },
            {
                "instance_id": "b",
                "patch_editloc": {
                    "status": "unavailable",
                    "reason": "missing_model_patch",
                    "intersection": 0,
                    "gold_size": 2,
                    "pred_size": 0,
                },
            },
        ]
    )

    assert result["patch_editloc"]["recall"] == pytest.approx(0.5)
    assert result["patch_editloc"]["precision"] == pytest.approx(1.0)
    assert result["patch_editloc"]["f1"] == pytest.approx(2 / 3)
    assert result["patch_editloc"]["available_instances"] == 1
    assert result["patch_editloc"]["unavailable_instances"] == 1


def test_aggregate_results_keeps_patch_editloc_unavailable_when_no_instances_available() -> None:
    result = evaluate.aggregate_results(
        [
            {
                "instance_id": "a",
                "patch_editloc": {
                    "status": "unavailable",
                    "reason": "missing_model_patch",
                    "intersection": 0,
                    "gold_size": 2,
                    "pred_size": 0,
                },
            },
            {
                "instance_id": "b",
                "patch_editloc": {
                    "status": "unavailable",
                    "reason": "no_model_edit_locations",
                    "intersection": 0,
                    "gold_size": 1,
                    "pred_size": 0,
                },
            },
        ]
    )

    assert result["patch_editloc"]["status"] == "unavailable"
    assert result["patch_editloc"]["reason"] == "no_available_instances"
    assert result["patch_editloc"]["recall"] is None
    assert result["patch_editloc"]["precision"] is None
    assert result["patch_editloc"]["f1"] is None
    assert result["patch_editloc"]["available_instances"] == 0
    assert result["patch_editloc"]["unavailable_instances"] == 2
