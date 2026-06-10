# SPDX-License-Identifier: Apache-2.0
# Fork note: Modified by Norbert Laszlo on 2026-06-09 from upstream ContextBench.
# Summary of changes: add regression coverage for symbol-only context, effective file coverage, empty trajectories, and EditLoc integrity.

from __future__ import annotations

import pytest
from unittest.mock import patch

from contextbench import evaluate
from contextbench.metrics.compute import compute_trajectory_metrics
from contextbench.parsers.gold import Gold
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


def test_gold_line_spans_includes_init_and_add_context() -> None:
    gold = Gold(
        {
            "inst_id": "task-1",
            "init_ctx": [{"file": "src/a.py", "start_line": 1, "end_line": 3}],
            "add_ctx": [{"file": "src/a.py", "start_line": 5, "end_line": 7}],
        }
    )

    assert gold.line_spans() == {"src/a.py": [(1, 3), (5, 7)]}
    assert gold.line_spans_init() == {"src/a.py": [(1, 3)]}


def test_aggregate_results_uses_macro_final_context_metrics() -> None:
    rows = [
        {
            "final": {
                "file": {"intersection": 1, "gold_size": 1, "pred_size": 1},
                "symbol": {"intersection": 1, "gold_size": 1, "pred_size": 1},
                "span": {"intersection": 1, "gold_size": 1, "pred_size": 1},
                "line": {"intersection": 1, "gold_size": 1, "pred_size": 1},
            },
            "trajectory": {"auc_coverage": {}, "redundancy": {}},
        },
        {
            "final": {
                "file": {"intersection": 1, "gold_size": 1, "pred_size": 100},
                "symbol": {"intersection": 1, "gold_size": 1, "pred_size": 100},
                "span": {"intersection": 1, "gold_size": 1, "pred_size": 100},
                "line": {"intersection": 1, "gold_size": 1, "pred_size": 100},
            },
            "trajectory": {"auc_coverage": {}, "redundancy": {}},
        },
    ]

    aggregate = evaluate.aggregate_results(rows)

    assert aggregate["final_file"]["aggregation"] == "macro"
    assert aggregate["final_file"]["coverage"] == 1.0
    assert aggregate["final_file"]["precision"] == pytest.approx(0.505)
    assert aggregate["final_file"]["f1"] == pytest.approx((1.0 + (2 * 1.0 * 0.01 / 1.01)) / 2)
    assert aggregate["pooled_final_file"]["precision"] == pytest.approx(2 / 101)


def test_aggregate_results_excludes_empty_gold_instances_from_macro_average() -> None:
    rows = [
        {
            "final": {
                "file": {"intersection": 1, "gold_size": 2, "pred_size": 2},
            },
            "trajectory": {"auc_coverage": {}, "redundancy": {}},
        },
        {
            # No gold at this granularity: 0/0 would score a perfect 1.0 and
            # must not be counted in the macro mean.
            "final": {
                "file": {"intersection": 0, "gold_size": 0, "pred_size": 0},
            },
            "trajectory": {"auc_coverage": {}, "redundancy": {}},
        },
    ]

    aggregate = evaluate.aggregate_results(rows)

    assert aggregate["final_file"]["coverage"] == pytest.approx(0.5)
    assert aggregate["final_file"]["precision"] == pytest.approx(0.5)
    assert aggregate["final_file"]["f1"] == pytest.approx(0.5)
    assert aggregate["final_file"]["num_instances"] == 1


def test_aggregate_results_omits_level_when_no_instance_has_gold() -> None:
    rows = [
        {
            "final": {
                "file": {"intersection": 1, "gold_size": 1, "pred_size": 1},
                "symbol": {"intersection": 0, "gold_size": 0, "pred_size": 3},
            },
            "trajectory": {"auc_coverage": {}, "redundancy": {}},
        },
    ]

    aggregate = evaluate.aggregate_results(rows)

    assert "final_symbol" not in aggregate
    assert aggregate["final_file"]["f1"] == pytest.approx(1.0)


def test_aggregate_results_excludes_missing_trajectory_levels() -> None:
    aggregate = evaluate.aggregate_results(
        [
            {
                "final": {
                    "file": {"intersection": 1, "gold_size": 1, "pred_size": 1},
                },
                "trajectory": {
                    "auc_coverage": {"line": 0.8},
                    "redundancy": {"line": 0.2},
                },
            },
            {
                "final": {
                    "file": {"intersection": 1, "gold_size": 1, "pred_size": 1},
                },
                "trajectory": {
                    "auc_coverage": {"file": 0.9},
                    "redundancy": {"file": 0.1},
                },
            },
        ]
    )

    assert aggregate["traj_auc_line"] == pytest.approx(0.8)
    assert aggregate["traj_redundancy_line"] == pytest.approx(0.2)
    assert aggregate["traj_auc_file"] == pytest.approx(0.9)
    assert aggregate["traj_redundancy_file"] == pytest.approx(0.1)


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

    def line_spans(self):
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


def test_evaluate_instance_passes_workspace_isolation_to_checkout(tmp_path) -> None:
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
            "pred_spans": {},
            "pred_symbols": {},
        },
    }

    with patch("contextbench.evaluate.checkout", return_value=str(repo_dir)) as checkout:
        evaluate.evaluate_instance(
            "task-1",
            gold,
            pred,
            "/tmp/cache",
            workspace_key="suite-baseline-eval",
            tmp_root="/tmp/worktrees",
        )

    checkout.assert_called_once_with(
        gold.repo_url,
        gold.commit,
        "/tmp/cache",
        workspace_key="suite-baseline-eval",
        tmp_root="/tmp/worktrees",
    )


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


def test_evaluate_instance_counts_missing_safe_final_paths_as_file_false_positives(tmp_path) -> None:
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
    assert result["final"]["file"]["pred_size"] == 2
    assert result["final"]["file"]["intersection"] == 1
    assert result["predicted_context_path_diagnostics"] == {
        "missing_final_paths": [".venv/bin"],
        "missing_trajectory_paths": [".venv/bin"],
        "missing_final_path_count": 1,
        "missing_trajectory_path_count": 1,
    }


def test_evaluate_instance_scores_empty_trajectory_when_no_steps(tmp_path) -> None:
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

    assert "error" not in result
    assert result["empty_trajectory"] is True
    assert result["num_steps"] == 0
    assert result["trajectory"]["steps"] == []


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
