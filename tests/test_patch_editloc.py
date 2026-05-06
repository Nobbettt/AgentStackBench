
from __future__ import annotations

import pytest

from contextbench.metrics.patch_editloc import (
    compute_patch_editloc,
    compute_patch_to_patch_overlap,
    parse_patch_edit_locations,
)


def test_parse_patch_edit_locations_extracts_replaced_old_lines() -> None:
    patch = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -10,3 +10,3 @@
 keep
-old_a
-old_b
+new_a
+new_b
 keep
"""

    assert parse_patch_edit_locations(patch) == {"src/a.py": [(11, 12)]}


def test_parse_patch_edit_locations_extracts_insertion_anchor() -> None:
    patch = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -20,0 +21,2 @@
+new_a
+new_b
"""

    assert parse_patch_edit_locations(patch) == {"src/a.py": [(20, 20)]}


def test_parse_patch_edit_locations_handles_new_file_as_first_line_anchor() -> None:
    patch = """diff --git a/src/new.py b/src/new.py
--- /dev/null
+++ b/src/new.py
@@ -0,0 +1,2 @@
+alpha
+beta
"""

    assert parse_patch_edit_locations(patch) == {"src/new.py": [(1, 1)]}


def test_compute_patch_editloc_reports_overlap_against_gold() -> None:
    gold_patch = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -10,3 +10,3 @@
-old_a
-old_b
-old_c
+new_a
+new_b
+new_c
"""
    model_patch = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -11,2 +11,2 @@
-old_b
-old_c
+new_b
+new_c
"""

    result = compute_patch_editloc(gold_patch, model_patch)

    assert result["status"] == "available"
    assert result["intersection"] == 2
    assert result["gold_size"] == 3
    assert result["pred_size"] == 2
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["precision"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(0.8)


def test_compute_patch_editloc_marks_missing_model_patch_unavailable() -> None:
    gold_patch = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -2 +2 @@
-old
+new
"""

    result = compute_patch_editloc(gold_patch, "")

    assert result["status"] == "unavailable"
    assert result["reason"] == "missing_model_patch"
    assert result["gold_size"] == 1
    assert result["pred_size"] == 0
    assert result["recall"] is None


def test_compute_patch_editloc_marks_unparseable_model_patch_unavailable() -> None:
    gold_patch = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -2 +2 @@
-old
+new
"""

    result = compute_patch_editloc(gold_patch, "not a unified diff")

    assert result["status"] == "unavailable"
    assert result["reason"] == "no_model_edit_locations"
    assert result["gold_size"] == 1
    assert result["pred_size"] == 0
    assert result["recall"] is None


def test_compute_patch_to_patch_overlap_is_symmetric_diagnostic() -> None:
    left_patch = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -3,2 +3,2 @@
-a
-b
+x
+y
"""
    right_patch = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -4,2 +4,2 @@
-b
-c
+y
+z
"""

    result = compute_patch_to_patch_overlap(left_patch, right_patch)

    assert result["status"] == "available"
    assert result["intersection"] == 1
    assert result["gold_size"] == 2
    assert result["pred_size"] == 2
    assert result["recall"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(0.5)
