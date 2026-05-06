# SPDX-License-Identifier: Apache-2.0
# Fork note: Modified by Norbert Laszlo on 2026-05-06 from upstream ContextBench.
# Summary of changes: export patch edit-location metrics.

"""Metrics module."""
from .compute import (
    coverage_precision,
    compute_granularity_metrics,
    compute_trajectory_metrics,
    span_total_bytes,
    span_intersection_bytes
)
from .patch_editloc import (
    compute_patch_editloc,
    compute_patch_to_patch_overlap,
    parse_patch_edit_locations,
)

__all__ = [
    'coverage_precision',
    'compute_granularity_metrics', 
    'compute_trajectory_metrics',
    'span_total_bytes',
    'span_intersection_bytes',
    'compute_patch_editloc',
    'compute_patch_to_patch_overlap',
    'parse_patch_edit_locations',
]
