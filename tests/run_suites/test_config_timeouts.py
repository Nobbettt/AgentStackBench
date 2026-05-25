# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from contextbench.run_suites import RunSuiteConfig, build_run_suite_variant

from .helpers import _write_task_inputs


def _base_config_payload(tmp_path, *, variant: dict[str, object]) -> dict[str, object]:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    return {
        "experiment_name": "timeout-config",
        "agent": "codex",
        "base_run": {
            "task_data": str(task_data),
            "task_csv": str(task_csv),
            "output_root": str(tmp_path / "results"),
            "repo_cache": str(tmp_path / "cache"),
            "timeout": 30,
            "runtime_backend": "host",
        },
        "variants": [variant],
        "postprocess": {"convert": False, "evaluate": False, "runtime_backend": "host"},
    }


@pytest.mark.parametrize("timeout", [0, -1])
def test_run_suite_config_rejects_nonpositive_variant_timeout(tmp_path, timeout: int) -> None:
    with pytest.raises(ValueError):
        RunSuiteConfig.model_validate(
            _base_config_payload(
                tmp_path,
                variant={"name": "bad-timeout", "timeout": timeout, "runtime_backend": "host"},
            )
        )


def test_build_run_suite_variant_uses_base_timeout_when_variant_timeout_is_missing(tmp_path) -> None:
    config = RunSuiteConfig.model_validate(
        _base_config_payload(
            tmp_path,
            variant={"name": "baseline", "runtime_backend": "host"},
        )
    )

    effective = build_run_suite_variant(config, config.variants[0])

    assert effective.timeout == 30


def test_build_run_suite_variant_uses_positive_variant_timeout(tmp_path) -> None:
    config = RunSuiteConfig.model_validate(
        _base_config_payload(
            tmp_path,
            variant={"name": "longer", "timeout": 45, "runtime_backend": "host"},
        )
    )

    effective = build_run_suite_variant(config, config.variants[0])

    assert effective.timeout == 45
