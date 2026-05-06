
"""Run suite orchestration."""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from ..artifact_sanitization import (
    SanitizationContext,
    assert_no_private_paths,
    assert_paths_have_no_private_artifacts,
    sanitize_artifact_tree,
    sanitize_artifact_tree_in_place,
    sanitize_json_value,
)
from ..coding_agents.constants import DEFAULT_SUBSET_CSV
from ..coding_agents.files import append_jsonl, ensure_dir, read_json, safe_path_component, write_json
from ..coding_agents.runtime import run_coding_agent_task
from ..coding_agents.task_data import count_task_rows, load_tasks
from ..core.repo import remove_worktree
from ..extractors import available as treesitter_available
from ..parsers import GoldLoader
from .config import build_run_suite_variant
from .env_files import read_env_file, redact_secrets
from .helpers import (
    flatten_metrics,
    record_is_resume_complete,
    stable_json_hash,
    task_key,
    task_record_path,
    utc_now,
)
from .postprocess import (
    _docker_available,
    _docker_host_socket_path,
    _docker_image_available,
    _docker_image_id,
    _postprocess_image_supports_evaluation,
    _run_resolution_command,
    convert_records_to_jsonl,
    describe_resolution_backend_support,
    evaluate_prediction_file,
    evaluate_resolution_for_suite,
)
from .types import EffectiveVariantConfig, RunSuiteConfig

_POSTPROCESS_FINGERPRINT_VERSION = 4


@dataclass
class _PreparedVariant:
    variant: EffectiveVariantConfig
    entry: dict[str, object]
    raw_root: Path
    task_results_path: Path
    pred_path: Path
    eval_results_path: Path
    eval_summary_path: Path
    resolution_summary_path: Path
    started_monotonic: float
    postprocess_failed: bool = False


class RunSuiteRunner:
    def __init__(
        self,
        config: RunSuiteConfig,
        *,
        max_workers: int | None = None,
        resume: bool = False,
        skip_convert: bool = False,
        skip_evaluate: bool = False,
        skip_resolve: bool = False,
        resume_resolution: bool = False,
    ) -> None:
        self.config = config
        self.resume = resume
        self.skip_convert = skip_convert
        self.skip_evaluate = skip_evaluate
        self.skip_resolve = skip_resolve
        self.resume_resolution = resume_resolution
        worker_cap = max_workers if max_workers is not None else config.parallelism.max_workers
        self.max_workers = max(1, int(worker_cap))
        self.experiment_dir = config.base_run.output_root / safe_path_component(config.experiment_name)
        self.manifest_path = self.experiment_dir / "manifest.json"
        self.summary_json_path = self.experiment_dir / "summary.json"
        self.summary_csv_path = self.experiment_dir / "summary.csv"
        self.experiment_config_path = self.experiment_dir / "experiment.json"
        self.public_artifacts_dir = self.experiment_dir / "public-artifacts"
        self._run_invocation_key = safe_path_component(f"{time.time_ns()}")
        self.postprocess_runtime_metadata: dict[str, object] = {
            "backend": self.config.postprocess.runtime_backend,
        }
        self._validate_postprocess_environment()

    def _validate_postprocess_environment(self) -> None:
        postprocess_runtime_needed = (
            (self.config.postprocess.convert and not self.skip_convert)
            or (self.config.postprocess.evaluate and not self.skip_evaluate)
        )
        if not postprocess_runtime_needed:
            return
        if self.config.postprocess.runtime_backend == "docker":
            if not _docker_available():
                raise RuntimeError("Docker is required for postprocess.runtime_backend='docker'.")
            if not _docker_image_available(self.config.postprocess.runtime_image):
                raise RuntimeError(
                    "Postprocess Docker image is not available: "
                    f"{self.config.postprocess.runtime_image}. Run 'python3 -m contextbench.run_suites_setup postprocess-image'."
                )
            image_id = _docker_image_id(self.config.postprocess.runtime_image)
            if not image_id:
                raise RuntimeError(
                    "Could not determine the immutable image id for postprocess Docker image: "
                    f"{self.config.postprocess.runtime_image}."
                )
            self.postprocess_runtime_metadata = {
                "backend": "docker",
                "image": self.config.postprocess.runtime_image,
                "image_id": image_id,
            }
            if self.config.postprocess.evaluate and not self.skip_evaluate:
                supports_evaluation, detail = _postprocess_image_supports_evaluation(self.config.postprocess.runtime_image)
                if not supports_evaluation:
                    raise RuntimeError(
                        "Postprocess Docker image is missing required evaluation parsers: "
                        f"{self.config.postprocess.runtime_image}. "
                        "Run 'python3 -m contextbench.run_suites_setup postprocess-image --force'. "
                        f"Details: {detail}"
                    )
            return
        if self.skip_evaluate or not self.config.postprocess.evaluate:
            pass
        else:
            if not treesitter_available():
                raise RuntimeError(
                    "Tree-sitter is not available for evaluation. "
                    "Install the declared dependencies for this Python version before running a suite with evaluation enabled."
                )

    @staticmethod
    def _resume_compatible_effective_config(
        previous_config: object,
        current_config: dict[str, object],
    ) -> bool:
        if not isinstance(previous_config, dict):
            return False
        previous = dict(previous_config)
        current = dict(current_config)
        previous.pop("limit", None)
        current.pop("limit", None)
        return previous == current

    def _load_tasks(self) -> tuple[list[dict[str, object]], dict[str, object]]:
        base = self.config.base_run
        subset_csv = base.subset_csv or base.task_csv
        source_count = count_task_rows(base.task_data)
        tasks = load_tasks(
            base.task_data,
            subset_csv=subset_csv,
            bench_filter=base.bench,
            instance_filter=base.instances,
            limit=base.limit,
        )
        if not tasks:
            raise RuntimeError("No tasks matched the configured task selection")

        task_index = [
            {
                "bench": task.get("bench"),
                "instance_id": task.get("instance_id"),
                "original_inst_id": task.get("original_inst_id"),
                "repo_url": task.get("repo_url"),
                "commit": task.get("commit"),
            }
            for task in tasks
        ]
        bench_counts = Counter(str(task.get("bench") or "Unknown") for task in tasks)
        task_set = {
            "count": len(tasks),
            "source_count": source_count,
            "selection_kind": self._task_selection_kind(source_count=source_count, selected_count=len(tasks)),
            "hash": stable_json_hash(task_index),
            "bench_counts": dict(sorted(bench_counts.items())),
            "task_ids": [task_key(task) for task in tasks],
        }
        if subset_csv:
            task_set["selection_path"] = str(subset_csv)
        return tasks, task_set

    def _task_selection_kind(self, *, source_count: int, selected_count: int) -> str:
        base = self.config.base_run
        if not (base.subset_csv or base.task_csv or base.bench or base.instances or base.limit > 0) and selected_count == source_count:
            return "full_dataset"
        if (base.subset_csv or base.task_csv) and selected_count < source_count:
            subset_path = base.subset_csv or base.task_csv
            is_default_representative_subset = False
            if subset_path is not None:
                try:
                    is_default_representative_subset = Path(subset_path).resolve() == DEFAULT_SUBSET_CSV.resolve()
                except Exception:
                    is_default_representative_subset = False
            return "representative_subset" if is_default_representative_subset else "configured_subset"
        if base.bench or base.instances:
            return "filtered_selection"
        if base.limit > 0:
            return "limited_selection"
        return "configured_selection"

    def _write_failure_proof(self, *, name: str, payload: dict[str, object]) -> Path:
        ensure_dir(self.experiment_dir)
        path = self.experiment_dir / f"{safe_path_component(name)}.failure.json"
        write_json(path, {"failure": name, "created_at": utc_now(), **payload})
        return path

    def _validate_preflight(self, tasks: list[dict[str, object]], variants: list[EffectiveVariantConfig]) -> None:
        failures: list[dict[str, object]] = []

        full_markers = " ".join(
            str(item or "")
            for item in (self.config.experiment_name, self.config.description)
        ).lower()
        if "full" in full_markers and self.config.base_run.limit > 0:
            failures.append(
                {
                    "kind": "limited_full_suite_config",
                    "limit": self.config.base_run.limit,
                    "message": "Run-suite name/description indicates a full run, but base_run.limit is nonzero.",
                }
            )
        if "full" in full_markers:
            base = self.config.base_run
            selectors: dict[str, object] = {}
            if base.task_csv is not None:
                selectors["task_csv"] = str(base.task_csv)
            if base.subset_csv is not None:
                selectors["subset_csv"] = str(base.subset_csv)
            if base.bench:
                selectors["bench"] = base.bench
            if base.instances:
                selectors["instances"] = base.instances
            if selectors:
                failures.append(
                    {
                        "kind": "selected_full_suite_config",
                        "selectors": selectors,
                        "message": (
                            "Run-suite name/description indicates a full run, but task selection filters are configured. "
                            "Use task_csv=null, subset_csv=null, bench=null, instances=null, and limit=0 for the full dataset."
                        ),
                    }
                )

        gold_loader = None
        if self.config.postprocess.evaluate and not self.skip_evaluate:
            try:
                gold_loader = GoldLoader(str(self.config.postprocess.gold_path))
            except Exception as exc:
                failures.append(
                    {
                        "kind": "gold_load_failed",
                        "path": str(self.config.postprocess.gold_path),
                        "error": str(exc),
                    }
                )

        if gold_loader is not None:
            missing_gold: list[str] = []
            missing_gold_repo_or_commit: list[str] = []
            for task in tasks:
                instance_id = str(task.get("instance_id") or task.get("original_inst_id") or "").strip()
                gold = gold_loader.get(instance_id)
                if gold is None:
                    missing_gold.append(instance_id)
                    continue
                if not str(getattr(gold, "repo_url", "") or "").strip() or not str(getattr(gold, "commit", "") or "").strip():
                    missing_gold_repo_or_commit.append(instance_id)
            if missing_gold:
                failures.append(
                    {
                        "kind": "missing_gold",
                        "count": len(missing_gold),
                        "instance_ids": missing_gold[:50],
                    }
                )
            if missing_gold_repo_or_commit:
                failures.append(
                    {
                        "kind": "missing_gold_repo_or_commit",
                        "count": len(missing_gold_repo_or_commit),
                        "instance_ids": missing_gold_repo_or_commit[:50],
                    }
                )

        missing_task_repo_metadata: list[str] = []
        for task in tasks:
            instance_id = str(task.get("instance_id") or task.get("original_inst_id") or "").strip()
            if not str(task.get("repo_url") or "").strip() or not str(task.get("commit") or task.get("base_commit") or "").strip():
                missing_task_repo_metadata.append(instance_id)
        if missing_task_repo_metadata:
            failures.append(
                {
                    "kind": "missing_task_repo_metadata",
                    "count": len(missing_task_repo_metadata),
                    "instance_ids": missing_task_repo_metadata[:50],
                    "message": "Selected tasks must include repo_url and commit/base_commit before agent execution starts.",
                }
            )

        if self.config.postprocess.resolve and not self.skip_resolve:
            unsupported_or_unavailable = [
                item
                for item in describe_resolution_backend_support(
                    sorted({str(task.get("bench") or "").strip() for task in tasks if str(task.get("bench") or "").strip()})
                )
                if item["status"] != "available"
            ]
            if unsupported_or_unavailable:
                failures.append(
                    {
                        "kind": "resolution_backend_unavailable",
                        "backends": unsupported_or_unavailable,
                    }
                )

        docker_runtime_needed = any(variant.runtime_backend == "docker" for variant in variants)
        if docker_runtime_needed:
            if not _docker_available():
                failures.append({"kind": "docker_unavailable", "message": "Docker is required for docker runtime variants."})
            else:
                missing_images = [
                    {
                        "variant": variant.name,
                        "image": variant.runtime_image,
                    }
                    for variant in variants
                    if variant.runtime_backend == "docker" and not _docker_image_available(variant.runtime_image)
                ]
                if missing_images:
                    failures.append({"kind": "runtime_image_missing", "images": missing_images})

        claude_docker_variants = [
            variant.name
            for variant in variants
            if variant.agent == "claude" and variant.runtime_backend == "docker"
        ]
        if claude_docker_variants:
            failures.append(
                {
                    "kind": "claude_docker_auth_unsupported",
                    "variants": claude_docker_variants,
                    "message": (
                        "Claude Docker runtime is not enabled because local Claude authentication is not "
                        "copied into the container. Use runtime_backend='host' for Claude until a deterministic "
                        "container auth path is implemented."
                    ),
                }
            )

        missing_prompt = [
            str(task.get("instance_id") or task.get("original_inst_id") or "").strip()
            for task in tasks
            if not str(task.get("prompt") or "").strip()
        ]
        if missing_prompt:
            failures.append({"kind": "missing_prompt", "count": len(missing_prompt), "instance_ids": missing_prompt[:50]})

        if failures:
            proof_path = self._write_failure_proof(
                name="preflight",
                payload={
                    "task_count": len(tasks),
                    "failures": failures,
                },
            )
            raise RuntimeError(f"Run-suite preflight failed; proof written to {proof_path}")

    def _initial_variant_entry(self, variant: EffectiveVariantConfig) -> dict[str, object]:
        variant_dir = self.experiment_dir / "variants" / variant.slug
        return {
            "name": variant.name,
            "slug": variant.slug,
            "description": variant.description,
            "labels": list(variant.labels),
            "notes": variant.notes,
            "status": "pending",
            "config_hash": stable_json_hash(variant.model_dump(mode="json")),
            "effective_config_path": str(variant_dir / "effective-config.json"),
            "output_dir": str(variant_dir),
            "raw_runs_dir": str(variant_dir / "agent_runs"),
            "task_results_path": str(variant_dir / "task-results.jsonl"),
            "pred_path": None,
            "eval_results_path": None,
            "eval_summary_path": None,
            "resolution_summary_path": None,
            "started_at": None,
            "completed_at": None,
            "duration_ms": None,
            "task_counts": {
                "total": 0,
                "completed": 0,
                "failed": 0,
                "timeout": 0,
                "skipped": 0,
            },
            "metrics": {},
            "errors": [],
            "warnings": [],
        }

    def _write_manifest(
        self,
        *,
        started_at: str,
        completed_at: str | None,
        task_set: dict[str, object],
        variant_entries: list[dict[str, object]],
    ) -> None:
        statuses = {entry["status"] for entry in variant_entries}
        if completed_at is None:
            manifest_status = "running"
        elif statuses <= {"completed"}:
            manifest_status = "completed"
        elif "failed" in statuses or "postprocess_failed" in statuses:
            manifest_status = "failed"
        else:
            manifest_status = "completed_with_failures"
        manifest = {
            "experiment_name": self.config.experiment_name,
            "description": self.config.description,
            "agent": self.config.agent,
            "status": manifest_status,
            "started_at": started_at,
            "completed_at": completed_at,
            "max_workers": self.max_workers,
            "resume": self.resume,
            "postprocess_runtime": self.postprocess_runtime_metadata,
            "task_set": task_set,
            "variants": variant_entries,
        }
        write_json(self.manifest_path, manifest)

    def _write_summary(self, variant_entries: list[dict[str, object]]) -> None:
        rows: list[dict[str, object]] = []
        for entry in variant_entries:
            row = {
                "variant": entry["name"],
                "status": entry["status"],
                "total_tasks": entry["task_counts"]["total"],
                "completed_tasks": entry["task_counts"]["completed"],
                "failed_tasks": entry["task_counts"]["failed"],
                "timeout_tasks": entry["task_counts"]["timeout"],
                "skipped_tasks": entry["task_counts"]["skipped"],
                "warning_count": len(entry.get("warnings") or []),
                "warnings": " | ".join(str(item) for item in (entry.get("warnings") or [])),
                "pred_path": entry.get("pred_path") or "",
                "eval_results_path": entry.get("eval_results_path") or "",
                "eval_summary_path": entry.get("eval_summary_path") or "",
            }
            row.update(flatten_metrics(entry.get("metrics") or {}))
            rows.append(row)

        write_json(self.summary_json_path, rows)
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        ensure_dir(self.summary_csv_path.parent)
        with open(self.summary_csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _write_public_artifacts(self) -> None:
        repo_root = Path.cwd().resolve()
        try:
            self.experiment_dir.resolve().relative_to(repo_root)
        except ValueError:
            pass
        else:
            sanitize_artifact_tree_in_place(
                source_dir=self.experiment_dir,
                repo_root=repo_root,
            )
        sanitize_artifact_tree(
            source_dir=self.experiment_dir,
            output_dir=self.public_artifacts_dir,
            repo_root=repo_root,
            overwrite=True,
        )
        assert_paths_have_no_private_artifacts([self.public_artifacts_dir])

    def _sanitize_experiment_artifact(self, value: dict[str, object], *, label: str) -> dict[str, object]:
        context = SanitizationContext(
            repo_root=Path.cwd().resolve(),
            suite_dir=self.experiment_dir.resolve(),
        )
        sanitized = sanitize_json_value(value, context=context)
        if not isinstance(sanitized, dict):
            raise TypeError(f"{label} sanitizer returned non-object payload")
        assert_no_private_paths(sanitized, label=label)
        return sanitized

    def _sanitize_variant_artifact(self, state: _PreparedVariant, value: dict[str, object], *, label: str) -> dict[str, object]:
        context = SanitizationContext(
            repo_root=Path.cwd().resolve(),
            suite_dir=self.experiment_dir.resolve(),
            task_dir=state.pred_path.parent.resolve(),
        )
        sanitized = sanitize_json_value(value, context=context)
        if not isinstance(sanitized, dict):
            raise TypeError(f"{label} sanitizer returned non-object payload")
        assert_no_private_paths(sanitized, label=label)
        return sanitized

    @staticmethod
    def _remove_legacy_runtime_artifacts(raw_root: Path) -> None:
        if not raw_root.exists():
            return
        for legacy_runtime_dir in raw_root.rglob("codex-runtime"):
            if legacy_runtime_dir.is_dir():
                shutil.rmtree(legacy_runtime_dir)

    def _prepare_variant_dir(self, variant: EffectiveVariantConfig) -> tuple[Path, dict[str, object]]:
        variant_dir = self.experiment_dir / "variants" / variant.slug
        effective_config = variant.model_dump(mode="json")
        artifact_config = self._sanitize_experiment_artifact(
            {"effective_config": redact_secrets(effective_config)},
            label=str(variant_dir / "effective-config.json"),
        )["effective_config"]
        if not isinstance(artifact_config, dict):
            raise TypeError("effective config sanitizer returned non-object payload")
        config_payload = {
            "config_hash": stable_json_hash(artifact_config),
            "effective_config": artifact_config,
        }
        if self.config.base_run.rerun and variant_dir.exists():
            shutil.rmtree(variant_dir)
        elif variant_dir.exists():
            effective_config_path = variant_dir / "effective-config.json"
            if effective_config_path.exists():
                previous = read_json(effective_config_path)
                previous_hash = previous.get("config_hash") if isinstance(previous, dict) else None
                previous_effective_config = (
                    previous.get("effective_config")
                    if isinstance(previous, dict)
                    else None
                )
                if previous_hash != config_payload["config_hash"]:
                    if not (
                        self.resume
                        and self._resume_compatible_effective_config(previous_effective_config, artifact_config)
                    ):
                        raise RuntimeError(
                            f"Variant '{variant.name}' already exists with a different effective config. "
                            "Use a new experiment name or enable rerun."
                        )
            if not self.resume:
                raise RuntimeError(
                    f"Variant '{variant.name}' already exists. Re-run with --resume or set base_run.rerun=true."
                )
        ensure_dir(variant_dir)
        write_json(
            variant_dir / "effective-config.json",
            config_payload,
        )
        task_results_path = variant_dir / "task-results.jsonl"
        if task_results_path.exists():
            task_results_path.unlink()
        return variant_dir, config_payload

    def _prepare_variant_state(
        self,
        variant: EffectiveVariantConfig,
        entry: dict[str, object],
        *,
        total_tasks: int,
    ) -> _PreparedVariant:
        variant_dir, config_payload = self._prepare_variant_dir(variant)
        raw_root = variant_dir / "agent_runs"
        self._remove_legacy_runtime_artifacts(raw_root)
        task_results_path = variant_dir / "task-results.jsonl"
        pred_path = variant_dir / "pred.jsonl"
        eval_results_path = variant_dir / "eval.jsonl"
        eval_summary_path = variant_dir / "eval-summary.json"
        resolution_summary_path = variant_dir / "resolution-summary.json"
        for path in (pred_path, eval_results_path, eval_summary_path, resolution_summary_path):
            if self.resume:
                continue
            if path.exists():
                path.unlink()

        entry.update(
            {
                "status": "running",
                "started_at": utc_now(),
                "completed_at": None,
                "duration_ms": None,
                "task_counts": {
                    "total": total_tasks,
                    "completed": 0,
                    "failed": 0,
                    "timeout": 0,
                    "skipped": 0,
                },
                "metrics": {},
                "errors": [],
                "warnings": [],
                "pred_path": None,
                "eval_results_path": None,
                "eval_summary_path": None,
                "resolution_summary_path": None,
                "raw_runs_dir": str(raw_root),
                "task_results_path": str(task_results_path),
                "effective_config_path": str(variant_dir / "effective-config.json"),
                "config_hash": config_payload["config_hash"],
            }
        )
        return _PreparedVariant(
            variant=variant,
            entry=entry,
            raw_root=raw_root,
            task_results_path=task_results_path,
            pred_path=pred_path,
            eval_results_path=eval_results_path,
            eval_summary_path=eval_summary_path,
            resolution_summary_path=resolution_summary_path,
            started_monotonic=time.time(),
        )

    def _task_record_path(self, state: _PreparedVariant, task: dict[str, object]) -> Path:
        return task_record_path(raw_root=state.raw_root, agent=state.variant.agent, task=task)

    def _task_output_dir(self, state: _PreparedVariant, task: dict[str, object]) -> Path:
        task_id = safe_path_component(task_key(task) or "task")
        bench = str(task.get("bench") or "Verified")
        return state.raw_root / state.variant.agent / bench / task_id

    @staticmethod
    def _postprocess_container_path(variant_root: Path, path: Path) -> str:
        relative = path.resolve().relative_to(variant_root.resolve())
        return str((Path("/work") / relative).as_posix())

    @staticmethod
    def _postprocess_repo_path(path: Path) -> str:
        try:
            relative = path.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            raise RuntimeError(f"Postprocess docker backend only supports repo-local gold paths for now: {path}")
        return str((Path("/repo") / relative).as_posix())

    def _run_postprocess_docker_command(
        self,
        *,
        variant_root: Path,
        command: list[str],
        log_path: Path,
        extra_mounts: list[tuple[Path, str]] | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> tuple[int, str]:
        socket_path = _docker_host_socket_path()
        docker_command = [
            "docker",
            "run",
            "--rm",
            "-w",
            "/repo",
            "-v",
            f"{variant_root.resolve()}:/work:rw",
            "-v",
            f"{Path.cwd().resolve()}:/repo:ro",
        ]
        for host_path, container_path in extra_mounts or []:
            docker_command.extend(["-v", f"{host_path.resolve()}:{container_path}:rw"])
        if socket_path is not None:
            docker_command.extend(["-v", f"{socket_path}:/var/run/docker.sock:rw"])
            docker_command.extend(["-e", "DOCKER_HOST=unix:///var/run/docker.sock"])
        env = read_env_file(self.config.postprocess.env_file)
        if extra_env:
            env.update(extra_env)
        for key, value in sorted(env.items()):
            docker_command.extend(["-e", f"{key}={value}"])
        docker_command.append(str(self.config.postprocess.runtime_image))
        docker_command.extend(command)
        return _run_resolution_command(
            command=docker_command,
            cwd=variant_root.resolve(),
            log_path=log_path,
            log_prefix=f"[postprocess:{variant_root.name}]",
            env=None,
        )

    def _convert_records_to_jsonl_docker(
        self,
        *,
        state: _PreparedVariant,
    ) -> dict[str, object]:
        variant_root = state.pred_path.parent.resolve()
        summary_path = variant_root / "conversion-summary.json"
        log_path = variant_root / "conversion-command.log"
        source_dir = state.raw_root / state.variant.agent
        command = [
            "-m",
            "contextbench.run_suites_postprocess",
            "convert",
            "--source-dir",
            self._postprocess_container_path(variant_root, source_dir),
            "--expected-agent",
            state.variant.agent,
            "--out-path",
            self._postprocess_container_path(variant_root, state.pred_path),
            "--summary-path",
            self._postprocess_container_path(variant_root, summary_path),
        ]
        returncode, tail = self._run_postprocess_docker_command(
            variant_root=variant_root,
            command=command,
            log_path=log_path,
        )
        if returncode == 0:
            stale_error = variant_root / "conversion-error.json"
            if stale_error.exists():
                stale_error.unlink()
            return read_json(summary_path)
        write_json(
            variant_root / "conversion-error.json",
            {
                "exit_code": returncode,
                "log_path": str(log_path),
                "tail": tail,
            },
        )
        raise RuntimeError(f"Containerized conversion failed: {tail.strip()}\nFull log: {log_path}")

    def _evaluate_prediction_file_docker(
        self,
        *,
        state: _PreparedVariant,
        selected_task_count: int,
    ) -> dict[str, object]:
        variant_root = state.pred_path.parent.resolve()
        eval_cache = self._prepare_postprocess_evaluation_cache(state)
        log_path = variant_root / "evaluation-command.log"
        command = [
            "-m",
            "contextbench.run_suites_postprocess",
            "evaluate",
            "--gold-path",
            self._postprocess_repo_path(self.config.postprocess.gold_path),
            "--pred-path",
            self._postprocess_container_path(variant_root, state.pred_path),
            "--cache-dir",
            "/cache/eval",
            "--out-path",
            self._postprocess_container_path(variant_root, state.eval_results_path),
            "--summary-path",
            self._postprocess_container_path(variant_root, state.eval_summary_path),
            "--selected-task-count",
            str(selected_task_count),
        ]
        returncode, tail = self._run_postprocess_docker_command(
            variant_root=variant_root,
            command=command,
            log_path=log_path,
            extra_mounts=[(eval_cache, "/cache/eval")],
            extra_env={
                "CONTEXTBENCH_TMP_ROOT": "/cache/eval/worktrees",
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "safe.directory",
                "GIT_CONFIG_VALUE_0": "/cache/eval/*",
            },
        )
        if returncode == 0:
            stale_error = variant_root / "evaluation-error.json"
            if stale_error.exists():
                stale_error.unlink()
            return read_json(state.eval_summary_path)
        write_json(
            variant_root / "evaluation-error.json",
            {
                "exit_code": returncode,
                "log_path": str(log_path),
                "tail": tail,
            },
        )
        raise RuntimeError(f"Containerized evaluation failed: {tail.strip()}\nFull log: {log_path}")

    def _postprocess_evaluation_cache_dir(self, state: _PreparedVariant) -> Path:
        base_cache = self.config.postprocess.cache_dir or state.variant.repo_cache
        return (
            base_cache
            / "postprocess-eval"
            / safe_path_component(self.config.experiment_name)
            / state.variant.slug
        ).resolve()

    def _prepare_postprocess_evaluation_cache(self, state: _PreparedVariant) -> Path:
        eval_cache = self._postprocess_evaluation_cache_dir(state)
        ensure_dir(eval_cache)
        worktree_root = eval_cache / "worktrees"
        if worktree_root.exists():
            if worktree_root.is_symlink() or not worktree_root.is_dir():
                raise RuntimeError(f"Evaluation worktree root is not a directory: {worktree_root}")
            shutil.rmtree(worktree_root)
        ensure_dir(worktree_root)
        return eval_cache

    def _evaluate_prediction_file_host(
        self,
        *,
        state: _PreparedVariant,
        selected_task_count: int,
    ) -> dict[str, object]:
        eval_cache = self._prepare_postprocess_evaluation_cache(state)
        previous_tmp_root = os.environ.get("CONTEXTBENCH_TMP_ROOT")
        os.environ["CONTEXTBENCH_TMP_ROOT"] = str(eval_cache / "worktrees")
        try:
            return evaluate_prediction_file(
                gold_path=self.config.postprocess.gold_path,
                pred_path=state.pred_path,
                cache_dir=eval_cache,
                out_path=state.eval_results_path,
                selected_task_count=selected_task_count,
            )
        finally:
            if previous_tmp_root is None:
                os.environ.pop("CONTEXTBENCH_TMP_ROOT", None)
            else:
                os.environ["CONTEXTBENCH_TMP_ROOT"] = previous_tmp_root

    def _clear_task_outputs(self, state: _PreparedVariant, task: dict[str, object]) -> None:
        task_dir = self._task_output_dir(state, task)
        if task_dir.exists():
            shutil.rmtree(task_dir)

    def _task_is_resume_complete(self, states: list[_PreparedVariant], task: dict[str, object]) -> bool:
        return all(record_is_resume_complete(self._task_record_path(state, task)) for state in states)

    def _workspace_key(self, state: _PreparedVariant, task: dict[str, object]) -> str:
        parts = [
            safe_path_component(self.config.experiment_name),
            self._run_invocation_key,
            safe_path_component(task_key(task) or "task"),
            state.variant.slug,
        ]
        return "__".join(part for part in parts if part)

    def _run_variant_task(self, state: _PreparedVariant, task: dict[str, object]) -> dict[str, object]:
        bench = str(task.get("bench") or "Verified")
        record = run_coding_agent_task(
            task=task,
            agent=state.variant.agent,
            output_dir=state.raw_root / state.variant.agent / bench,
            cache_dir=state.variant.repo_cache,
            schema_path=state.variant.schema_path,
            timeout=state.variant.timeout,
            model=state.variant.model,
            reasoning_effort=state.variant.reasoning_effort,
            agent_args=state.variant.agent_args,
            env_overrides=state.variant.env,
            prompt_preamble=state.variant.setup.prompt_preamble,
            setup=state.variant.setup.model_dump(mode="python"),
            workspace_key=self._workspace_key(state, task),
            runtime_backend=state.variant.runtime_backend,
            runtime_image=state.variant.runtime_image,
            runtime_env=state.variant.runtime_env,
            runtime_setup_commands=state.variant.runtime_setup_commands,
            runtime_keep_failed=state.variant.runtime_keep_failed,
        )
        cleanup_error: str | None = None
        workspace_cleaned = False
        if str(record.get("status") or "") == "completed" and not record.get("timeout"):
            try:
                remove_worktree(
                    str(record.get("repo_url") or task.get("repo_url") or ""),
                    str(state.variant.repo_cache),
                    str(record.get("workspace_path") or ""),
                )
                workspace_cleaned = True
            except Exception as exc:  # pragma: no cover - defensive guard
                cleanup_error = str(exc)
        return {
            "record": record,
            "record_path": self._task_record_path(state, task),
            "workspace_cleaned": workspace_cleaned,
            "cleanup_error": cleanup_error,
        }

    def _record_skipped_task(self, state: _PreparedVariant, task: dict[str, object], task_id: str) -> None:
        counts = state.entry["task_counts"]
        counts["skipped"] += 1
        record_path = self._task_record_path(state, task)
        status = "skipped"
        ok = None
        timeout = False
        try:
            record = read_json(record_path)
        except Exception:
            record = None
        if isinstance(record, dict):
            timeout = bool(record.get("timeout"))
            ok = bool(record.get("ok"))
            if timeout:
                status = "timeout"
            elif "ok" in record and not ok:
                status = "failed"
            else:
                status = str(record.get("status") or "completed")
        append_jsonl(
            state.task_results_path,
            {
                "instance_id": task_id,
                "bench": task.get("bench"),
                "status": status,
                "record_path": str(record_path),
                "resumed": True,
                "timeout": timeout,
                "ok": ok,
            },
        )

    def _record_variant_exception(
        self,
        state: _PreparedVariant,
        task: dict[str, object],
        task_id: str,
        exc: Exception,
    ) -> None:
        counts = state.entry["task_counts"]
        counts["failed"] += 1
        state.entry["errors"].append(f"{task_id}: {exc}")
        append_jsonl(
            state.task_results_path,
            {
                "instance_id": task_id,
                "bench": task.get("bench"),
                "status": "error",
                "error": str(exc),
            },
        )

    def _record_variant_result(
        self,
        state: _PreparedVariant,
        task: dict[str, object],
        task_id: str,
        result: dict[str, object],
    ) -> None:
        record = result["record"]
        record_status = str(record.get("status") or "")
        counts = state.entry["task_counts"]
        if record.get("timeout"):
            counts["timeout"] += 1
        elif "ok" in record and not bool(record.get("ok")):
            counts["failed"] += 1
        elif record_status == "completed":
            counts["completed"] += 1
        else:
            counts["failed"] += 1

        append_jsonl(
            state.task_results_path,
            {
                "instance_id": task_id,
                "bench": task.get("bench"),
                "status": (
                    "timeout"
                    if record.get("timeout")
                    else "failed"
                    if "ok" in record and not bool(record.get("ok"))
                    else record_status or "failed"
                ),
                "record_path": str(result["record_path"]),
                "task_dir": record.get("task_dir"),
                "timeout": bool(record.get("timeout")),
                "ok": bool(record.get("ok")),
                "workspace_cleaned": bool(result["workspace_cleaned"]),
            },
        )
        cleanup_error = result.get("cleanup_error")
        if cleanup_error:
            state.entry["errors"].append(f"{task_id}: workspace cleanup failed: {cleanup_error}")

    def _recalculate_task_counts(self, state: _PreparedVariant, tasks: list[dict[str, object]]) -> dict[str, int]:
        counts = {
            "total": len(tasks),
            "completed": 0,
            "failed": 0,
            "timeout": 0,
            "skipped": 0,
        }
        for task in tasks:
            record_path = self._task_record_path(state, task)
            if not record_path.exists():
                counts["skipped"] += 1
                continue
            try:
                record = read_json(record_path)
            except Exception:
                counts["failed"] += 1
                continue
            if not isinstance(record, dict):
                counts["failed"] += 1
                continue
            if record.get("timeout"):
                counts["timeout"] += 1
                continue
            if "ok" in record and not bool(record.get("ok")):
                counts["failed"] += 1
                continue
            if str(record.get("status") or "") == "completed":
                counts["completed"] += 1
            else:
                counts["failed"] += 1
        return counts

    def _state_metrics(self, state: _PreparedVariant) -> dict[str, object]:
        metrics = state.entry.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
            state.entry["metrics"] = metrics
        return metrics

    def _conversion_summary_path(self, state: _PreparedVariant) -> Path:
        return state.pred_path.parent / "conversion-summary.json"

    @staticmethod
    def _file_sha256(path: Path) -> str | None:
        if not path.exists() or not path.is_file():
            return None
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _task_record_fingerprint(self, state: _PreparedVariant, tasks: list[dict[str, object]]) -> str:
        rows: list[dict[str, object]] = []
        for task in tasks:
            record_path = self._task_record_path(state, task)
            rows.append(
                {
                    "task_id": task_key(task),
                    "bench": task.get("bench"),
                    "record_path": str(record_path.relative_to(state.pred_path.parent)) if record_path.exists() else str(record_path),
                    "record_sha256": self._file_sha256(record_path),
                }
            )
        return stable_json_hash(rows)

    def _conversion_input_fingerprint(self, state: _PreparedVariant, tasks: list[dict[str, object]]) -> str:
        runtime_image_id = (
            self.postprocess_runtime_metadata.get("image_id")
            if self.config.postprocess.runtime_backend == "docker"
            else None
        )
        return stable_json_hash(
            {
                "stage": "conversion",
                "version": _POSTPROCESS_FINGERPRINT_VERSION,
                "agent": state.variant.agent,
                "task_count": len(tasks),
                "runtime_backend": self.config.postprocess.runtime_backend,
                "runtime_image": self.config.postprocess.runtime_image,
                "runtime_image_id": runtime_image_id,
                "task_results_sha256": self._file_sha256(state.task_results_path),
                "records": self._task_record_fingerprint(state, tasks),
            }
        )

    def _evaluation_input_fingerprint(self, state: _PreparedVariant, tasks: list[dict[str, object]]) -> str:
        runtime_image_id = (
            self.postprocess_runtime_metadata.get("image_id")
            if self.config.postprocess.runtime_backend == "docker"
            else None
        )
        return stable_json_hash(
            {
                "stage": "evaluation",
                "version": _POSTPROCESS_FINGERPRINT_VERSION,
                "task_count": len(tasks),
                "runtime_backend": self.config.postprocess.runtime_backend,
                "runtime_image": self.config.postprocess.runtime_image,
                "runtime_image_id": runtime_image_id,
                "gold_path": str(self.config.postprocess.gold_path.resolve()),
                "gold_sha256": self._file_sha256(self.config.postprocess.gold_path),
                "pred_sha256": self._file_sha256(state.pred_path),
            }
        )

    def _resolution_input_fingerprint(self, state: _PreparedVariant, tasks: list[dict[str, object]]) -> str:
        env_file = self.config.postprocess.env_file
        return stable_json_hash(
            {
                "stage": "resolution",
                "version": _POSTPROCESS_FINGERPRINT_VERSION,
                "agent": state.variant.agent,
                "source_dir": str((state.raw_root / state.variant.agent).resolve()),
                "task_results_sha256": self._file_sha256(state.task_results_path),
                "records": self._task_record_fingerprint(state, tasks),
                "harness_args": self.config.postprocess.resolve_harness_args,
                "resolve_workers": self.config.postprocess.resolve_workers,
                "env_file": str(env_file.resolve()) if env_file else None,
                "env_file_sha256": self._file_sha256(env_file) if env_file else None,
            }
        )

    @staticmethod
    def _summary_fingerprint_matches(summary: object, fingerprint: str) -> bool:
        return (
            isinstance(summary, dict)
            and summary.get("fingerprint_version") == _POSTPROCESS_FINGERPRINT_VERSION
            and summary.get("input_fingerprint") == fingerprint
        )

    def _run_conversion_stage(self, state: _PreparedVariant, tasks: list[dict[str, object]]) -> None:
        if not self.config.postprocess.convert or self.skip_convert:
            return
        metrics = self._state_metrics(state)
        summary_path = self._conversion_summary_path(state)
        input_fingerprint = self._conversion_input_fingerprint(state, tasks)
        if self.resume and state.pred_path.exists() and summary_path.exists():
            summary = read_json(summary_path)
            if self._summary_fingerprint_matches(summary, input_fingerprint):
                print(f"[postprocess] variant {state.variant.name}: reusing conversion summary", flush=True)
                metrics["conversion"] = summary
                metrics["prediction_count"] = summary.get("prediction_count")
                return
            print(f"[postprocess] variant {state.variant.name}: conversion summary is stale; regenerating", flush=True)

        print(f"[postprocess] variant {state.variant.name}: starting conversion", flush=True)
        raw_agent_dir = state.raw_root / state.variant.agent
        try:
            if self.config.postprocess.runtime_backend == "docker":
                conversion_summary = self._convert_records_to_jsonl_docker(state=state)
            else:
                conversion_summary = convert_records_to_jsonl(
                    source_dir=raw_agent_dir,
                    expected_agent=state.variant.agent,
                    out_path=state.pred_path,
            )
            conversion_summary["input_fingerprint"] = input_fingerprint
            conversion_summary["fingerprint_version"] = _POSTPROCESS_FINGERPRINT_VERSION
            conversion_summary = self._sanitize_variant_artifact(
                state,
                conversion_summary,
                label=str(summary_path),
            )
            write_json(summary_path, conversion_summary)
            metrics["conversion"] = conversion_summary
            metrics["prediction_count"] = conversion_summary["prediction_count"]
        except Exception as exc:
            state.entry["errors"].append(f"postprocess conversion: {exc}")
            state.postprocess_failed = True

    def _run_evaluation_stage(self, state: _PreparedVariant, tasks: list[dict[str, object]]) -> None:
        if not self.config.postprocess.evaluate or self.skip_evaluate:
            return
        if state.postprocess_failed:
            print(
                f"[postprocess] variant {state.variant.name}: skipping evaluation after earlier postprocess failure",
                flush=True,
            )
            return
        metrics = self._state_metrics(state)
        input_fingerprint = self._evaluation_input_fingerprint(state, tasks)
        if self.resume and state.eval_results_path.exists() and state.eval_summary_path.exists():
            summary = read_json(state.eval_summary_path)
            if self._summary_fingerprint_matches(summary, input_fingerprint):
                print(f"[postprocess] variant {state.variant.name}: reusing evaluation summary", flush=True)
                metrics["evaluation"] = summary
                return
            print(f"[postprocess] variant {state.variant.name}: evaluation summary is stale; regenerating", flush=True)

        print(f"[postprocess] variant {state.variant.name}: starting evaluation", flush=True)
        try:
            if not state.pred_path.exists():
                raise RuntimeError("Prediction file is missing; conversion must succeed before evaluation")
            if self.config.postprocess.runtime_backend == "docker":
                metrics["evaluation"] = self._evaluate_prediction_file_docker(
                    state=state,
                    selected_task_count=len(tasks),
                )
            else:
                metrics["evaluation"] = self._evaluate_prediction_file_host(
                    state=state,
                    selected_task_count=len(tasks),
                )
            metrics["evaluation"]["input_fingerprint"] = input_fingerprint
            metrics["evaluation"]["fingerprint_version"] = _POSTPROCESS_FINGERPRINT_VERSION
            metrics["evaluation"] = self._sanitize_variant_artifact(
                state,
                metrics["evaluation"],
                label=str(state.eval_summary_path),
            )
            write_json(state.eval_summary_path, metrics["evaluation"])
            print(f"[postprocess] variant {state.variant.name}: evaluation summary written", flush=True)
        except Exception as exc:
            state.entry["errors"].append(f"postprocess evaluation: {exc}")
            state.postprocess_failed = True

    def _run_resolution_stage(self, state: _PreparedVariant, tasks: list[dict[str, object]]) -> None:
        if not self.config.postprocess.resolve or self.skip_resolve:
            return
        if state.postprocess_failed:
            print(
                f"[postprocess] variant {state.variant.name}: skipping resolution after earlier postprocess failure",
                flush=True,
            )
            return
        metrics = self._state_metrics(state)
        input_fingerprint = self._resolution_input_fingerprint(state, tasks)
        if self.resume and state.resolution_summary_path.exists():
            summary = read_json(state.resolution_summary_path)
            if isinstance(summary, dict):
                summary_status = str(summary.get("status") or "")
                summary_partial = bool(summary.get("is_partial")) or summary_status == "partial"
                if summary_status == "completed" and not summary_partial and self._summary_fingerprint_matches(summary, input_fingerprint):
                    print(f"[postprocess] variant {state.variant.name}: reusing resolution summary", flush=True)
                    metrics["resolution"] = summary
                    return
                print(f"[postprocess] variant {state.variant.name}: resolution summary is stale or incomplete; regenerating", flush=True)

        print(f"[postprocess] variant {state.variant.name}: starting pass@1 resolution evaluation", flush=True)
        raw_agent_dir = state.raw_root / state.variant.agent
        try:
            metrics["resolution"] = evaluate_resolution_for_suite(
                source_dir=raw_agent_dir,
                expected_agent=state.variant.agent,
                suite_name=self.config.experiment_name,
                variant_name=state.variant.name,
                work_dir=state.pred_path.parent,
                max_workers=self.config.postprocess.resolve_workers,
                harness_args=self.config.postprocess.resolve_harness_args,
                env=read_env_file(self.config.postprocess.env_file),
                run_suffix=self._run_invocation_key,
                resume_existing_resolution=self.resume_resolution,
                clean_resolution_artifacts=not self.resume_resolution,
            )
            metrics["resolution"]["input_fingerprint"] = input_fingerprint
            metrics["resolution"]["fingerprint_version"] = _POSTPROCESS_FINGERPRINT_VERSION
            metrics["resolution"] = self._sanitize_variant_artifact(
                state,
                metrics["resolution"],
                label=str(state.resolution_summary_path),
            )
            write_json(state.resolution_summary_path, metrics["resolution"])
            print(f"[postprocess] variant {state.variant.name}: resolution summary written", flush=True)
        except Exception as exc:
            state.entry["errors"].append(f"postprocess resolution: {exc}")
            state.postprocess_failed = True

    def _integrity_report(
        self,
        *,
        state: _PreparedVariant,
        tasks: list[dict[str, object]],
        counts: dict[str, int],
        conversion_summary: dict[str, object],
        evaluation_summary: dict[str, object],
        resolution_summary: dict[str, object],
    ) -> dict[str, object]:
        checks: list[dict[str, object]] = []

        def add_check(name: str, ok: bool, **details: object) -> None:
            checks.append({"name": name, "ok": bool(ok), **details})

        selected_count = len(tasks)
        add_check("task_count_matches_selection", counts.get("total") == selected_count, total=counts.get("total"), selected=selected_count)
        add_check("no_task_failures", counts.get("failed") == 0 and counts.get("timeout") == 0, failed=counts.get("failed"), timeout=counts.get("timeout"))

        if self.config.postprocess.convert and not self.skip_convert:
            add_check(
                "conversion_complete",
                int(conversion_summary.get("prediction_count") or 0) == selected_count
                and int(conversion_summary.get("missing_record_path_count") or 0) == 0
                and int(conversion_summary.get("nonconvertible_record_count") or 0) == 0
                and not bool(conversion_summary.get("is_partial")),
                prediction_count=conversion_summary.get("prediction_count"),
                selected_task_count=selected_count,
                missing_record_path_count=conversion_summary.get("missing_record_path_count"),
                nonconvertible_record_count=conversion_summary.get("nonconvertible_record_count"),
                is_partial=conversion_summary.get("is_partial"),
            )

        if self.config.postprocess.evaluate and not self.skip_evaluate:
            add_check(
                "evaluation_complete_and_error_free",
                int(evaluation_summary.get("evaluated_prediction_count") or 0) == selected_count
                and int(evaluation_summary.get("num_valid") or 0) == selected_count
                and not bool(evaluation_summary.get("has_errors"))
                and not bool(evaluation_summary.get("error_counts"))
                and not bool(evaluation_summary.get("is_partial")),
                evaluated_prediction_count=evaluation_summary.get("evaluated_prediction_count"),
                num_valid=evaluation_summary.get("num_valid"),
                selected_task_count=selected_count,
                error_counts=evaluation_summary.get("error_counts") or {},
                is_partial=evaluation_summary.get("is_partial"),
            )

        if self.config.postprocess.resolve and not self.skip_resolve:
            add_check(
                "resolution_complete",
                str(resolution_summary.get("status") or "") == "completed"
                and not bool(resolution_summary.get("is_partial"))
                and int(resolution_summary.get("evaluated_task_count") or 0) == int(resolution_summary.get("task_count") or 0)
                and int(resolution_summary.get("task_count") or 0) == selected_count,
                status=resolution_summary.get("status"),
                is_partial=resolution_summary.get("is_partial"),
                task_count=resolution_summary.get("task_count"),
                evaluated_task_count=resolution_summary.get("evaluated_task_count"),
                failed_benches=resolution_summary.get("failed_benches") or [],
                unsupported_benches=resolution_summary.get("unsupported_benches") or [],
            )

        failed_checks = [check for check in checks if not check.get("ok")]
        report = {
            "variant": state.variant.name,
            "created_at": utc_now(),
            "ok": not failed_checks,
            "checks": checks,
            "failed_checks": failed_checks,
        }
        write_json(state.pred_path.parent / "integrity.json", report)
        return report

    def _finalize_variant(self, state: _PreparedVariant, tasks: list[dict[str, object]]) -> None:
        metrics = self._state_metrics(state)
        variant_status = "postprocess_failed" if state.postprocess_failed else "completed"
        warnings: list[str] = []
        counts = self._recalculate_task_counts(state, tasks)
        conversion_summary = metrics.get("conversion") if isinstance(metrics.get("conversion"), dict) else {}
        evaluation_summary = metrics.get("evaluation") if isinstance(metrics.get("evaluation"), dict) else {}
        resolution_summary = metrics.get("resolution") if isinstance(metrics.get("resolution"), dict) else {}

        conversion_partial = bool(conversion_summary.get("is_partial"))
        evaluation_partial = bool(evaluation_summary.get("is_partial"))
        resolution_status = str(resolution_summary.get("status") or "")
        resolution_partial = bool(resolution_summary.get("is_partial")) or resolution_status == "partial"

        if conversion_partial:
            warnings.append(
                "Conversion covered a subset of selected tasks "
                f"({conversion_summary.get('prediction_count')}/{conversion_summary.get('selected_task_count')})."
            )
        if evaluation_partial:
            warnings.append(
                "Evaluation covered a subset of selected tasks "
                f"({evaluation_summary.get('evaluated_prediction_count')}/{evaluation_summary.get('selected_task_count')})."
            )
        if evaluation_summary.get("has_errors") or evaluation_summary.get("error_counts"):
            warnings.append(
                "Evaluation produced row errors: "
                + ", ".join(
                    f"{key}={value}"
                    for key, value in sorted((evaluation_summary.get("error_counts") or {}).items())
                )
                + "."
            )
        if resolution_partial:
            parts: list[str] = []
            if resolution_summary.get("successful_benches"):
                parts.append("successful: " + ", ".join(str(item) for item in resolution_summary.get("successful_benches") or []))
            if resolution_summary.get("failed_benches"):
                parts.append("failed: " + ", ".join(str(item) for item in resolution_summary.get("failed_benches") or []))
            if resolution_summary.get("unsupported_benches"):
                parts.append("unsupported: " + ", ".join(str(item) for item in resolution_summary.get("unsupported_benches") or []))
            if resolution_summary.get("partial_benches"):
                parts.append("partial coverage: " + ", ".join(str(item) for item in resolution_summary.get("partial_benches") or []))
            detail = f" ({'; '.join(parts)})" if parts else ""
            warnings.append("Resolution evaluation covered only a subset of benches" + detail + ".")

        metrics["postprocess_partial"] = bool(warnings)
        metrics["conversion_partial"] = conversion_partial
        metrics["evaluation_partial"] = evaluation_partial
        metrics["resolution_partial"] = resolution_partial
        integrity = self._integrity_report(
            state=state,
            tasks=tasks,
            counts=counts,
            conversion_summary=conversion_summary,
            evaluation_summary=evaluation_summary,
            resolution_summary=resolution_summary,
        )
        metrics["integrity"] = integrity

        if variant_status == "completed" and (counts["failed"] > 0 or counts["timeout"] > 0):
            variant_status = "completed_with_failures"
        elif variant_status == "completed" and resolution_status == "failed":
            variant_status = "postprocess_failed"
        elif variant_status == "completed" and not integrity["ok"]:
            variant_status = "postprocess_failed"
            failed_names = ", ".join(str(check.get("name")) for check in integrity["failed_checks"])
            state.entry["errors"].append(f"integrity checks failed: {failed_names}")

        state.entry.update(
            {
                "status": variant_status,
                "completed_at": utc_now(),
                "duration_ms": int((time.time() - state.started_monotonic) * 1000),
                "task_counts": counts,
                "metrics": metrics,
                "warnings": warnings,
                "pred_path": str(state.pred_path) if state.pred_path.exists() else None,
                "eval_results_path": str(state.eval_results_path) if state.eval_results_path.exists() else None,
                "eval_summary_path": str(state.eval_summary_path) if state.eval_summary_path.exists() else None,
                "resolution_summary_path": str(state.resolution_summary_path) if state.resolution_summary_path.exists() else None,
            }
        )

    def run(self) -> int:
        tasks, task_set = self._load_tasks()
        effective_variants = [
            build_run_suite_variant(self.config, variant)
            for variant in self.config.variants
            if variant.enabled
        ]
        if not effective_variants:
            raise RuntimeError("No enabled variants remain after config filtering")

        ensure_dir(self.experiment_dir)
        write_json(
            self.experiment_config_path,
            self._sanitize_experiment_artifact(
                redact_secrets(self.config.model_dump(mode="json")),
                label=str(self.experiment_config_path),
            ),
        )
        self._validate_preflight(tasks, effective_variants)
        print(
            "[preflight] selected "
            f"{task_set['count']} tasks by bench: {task_set['bench_counts']}",
            flush=True,
        )

        started_at = utc_now()
        variant_entries = [self._initial_variant_entry(variant) for variant in effective_variants]
        states = [
            self._prepare_variant_state(variant, entry, total_tasks=len(tasks))
            for variant, entry in zip(effective_variants, variant_entries, strict=False)
        ]
        self._write_manifest(
            started_at=started_at,
            completed_at=None,
            task_set=task_set,
            variant_entries=variant_entries,
        )

        workers = min(self.max_workers, len(states))
        for index, task in enumerate(tasks, start=1):
            task_id = task_key(task) or f"task-{index}"
            bench = str(task.get("bench") or "Verified")
            if self.resume and self._task_is_resume_complete(states, task):
                print(f"[task {index}/{len(tasks)}] skip {bench} | {task_id}", flush=True)
                for state in states:
                    self._record_skipped_task(state, task, task_id)
                self._write_manifest(
                    started_at=started_at,
                    completed_at=None,
                    task_set=task_set,
                    variant_entries=variant_entries,
                )
                continue

            print(f"[task {index}/{len(tasks)}] run {bench} | {task_id}", flush=True)
            for state in states:
                self._clear_task_outputs(state, task)

            future_map = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for state in states:
                    future = executor.submit(self._run_variant_task, state, task)
                    future_map[future] = state

                for future in as_completed(future_map):
                    state = future_map[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        self._record_variant_exception(state, task, task_id, exc)
                    else:
                        self._record_variant_result(state, task, task_id, result)

            self._write_manifest(
                started_at=started_at,
                completed_at=None,
                task_set=task_set,
                variant_entries=variant_entries,
            )

        for state in states:
            self._run_conversion_stage(state, tasks)
        for state in states:
            self._run_evaluation_stage(state, tasks)
        for state in states:
            self._run_resolution_stage(state, tasks)
        for state in states:
            self._finalize_variant(state, tasks)

        completed_at = utc_now()
        self._write_manifest(
            started_at=started_at,
            completed_at=completed_at,
            task_set=task_set,
            variant_entries=variant_entries,
        )
        self._write_summary(variant_entries)
        self._write_public_artifacts()

        bad_statuses = {"failed", "completed_with_failures", "postprocess_failed"}
        return 0 if all(entry["status"] not in bad_statuses for entry in variant_entries) else 1
