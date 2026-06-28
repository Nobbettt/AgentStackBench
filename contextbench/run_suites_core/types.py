# SPDX-License-Identifier: Apache-2.0
# Fork note: Modified by Norbert Laszlo on 2026-05-21 from upstream ContextBench.
# Summary of changes: add fork run-suite resolution, scheduling, and runtime prebuild controls.

"""Pydantic models for run suite configuration and state."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..agents.registry import get_coding_agent_adapter, iter_coding_agent_adapters, normalize_coding_agent_name
from ..coding_agents.constants import (
    DEFAULT_AGENT_RUNTIME_IMAGES,
    DEFAULT_CACHE_DIR,
    DEFAULT_GOLD_PATH,
    DEFAULT_POSTPROCESS_RUNTIME_IMAGE,
    DEFAULT_SUBSET_CSV,
    REPO_ROOT,
)
from ..coding_agents.files import safe_path_component
from .helpers import normalize_str_list

RuntimeTargetRoot = Literal[
    "task_dir",
    "runtime_root",
    "home_dir",
    "claude_home",
    "codex_home",
    "xdg_config_home",
    "xdg_data_home",
    "xdg_cache_home",
]

RuntimeBackend = Literal["host", "docker"]
RuntimeImageSource = Literal["configured", "resolution"]
SelectionAssertion = Literal["full_dataset", "configured_selection"]
AgentScheduler = Literal["global", "per_task"]

ReasoningLevel = Literal[
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
]

SUPPORTED_RUNTIME_TARGET_ROOTS: frozenset[str] = frozenset(get_args(RuntimeTargetRoot))
SUPPORTED_REASONING_LEVELS: frozenset[str] = frozenset(get_args(ReasoningLevel))
SUPPORTED_CODING_AGENTS: frozenset[str] = frozenset(adapter.name for adapter in iter_coding_agent_adapters())


def _normalize_runtime_image_map(value: object, *, field_name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object mapping benchmark names to Docker images")
    normalized: dict[str, str] = {}
    for raw_bench, raw_image in value.items():
        bench = str(raw_bench).strip()
        image = str(raw_image).strip()
        if not bench:
            raise ValueError(f"{field_name} contains an empty benchmark name")
        if not image:
            raise ValueError(f"{field_name}.{bench} must be a non-empty Docker image")
        normalized[bench] = image
    return normalized


def _normalize_runtime_python_image_map(value: object, *, field_name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object mapping Python versions to Docker images")
    normalized: dict[str, str] = {}
    for raw_version, raw_image in value.items():
        version = str(raw_version).strip()
        image = str(raw_image).strip()
        if not version:
            raise ValueError(f"{field_name} contains an empty Python version")
        if not image:
            raise ValueError(f"{field_name}.{version} must be a non-empty Docker image")
        normalized[version] = image
    return normalized


class MaterializedFileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    content: Any
    format: Literal["text", "json"] = "text"
    target_root: RuntimeTargetRoot = "task_dir"


class CopyPathConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Path
    destination: str = "."
    target_root: RuntimeTargetRoot = "task_dir"


class VariantSetupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_preamble: str | None = None
    setup_prompt: str | None = None
    setup_prompt_timeout: int | None = Field(default=None, gt=0)
    copy_paths: list[CopyPathConfig] = Field(default_factory=list)
    files_to_materialize: list[MaterializedFileConfig] = Field(default_factory=list)
    claude_settings_overrides: dict[str, Any] = Field(default_factory=dict)
    claude_mcp_config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt_preamble", "setup_prompt", mode="before")
    @classmethod
    def normalize_optional_prompt_text(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class RuntimePrebuildConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    commands: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    image_tag: str | None = None
    pull_base_image: bool = False
    force_rebuild: bool = False

    @field_validator("commands", mode="before")
    @classmethod
    def normalize_commands(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("runtime_prebuild.commands must be a list")
        return [text for text in (str(item).strip() for item in value) if text]

    @field_validator("image_tag", mode="before")
    @classmethod
    def normalize_optional_image_tag(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @model_validator(mode="after")
    def validate_enabled_commands(self) -> "RuntimePrebuildConfig":
        if self.enabled and not self.commands:
            raise ValueError("runtime_prebuild.commands must be non-empty when runtime_prebuild.enabled=true")
        return self


class BaseRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_data: Path = DEFAULT_GOLD_PATH
    task_csv: Path | None = DEFAULT_SUBSET_CSV
    subset_csv: Path | None = None
    bench: list[str] | None = None
    instances: list[str] | None = None
    selection_assertion: SelectionAssertion | None = None
    limit: int = Field(default=0, ge=0)
    timeout: int = Field(default=1800, gt=0)
    repo_cache: Path = DEFAULT_CACHE_DIR
    output_root: Path = REPO_ROOT / "results" / "run_suites"
    schema_path: Path | None = None
    model: str | None = None
    reasoning_effort: ReasoningLevel | None = None
    rerun: bool = False
    agent_args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    setup: VariantSetupConfig = Field(default_factory=VariantSetupConfig)
    runtime_backend: RuntimeBackend = "docker"
    runtime_image_source: RuntimeImageSource = "configured"
    runtime_image: str | None = None
    runtime_images_by_bench: dict[str, str] = Field(default_factory=dict)
    runtime_images_by_python: dict[str, str] = Field(default_factory=dict)
    runtime_platform: str | None = None
    runtime_env_file: Path | None = None
    runtime_env: dict[str, str] = Field(default_factory=dict)
    runtime_setup_timeout: int | None = Field(default=None, gt=0)
    runtime_validation_timeout: int | None = Field(default=None, gt=0)
    runtime_setup_cache: bool = False
    runtime_setup_cache_dir: Path | None = None
    runtime_prebuild: RuntimePrebuildConfig = Field(default_factory=RuntimePrebuildConfig)
    runtime_setup_commands: list[str] = Field(default_factory=list)
    runtime_validation_commands: list[str] = Field(default_factory=list)
    diff_exclude_paths: list[str] = Field(default_factory=list)
    required_tool_call_patterns: list[str] = Field(default_factory=list)
    required_command_patterns: list[str] = Field(default_factory=list)
    required_available_tool_patterns: list[str] = Field(default_factory=list)
    runtime_keep_failed: bool = False

    @field_validator("bench", "instances", mode="before")
    @classmethod
    def normalize_optional_lists(cls, value: object) -> list[str] | None:
        return normalize_str_list(value)

    @field_validator("reasoning_effort", mode="before")
    @classmethod
    def normalize_reasoning_effort(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        return text or None

    @field_validator("runtime_images_by_bench", mode="before")
    @classmethod
    def normalize_runtime_images_by_bench(cls, value: object) -> dict[str, str]:
        return _normalize_runtime_image_map(value, field_name="base_run.runtime_images_by_bench")

    @field_validator("runtime_images_by_python", mode="before")
    @classmethod
    def normalize_runtime_images_by_python(cls, value: object) -> dict[str, str]:
        return _normalize_runtime_python_image_map(value, field_name="base_run.runtime_images_by_python")


class VariantConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    enabled: bool = True
    labels: list[str] = Field(default_factory=list)
    notes: str | None = None
    model: str | None = None
    reasoning_effort: ReasoningLevel | None = None
    timeout: int | None = Field(default=None, gt=0)
    agent_args_add: list[str] = Field(default_factory=list)
    agent_args_replace: list[str] | None = None
    env_add: dict[str, str] = Field(default_factory=dict)
    env_replace: dict[str, str] | None = None
    setup: VariantSetupConfig = Field(default_factory=VariantSetupConfig)
    runtime_backend: RuntimeBackend | None = None
    runtime_image_source: RuntimeImageSource | None = None
    runtime_image: str | None = None
    runtime_images_by_bench_add: dict[str, str] = Field(default_factory=dict)
    runtime_images_by_bench_replace: dict[str, str] | None = None
    runtime_images_by_python_add: dict[str, str] = Field(default_factory=dict)
    runtime_images_by_python_replace: dict[str, str] | None = None
    runtime_platform: str | None = None
    runtime_env_file: Path | None = None
    runtime_env_add: dict[str, str] = Field(default_factory=dict)
    runtime_env_replace: dict[str, str] | None = None
    runtime_setup_timeout: int | None = Field(default=None, gt=0)
    runtime_validation_timeout: int | None = Field(default=None, gt=0)
    runtime_setup_cache: bool | None = None
    runtime_setup_cache_dir: Path | None = None
    runtime_prebuild: RuntimePrebuildConfig | None = None
    runtime_setup_commands_add: list[str] = Field(default_factory=list)
    runtime_setup_commands_replace: list[str] | None = None
    runtime_validation_commands_add: list[str] = Field(default_factory=list)
    runtime_validation_commands_replace: list[str] | None = None
    diff_exclude_paths_add: list[str] = Field(default_factory=list)
    diff_exclude_paths_replace: list[str] | None = None
    required_tool_call_patterns_add: list[str] = Field(default_factory=list)
    required_tool_call_patterns_replace: list[str] | None = None
    required_command_patterns_add: list[str] = Field(default_factory=list)
    required_command_patterns_replace: list[str] | None = None
    required_available_tool_patterns_add: list[str] = Field(default_factory=list)
    required_available_tool_patterns_replace: list[str] | None = None
    runtime_keep_failed: bool | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("Variant names must be non-empty")
        return name

    @field_validator("reasoning_effort", mode="before")
    @classmethod
    def normalize_reasoning_effort(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        return text or None

    @field_validator("runtime_images_by_bench_add", mode="before")
    @classmethod
    def normalize_runtime_images_by_bench_add(cls, value: object) -> dict[str, str]:
        return _normalize_runtime_image_map(value, field_name="variants[].runtime_images_by_bench_add")

    @field_validator("runtime_images_by_bench_replace", mode="before")
    @classmethod
    def normalize_runtime_images_by_bench_replace(cls, value: object) -> dict[str, str] | None:
        if value is None:
            return None
        return _normalize_runtime_image_map(value, field_name="variants[].runtime_images_by_bench_replace")

    @field_validator("runtime_images_by_python_add", mode="before")
    @classmethod
    def normalize_runtime_images_by_python_add(cls, value: object) -> dict[str, str]:
        return _normalize_runtime_python_image_map(value, field_name="variants[].runtime_images_by_python_add")

    @field_validator("runtime_images_by_python_replace", mode="before")
    @classmethod
    def normalize_runtime_images_by_python_replace(cls, value: object) -> dict[str, str] | None:
        if value is None:
            return None
        return _normalize_runtime_python_image_map(value, field_name="variants[].runtime_images_by_python_replace")


class ParallelismConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_workers: int = Field(default=1, gt=0)
    agent_workers: int | None = Field(default=None, gt=0)
    scheduler: AgentScheduler = "global"


class PostprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    convert: bool = True
    evaluate: bool = True
    resolve: bool = False
    runtime_backend: RuntimeBackend = "docker"
    runtime_image: str | None = DEFAULT_POSTPROCESS_RUNTIME_IMAGE
    gold_path: Path = DEFAULT_GOLD_PATH
    cache_dir: Path | None = None
    env_file: Path | None = None
    resolve_workers: int = Field(default=1, gt=0)
    prebuild_resolution_images: bool = False
    prebuild_resolution_workers: int | None = Field(default=None, gt=0)
    swebench_timeout: int = Field(default=1800, gt=0)
    self_clean_resolution_artifacts: bool = True
    self_clean_resolution_docker_images: bool = True
    rerun_empty_patch_records_on_resume: bool = False
    resolve_harness_args: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_convert_evaluate(self) -> "PostprocessConfig":
        if self.evaluate and not self.convert:
            raise ValueError("postprocess.evaluate requires postprocess.convert=true")
        if self.runtime_backend == "docker" and not str(self.runtime_image or "").strip():
            raise ValueError("postprocess.runtime_image is required when postprocess.runtime_backend='docker'")
        return self


class RunSuiteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    description: str | None = None
    agent: str
    base_run: BaseRunConfig = Field(default_factory=BaseRunConfig)
    variants: list[VariantConfig]
    parallelism: ParallelismConfig = Field(default_factory=ParallelismConfig)
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)

    @field_validator("experiment_name")
    @classmethod
    def validate_experiment_name(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("Experiment name must be non-empty")
        return name

    @field_validator("agent", mode="before")
    @classmethod
    def normalize_agent(cls, value: object) -> str:
        normalized = normalize_coding_agent_name(value)
        if normalized is None:
            available = ", ".join(sorted(SUPPORTED_CODING_AGENTS))
            raise ValueError(f"Unsupported coding agent adapter: {value!r}. Available: {available}")
        return normalized

    @model_validator(mode="after")
    def validate_variants(self) -> "RunSuiteConfig":
        if not self.variants:
            raise ValueError("At least one variant is required")
        if self.base_run.schema_path is None:
            self.base_run.schema_path = get_coding_agent_adapter(self.agent).output_schema_path
        if (
            self.base_run.runtime_backend == "docker"
            and self.base_run.runtime_image_source == "configured"
            and self.base_run.runtime_image is None
        ):
            self.base_run.runtime_image = DEFAULT_AGENT_RUNTIME_IMAGES.get(self.agent)
        names = [variant.name for variant in self.variants]
        if len(names) != len(set(names)):
            raise ValueError("Variant names must be unique")
        slugs = [safe_path_component(name) for name in names]
        if len(slugs) != len(set(slugs)):
            raise ValueError("Variant names must remain unique after path normalization")
        self._validate_setup_target_roots(self.base_run.setup, location="base_run.setup")
        self._validate_agent_specific_setup(self.base_run.setup, location="base_run.setup")
        self._validate_reasoning_effort(self.base_run.reasoning_effort, location="base_run.reasoning_effort")
        self._validate_runtime_config(
            self.base_run.runtime_backend,
            self.base_run.runtime_image_source,
            self.base_run.runtime_image,
            self.base_run.runtime_images_by_bench,
            self.base_run.runtime_images_by_python,
            self.base_run.runtime_platform,
            location="base_run",
        )
        for index, variant in enumerate(self.variants):
            self._validate_setup_target_roots(variant.setup, location=f"variants[{index}].setup")
            self._validate_agent_specific_setup(variant.setup, location=f"variants[{index}].setup")
            self._validate_reasoning_effort(variant.reasoning_effort, location=f"variants[{index}].reasoning_effort")
            effective_backend = variant.runtime_backend or self.base_run.runtime_backend
            effective_image_source = variant.runtime_image_source or self.base_run.runtime_image_source
            effective_image = variant.runtime_image if variant.runtime_image is not None else self.base_run.runtime_image
            effective_platform = (
                variant.runtime_platform
                if variant.runtime_platform is not None
                else self.base_run.runtime_platform
            )
            if (
                effective_backend == "docker"
                and effective_image_source == "configured"
                and effective_image is None
            ):
                effective_image = DEFAULT_AGENT_RUNTIME_IMAGES.get(self.agent)
            if effective_backend == "host" and variant.runtime_backend == "host" and variant.runtime_image is None:
                effective_image_source = "configured"
                effective_image = None
                effective_platform = None
            effective_images_by_bench = (
                dict(variant.runtime_images_by_bench_replace)
                if variant.runtime_images_by_bench_replace is not None
                else {
                    **self.base_run.runtime_images_by_bench,
                    **variant.runtime_images_by_bench_add,
                }
            )
            effective_images_by_python = (
                dict(variant.runtime_images_by_python_replace)
                if variant.runtime_images_by_python_replace is not None
                else {
                    **self.base_run.runtime_images_by_python,
                    **variant.runtime_images_by_python_add,
                }
            )
            if effective_backend == "host" and variant.runtime_backend == "host":
                effective_images_by_bench = {}
                effective_images_by_python = {}
            self._validate_runtime_config(
                effective_backend,
                effective_image_source,
                effective_image,
                effective_images_by_bench,
                effective_images_by_python,
                effective_platform,
                location=f"variants[{index}]",
            )
            effective_prebuild = variant.runtime_prebuild or self.base_run.runtime_prebuild
            if effective_prebuild.enabled and effective_backend != "docker":
                raise ValueError(
                    f"variants[{index}].runtime_prebuild requires runtime_backend='docker'"
                )
            if effective_prebuild.enabled and effective_image_source != "configured":
                raise ValueError(
                    f"variants[{index}].runtime_prebuild requires runtime_image_source='configured'"
                )
        return self

    def _validate_agent_specific_setup(self, setup: VariantSetupConfig, *, location: str) -> None:
        if self.agent == "claude":
            return
        invalid_entries = []
        if setup.claude_settings_overrides:
            invalid_entries.append(f"{location}.claude_settings_overrides")
        if setup.claude_mcp_config:
            invalid_entries.append(f"{location}.claude_mcp_config")
        if not invalid_entries:
            return
        details = ", ".join(invalid_entries)
        raise ValueError(
            f"Agent '{self.agent}' does not support Claude-specific setup fields; invalid entries: {details}"
        )

    def _validate_setup_target_roots(self, setup: VariantSetupConfig, *, location: str) -> None:
        allowed_roots = get_coding_agent_adapter(self.agent).supported_runtime_target_roots
        invalid_entries: list[str] = []

        for index, spec in enumerate(setup.copy_paths):
            if spec.target_root not in allowed_roots:
                invalid_entries.append(f"{location}.copy_paths[{index}].target_root={spec.target_root!r}")
        for index, spec in enumerate(setup.files_to_materialize):
            if spec.target_root not in allowed_roots:
                invalid_entries.append(f"{location}.files_to_materialize[{index}].target_root={spec.target_root!r}")

        if invalid_entries:
            allowed_display = ", ".join(sorted(allowed_roots))
            details = ", ".join(invalid_entries)
            raise ValueError(
                f"Agent '{self.agent}' only supports setup target_root values [{allowed_display}]; invalid entries: {details}"
            )

    def _validate_reasoning_effort(self, reasoning_effort: ReasoningLevel | None, *, location: str) -> None:
        if reasoning_effort is None:
            return
        allowed_levels = get_coding_agent_adapter(self.agent).supported_reasoning_efforts
        if reasoning_effort not in allowed_levels:
            allowed_display = ", ".join(sorted(allowed_levels))
            raise ValueError(
                f"Agent '{self.agent}' only supports reasoning_effort values [{allowed_display}]; "
                f"invalid entry: {location}={reasoning_effort!r}"
            )

    def _validate_runtime_config(
        self,
        runtime_backend: RuntimeBackend,
        runtime_image_source: RuntimeImageSource,
        runtime_image: str | None,
        runtime_images_by_bench: dict[str, str],
        runtime_images_by_python: dict[str, str],
        runtime_platform: str | None,
        *,
        location: str,
    ) -> None:
        if runtime_backend == "docker" and runtime_image_source == "configured" and not str(runtime_image or "").strip():
            raise ValueError(f"{location}.runtime_image is required when runtime_backend='docker'")
        if runtime_backend == "docker" and runtime_image_source == "resolution":
            if runtime_image is not None:
                raise ValueError(f"{location}.runtime_image cannot be set when runtime_image_source='resolution'")
            if runtime_images_by_bench:
                raise ValueError(f"{location}.runtime_images_by_bench cannot be set when runtime_image_source='resolution'")
            if runtime_images_by_python:
                raise ValueError(f"{location}.runtime_images_by_python cannot be set when runtime_image_source='resolution'")
            return
        if runtime_backend == "host" and runtime_image_source != "configured":
            raise ValueError(f"{location}.runtime_image_source='resolution' requires runtime_backend='docker'")
        if runtime_backend == "host" and runtime_image is not None:
            raise ValueError(f"{location}.runtime_image can only be set when runtime_backend='docker'")
        if runtime_backend == "host" and runtime_images_by_bench:
            raise ValueError(f"{location}.runtime_images_by_bench can only be set when runtime_backend='docker'")
        if runtime_backend == "host" and runtime_images_by_python:
            raise ValueError(f"{location}.runtime_images_by_python can only be set when runtime_backend='docker'")
        if runtime_backend == "host" and runtime_platform is not None:
            raise ValueError(f"{location}.runtime_platform can only be set when runtime_backend='docker'")


class EffectiveVariantConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    slug: str
    description: str | None = None
    labels: list[str] = Field(default_factory=list)
    notes: str | None = None
    agent: str
    task_data: Path
    task_csv: Path | None = None
    subset_csv: Path | None = None
    bench: list[str] | None = None
    instances: list[str] | None = None
    limit: int = 0
    timeout: int = 1800
    repo_cache: Path = DEFAULT_CACHE_DIR
    schema_path: Path
    model: str | None = None
    reasoning_effort: ReasoningLevel | None = None
    env: dict[str, str] = Field(default_factory=dict)
    agent_args: list[str] = Field(default_factory=list)
    setup: VariantSetupConfig = Field(default_factory=VariantSetupConfig)
    runtime_backend: RuntimeBackend = "docker"
    runtime_image_source: RuntimeImageSource = "configured"
    runtime_image: str | None = None
    runtime_images_by_bench: dict[str, str] = Field(default_factory=dict)
    runtime_images_by_python: dict[str, str] = Field(default_factory=dict)
    runtime_platform: str | None = None
    runtime_env: dict[str, str] = Field(default_factory=dict)
    runtime_setup_timeout: int | None = None
    runtime_validation_timeout: int | None = None
    runtime_setup_cache: bool = False
    runtime_setup_cache_dir: Path | None = None
    runtime_prebuild: RuntimePrebuildConfig = Field(default_factory=RuntimePrebuildConfig)
    runtime_setup_commands: list[str] = Field(default_factory=list)
    runtime_validation_commands: list[str] = Field(default_factory=list)
    diff_exclude_paths: list[str] = Field(default_factory=list)
    required_tool_call_patterns: list[str] = Field(default_factory=list)
    required_command_patterns: list[str] = Field(default_factory=list)
    required_available_tool_patterns: list[str] = Field(default_factory=list)
    runtime_keep_failed: bool = False

    @field_validator("agent", mode="before")
    @classmethod
    def normalize_agent(cls, value: object) -> str:
        normalized = normalize_coding_agent_name(value)
        if normalized is None:
            available = ", ".join(sorted(SUPPORTED_CODING_AGENTS))
            raise ValueError(f"Unsupported coding agent adapter: {value!r}. Available: {available}")
        return normalized
