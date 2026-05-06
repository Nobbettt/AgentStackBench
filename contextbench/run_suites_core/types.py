
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
    "codex_home",
    "xdg_config_home",
    "xdg_data_home",
    "xdg_cache_home",
]

RuntimeBackend = Literal["host", "docker"]

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


class BaseRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_data: Path = DEFAULT_GOLD_PATH
    task_csv: Path | None = DEFAULT_SUBSET_CSV
    subset_csv: Path | None = None
    bench: list[str] | None = None
    instances: list[str] | None = None
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
    runtime_image: str | None = None
    runtime_env_file: Path | None = None
    runtime_env: dict[str, str] = Field(default_factory=dict)
    runtime_setup_commands: list[str] = Field(default_factory=list)
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


class VariantConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    enabled: bool = True
    labels: list[str] = Field(default_factory=list)
    notes: str | None = None
    model: str | None = None
    reasoning_effort: ReasoningLevel | None = None
    timeout: int | None = None
    agent_args_add: list[str] = Field(default_factory=list)
    agent_args_replace: list[str] | None = None
    env_add: dict[str, str] = Field(default_factory=dict)
    env_replace: dict[str, str] | None = None
    setup: VariantSetupConfig = Field(default_factory=VariantSetupConfig)
    runtime_backend: RuntimeBackend | None = None
    runtime_image: str | None = None
    runtime_env_file: Path | None = None
    runtime_env_add: dict[str, str] = Field(default_factory=dict)
    runtime_env_replace: dict[str, str] | None = None
    runtime_setup_commands_add: list[str] = Field(default_factory=list)
    runtime_setup_commands_replace: list[str] | None = None
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


class ParallelismConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_workers: int = Field(default=1, gt=0)


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
        if self.base_run.runtime_backend == "docker" and self.base_run.runtime_image is None:
            self.base_run.runtime_image = DEFAULT_AGENT_RUNTIME_IMAGES.get(self.agent)
        names = [variant.name for variant in self.variants]
        if len(names) != len(set(names)):
            raise ValueError("Variant names must be unique")
        slugs = [safe_path_component(name) for name in names]
        if len(slugs) != len(set(slugs)):
            raise ValueError("Variant names must remain unique after path normalization")
        self._validate_setup_target_roots(self.base_run.setup, location="base_run.setup")
        self._validate_reasoning_effort(self.base_run.reasoning_effort, location="base_run.reasoning_effort")
        self._validate_runtime_config(
            self.base_run.runtime_backend,
            self.base_run.runtime_image,
            location="base_run",
        )
        for index, variant in enumerate(self.variants):
            self._validate_setup_target_roots(variant.setup, location=f"variants[{index}].setup")
            self._validate_reasoning_effort(variant.reasoning_effort, location=f"variants[{index}].reasoning_effort")
            effective_backend = variant.runtime_backend or self.base_run.runtime_backend
            effective_image = variant.runtime_image if variant.runtime_image is not None else self.base_run.runtime_image
            if effective_backend == "host" and variant.runtime_backend == "host" and variant.runtime_image is None:
                effective_image = None
            self._validate_runtime_config(
                effective_backend,
                effective_image,
                location=f"variants[{index}]",
            )
        return self

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

    def _validate_runtime_config(self, runtime_backend: RuntimeBackend, runtime_image: str | None, *, location: str) -> None:
        if self.agent == "claude" and runtime_backend == "docker":
            raise ValueError(f"{location}.runtime_backend='docker' is not supported for Claude; use runtime_backend='host'")
        if runtime_backend == "docker" and not str(runtime_image or "").strip():
            raise ValueError(f"{location}.runtime_image is required when runtime_backend='docker'")
        if runtime_backend == "host" and runtime_image is not None:
            raise ValueError(f"{location}.runtime_image can only be set when runtime_backend='docker'")


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
    runtime_image: str | None = None
    runtime_env: dict[str, str] = Field(default_factory=dict)
    runtime_setup_commands: list[str] = Field(default_factory=list)
    runtime_keep_failed: bool = False

    @field_validator("agent", mode="before")
    @classmethod
    def normalize_agent(cls, value: object) -> str:
        normalized = normalize_coding_agent_name(value)
        if normalized is None:
            available = ", ".join(sorted(SUPPORTED_CODING_AGENTS))
            raise ValueError(f"Unsupported coding agent adapter: {value!r}. Available: {available}")
        return normalized
