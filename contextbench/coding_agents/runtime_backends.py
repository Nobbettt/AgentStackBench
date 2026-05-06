
"""Execution backends for coding-agent task commands."""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .files import ensure_dir, usage_error
from .runtime_common import coerce_output_text, run_command as host_run_command
from .types import CommandResult

SUPPORTED_RUNTIME_BACKENDS = frozenset({"host", "docker"})


@dataclass(frozen=True)
class RuntimeBackendConfig:
    """Normalized runtime-backend configuration for one coding-agent task."""

    backend: str
    image: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    setup_commands: tuple[str, ...] = ()
    keep_failed: bool = False


@dataclass(frozen=True)
class RuntimeSetupResult:
    """Result for a runtime-level shell setup command."""

    command_result: CommandResult
    stdout_path: Path
    stderr_path: Path
    command: str


def normalize_runtime_backend_config(
    *,
    runtime_backend: str,
    runtime_image: str | None = None,
    runtime_env: Mapping[str, object] | None = None,
    runtime_setup_commands: Sequence[object] | None = None,
    runtime_keep_failed: bool = False,
) -> RuntimeBackendConfig:
    """Validate and normalize user-facing runtime backend options."""

    backend = str(runtime_backend or "").strip().lower()
    if not backend:
        raise usage_error("runtime_backend is required")
    if backend not in SUPPORTED_RUNTIME_BACKENDS:
        allowed = ", ".join(sorted(SUPPORTED_RUNTIME_BACKENDS))
        raise usage_error(f"Unsupported runtime backend: {runtime_backend!r}. Available: {allowed}")

    image = str(runtime_image or "").strip() or None
    if backend == "docker" and image is None:
        raise usage_error("runtime_image is required when runtime_backend='docker'")
    if backend == "host" and image is not None:
        raise usage_error("runtime_image can only be used with runtime_backend='docker'")

    env = {str(key): str(value) for key, value in dict(runtime_env or {}).items()}
    setup_commands = tuple(
        text
        for text in (str(item).strip() for item in (runtime_setup_commands or ()))
        if text
    )
    return RuntimeBackendConfig(
        backend=backend,
        image=image,
        env=env,
        setup_commands=setup_commands,
        keep_failed=bool(runtime_keep_failed),
    )


def create_task_runtime(
    config: RuntimeBackendConfig,
    *,
    workspace_path: Path,
    task_dir: Path,
    schema_path: Path | None,
    extra_writable_dirs: Sequence[Path] = (),
) -> "BaseTaskRuntime":
    if config.backend == "docker":
        return DockerTaskRuntime(
            config=config,
            workspace_path=workspace_path,
            task_dir=task_dir,
            schema_path=schema_path,
            extra_writable_dirs=tuple(extra_writable_dirs),
        )
    return HostTaskRuntime(config=config)


def docker_checkout_tmp_root(cache_dir: Path) -> Path:
    """Return a Docker-friendly default tmp root for benchmark worktrees."""

    return (cache_dir.parent / "worktrees").resolve()


def merge_command_env(config_env: Mapping[str, str], env: Mapping[str, str] | None) -> dict[str, str] | None:
    merged: dict[str, str] = {}
    merged.update(config_env)
    if env:
        merged.update({str(key): str(value) for key, value in env.items()})
    return merged or None


class BaseTaskRuntime:
    """Runtime lifecycle for commands that belong to one benchmark task."""

    config: RuntimeBackendConfig

    def start(self) -> None:
        return None

    def run_command(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        stdin_text: str | None,
        stdout_path: Path,
        stderr_path: Path,
        timeout: int,
        env: dict[str, str] | None = None,
        host_runner: Callable[..., CommandResult] = host_run_command,
    ) -> CommandResult:
        raise NotImplementedError

    def close(self, *, success: bool) -> None:
        return None

    def metadata(self) -> dict[str, object]:
        return {"backend": self.config.backend}


@dataclass
class HostTaskRuntime(BaseTaskRuntime):
    """Host execution backend matching the historical runtime behavior."""

    config: RuntimeBackendConfig

    def run_command(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        stdin_text: str | None,
        stdout_path: Path,
        stderr_path: Path,
        timeout: int,
        env: dict[str, str] | None = None,
        host_runner: Callable[..., CommandResult] = host_run_command,
    ) -> CommandResult:
        return host_runner(
            command,
            cwd=cwd,
            stdin_text=stdin_text,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout=timeout,
            env=merge_command_env(self.config.env, env),
        )


@dataclass
class DockerTaskRuntime(BaseTaskRuntime):
    """Persistent Docker container used for all commands in one task."""

    config: RuntimeBackendConfig
    workspace_path: Path
    task_dir: Path
    schema_path: Path | None
    extra_writable_dirs: tuple[Path, ...] = ()
    container_name: str | None = None
    _started: bool = False
    _timed_out: bool = False
    _image_id: str | None = None
    _exec_user: str | None = None

    def start(self) -> None:
        if self._started:
            return
        image = self.config.image
        if not image:
            raise usage_error("Docker runtime requires runtime_image")
        self._image_id = _docker_image_id(image)
        self._exec_user = _host_user_spec()

        self.container_name = self._container_name()
        command = [
            "docker",
            "run",
            "--detach",
            "--name",
            self.container_name,
            "--workdir",
            str(self.workspace_path),
        ]
        for key, value in sorted(self.config.env.items()):
            command.extend(["--env", f"{key}={value}"])
        for source, target, readonly in self._mounts():
            mount = f"type=bind,source={source},target={target}"
            if readonly:
                mount += ",readonly"
            command.extend(["--mount", mount])
        command.extend([image, "sh", "-lc", _container_bootstrap_command(self._exec_user)])

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise usage_error(
                "Docker runtime failed to start: "
                + (result.stderr or result.stdout or f"exit {result.returncode}").strip()
            )
        self._started = True

    def run_command(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        stdin_text: str | None,
        stdout_path: Path,
        stderr_path: Path,
        timeout: int,
        env: dict[str, str] | None = None,
        host_runner: Callable[..., CommandResult] = host_run_command,
    ) -> CommandResult:
        del host_runner
        self.start()
        ensure_dir(stdout_path.parent)
        ensure_dir(stderr_path.parent)
        if not self.container_name:
            raise usage_error("Docker runtime was not started")

        exec_command = [
            "docker",
            "exec",
            "-i",
            "--workdir",
            str(cwd),
        ]
        if self._exec_user:
            exec_command.extend(["--user", self._exec_user])
        merged_env = merge_command_env(self.config.env, env) or {}
        for key, value in sorted(merged_env.items()):
            exec_command.extend(["--env", f"{key}={value}"])
        exec_command.append(self.container_name)
        exec_command.extend(["timeout", "--foreground", "--kill-after", "10s", f"{int(timeout)}s"])
        exec_command.extend(str(part) for part in command)

        try:
            result = subprocess.run(
                exec_command,
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=timeout + 15,
                check=False,
            )
            stdout_path.write_text(coerce_output_text(result.stdout), encoding="utf-8")
            stderr_path.write_text(coerce_output_text(result.stderr), encoding="utf-8")
            timed_out = result.returncode in {124, 137}
            if timed_out:
                self._timed_out = True
            return {
                "ok": result.returncode == 0 and not timed_out,
                "exit_code": None if timed_out else result.returncode,
                "signal": "SIGTERM" if timed_out else None,
                "timeout": timed_out,
            }
        except subprocess.TimeoutExpired as exc:
            self._timed_out = True
            stdout_path.write_text(coerce_output_text(exc.stdout), encoding="utf-8")
            stderr_path.write_text(coerce_output_text(exc.stderr), encoding="utf-8")
            return {
                "ok": False,
                "exit_code": None,
                "signal": "SIGTERM",
                "timeout": True,
            }

    def close(self, *, success: bool) -> None:
        if not self.container_name or not self._started:
            return
        if self.config.keep_failed and not success:
            if self._timed_out:
                subprocess.run(
                    ["docker", "stop", "--time", "1", self.container_name],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self._started = False
            (self.task_dir / "docker-container.txt").write_text(self.container_name + "\n", encoding="utf-8")
            return
        subprocess.run(
            ["docker", "rm", "--force", self.container_name],
            capture_output=True,
            text=True,
            check=False,
        )
        self._started = False

    def metadata(self) -> dict[str, object]:
        return {
            "backend": self.config.backend,
            "image": self.config.image,
            "image_id": self._image_id,
            "container_name": self.container_name,
            "user": self._exec_user,
        }

    def _container_name(self) -> str:
        seed = f"{self.task_dir.resolve()}:{time.time_ns()}"
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
        return f"contextbench-{digest}"

    def _mounts(self) -> list[tuple[str, str, bool]]:
        mounts: list[tuple[Path, Path, bool]] = [
            (self.workspace_path.resolve(), self.workspace_path.resolve(), False),
            (self.task_dir.resolve(), self.task_dir.resolve(), False),
        ]
        mounts.extend((path.resolve(), path.resolve(), False) for path in self.extra_writable_dirs)
        mounts.extend((path, path, readonly) for path, readonly in _git_metadata_mounts(self.workspace_path))
        if self.schema_path is not None:
            schema_parent = self.schema_path.resolve().parent
            if not (
                _path_within(schema_parent, self.workspace_path.resolve())
                or _path_within(schema_parent, self.task_dir.resolve())
            ):
                mounts.append((schema_parent, schema_parent, True))

        deduped: list[tuple[str, str, bool]] = []
        seen: set[tuple[str, str]] = set()
        writable_sources = {
            str(self.workspace_path.resolve()),
            str(self.task_dir.resolve()),
            *(str(path.resolve()) for path in self.extra_writable_dirs),
        }
        for source, target, readonly in mounts:
            key = (str(source), str(target))
            if key in seen:
                continue
            seen.add(key)
            if not source.exists():
                raise usage_error(f"Docker runtime mount source does not exist: {source}")
            deduped.append((str(source), str(target), readonly and str(source) not in writable_sources))
        return deduped


def _path_within(path: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath([str(path), str(parent)]) == str(parent)
    except ValueError:
        return False


def _resolve_git_path(workspace_path: Path, value: str) -> Path:
    path = Path(value.strip())
    if not path.is_absolute():
        path = workspace_path / path
    return path.resolve()


def _host_user_spec() -> str | None:
    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    if getuid is None or getgid is None:
        return None
    try:
        return f"{getuid()}:{getgid()}"
    except OSError:
        return None


def _container_bootstrap_command(user_spec: str | None) -> str:
    if not user_spec:
        return "while true; do sleep 3600; done"
    uid, gid = user_spec.split(":", 1)
    return f"""
set -eu
if ! getent group {gid} >/dev/null 2>&1; then
  groupadd -g {gid} contextbench
fi
user_name="$(getent passwd {uid} | cut -d: -f1 || true)"
if [ -z "$user_name" ]; then
  group_name="$(getent group {gid} | cut -d: -f1)"
  useradd -m -u {uid} -g "$group_name" -s /bin/bash contextbench
  user_name=contextbench
fi
printf '%s ALL=(ALL) NOPASSWD:ALL\\n' "$user_name" > /etc/sudoers.d/contextbench
chmod 0440 /etc/sudoers.d/contextbench
while true; do sleep 3600; done
"""


def _git_metadata_mounts(workspace_path: Path) -> list[tuple[Path, bool]]:
    """Return external Git metadata paths needed by detached worktrees."""

    mounts: list[tuple[Path, bool]] = []
    for arg in ("--git-common-dir", "--git-dir"):
        try:
            result = subprocess.run(
                ["git", "-C", str(workspace_path), "rev-parse", arg],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            continue
        if result.returncode != 0:
            continue
        path = _resolve_git_path(workspace_path, result.stdout)
        if not path.exists() or _path_within(path, workspace_path.resolve()):
            continue
        readonly = arg == "--git-common-dir"
        existing_paths = [existing for existing, _ in mounts]
        if not any(path == existing for existing in existing_paths):
            mounts.append((path, readonly))
    return mounts


def _docker_image_id(image: str) -> str | None:
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def run_runtime_setup_commands(
    runtime: BaseTaskRuntime,
    *,
    commands: Sequence[str],
    workspace_path: Path,
    task_dir: Path,
    timeout: int,
    env: dict[str, str] | None,
) -> RuntimeSetupResult | None:
    """Run configured unscored shell setup commands before agent prompts."""

    for index, command in enumerate(commands, start=1):
        stdout_path = task_dir / f"runtime-setup-{index}.stdout.log"
        stderr_path = task_dir / f"runtime-setup-{index}.stderr.log"
        result = runtime.run_command(
            ["/bin/sh", "-lc", command],
            cwd=workspace_path,
            stdin_text=None,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout=timeout,
            env=env,
        )
        if not result["ok"]:
            return RuntimeSetupResult(
                command_result=result,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                command=command,
            )
    return None
