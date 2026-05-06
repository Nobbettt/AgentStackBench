import json
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from swerex.deployment.config import (
    DeploymentConfig,
    DockerDeploymentConfig,
    DummyDeploymentConfig,
    LocalDeploymentConfig,
)
from typing_extensions import Self

from sweagent.agent.problem_statement import (
    ProblemStatementConfig,
    SWEBenchMultimodalProblemStatement,
    TextProblemStatement,
)
from sweagent.environment.repo import GithubRepoConfig, LocalRepoConfig, PreExistingRepoConfig
from sweagent.environment.swe_env import EnvironmentConfig
from sweagent.utils.files import load_file
from sweagent.utils.log import get_logger

logger = get_logger("swea-config", emoji="🔧")

# Global flag to ensure we only patch DockerDeployment once
_swerex_patched = False


class AbstractInstanceSource(ABC):
    """Anything that adheres to this standard can be used to load instances."""

    @abstractmethod
    def get_instance_configs(self) -> list[EnvironmentConfig]: ...


class BatchInstance(BaseModel):
    """A single instance in a batch of instances.
    This specifies both the environment configuration and the problem statement.
    """

    env: EnvironmentConfig
    problem_statement: ProblemStatementConfig


def _slice_spec_to_slice(slice_spec: str) -> slice:
    if slice_spec == "":
        return slice(None)
    parts = slice_spec.split(":")
    values = [None if p == "" else int(p) for p in parts]
    if len(parts) == 1:
        return slice(values[0])
    if len(parts) == 2:
        return slice(values[0], values[1])
    if len(parts) == 3:
        return slice(values[0], values[1], values[2])
    msg = (
        f"Invalid slice specification: {slice_spec!r}. "
        "Here's the expected format: stop or start:stop or start:stop:step "
        "(i.e., it behaves exactly like python's list slicing `list[slice]`)."
    )
    raise ValueError(msg)


def _filter_batch_items(
    instances: list[BatchInstance], *, filter_: str, slice_: str = "", shuffle: bool = False
) -> list[BatchInstance]:
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x.problem_statement.id)
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_, instance.problem_statement.id)]
    after_filter = len(instances)
    if before_filter != after_filter:
        logger.info("Instance filter: %d -> %d instances", before_filter, after_filter)
    if slice_:
        instances = instances[_slice_spec_to_slice(slice_)]
        after_slice = len(instances)
        if before_filter != after_slice:
            logger.info("Instance slice: %d -> %d instances", before_filter, after_slice)
    return instances


class SimpleBatchInstance(BaseModel):
    """A simple way to configure a single instance in a batch of instances that all
    use similar deployment configurations.

    Predominantly used for benchmarking purposes. Assumes that the repository is already
    present in the docker container.
    """

    image_name: str
    problem_statement: str
    instance_id: str
    repo_name: str = ""
    """Specifies the repository to use. If empty, no repository is used.
    If the string does not contain a slash, it is interpreted as an already existing repository at the root
    of the docker container. If it contains the word "github", it is interpreted as a github repository.
    Else, it is interpreted as a local repository.
    """
    base_commit: str = "HEAD"
    """Used to reset repo."""
    extra_fields: dict[str, Any] = Field(default_factory=dict)
    """Any additional data to be added to the instance.
    This data will be available when formatting prompt templates.
    """

    # Ignore instead of allow because they should be added as `extra_fields`
    model_config = ConfigDict(extra="ignore")

    def to_full_batch_instance(self, deployment: DeploymentConfig) -> BatchInstance:
        """Merge the deployment options into the `SimpleBatchInstance` object to get a full `BatchInstance`."""
        # Very important: Make a copy of the deployment config because it will be shared among instances!!!
        deployment = deployment.model_copy(deep=True)

        if "issue_images" in self.extra_fields:
            problem_statement = SWEBenchMultimodalProblemStatement(
                text=self.problem_statement,
                issue_images=self.extra_fields.pop("issue_images"),
                id=self.instance_id,
                extra_fields=self.extra_fields,
            )
        else:
            problem_statement = TextProblemStatement(
                text=self.problem_statement, id=self.instance_id, extra_fields=self.extra_fields
            )

        if not self.repo_name:
            repo = None
        elif "github" in self.repo_name:
            repo = GithubRepoConfig(github_url=self.repo_name, base_commit=self.base_commit)
        elif "/" not in self.repo_name:
            repo = PreExistingRepoConfig(repo_name=self.repo_name, base_commit=self.base_commit)
        else:
            repo = LocalRepoConfig(path=Path(self.repo_name), base_commit=self.base_commit)
        if isinstance(deployment, LocalDeploymentConfig):
            if self.image_name:
                msg = "Local deployment does not support image_name"
                raise ValueError(msg)
            return BatchInstance(
                env=EnvironmentConfig(deployment=deployment, repo=repo), problem_statement=problem_statement
            )
        if isinstance(deployment, DummyDeploymentConfig):
            return BatchInstance(
                env=EnvironmentConfig(deployment=deployment, repo=repo), problem_statement=problem_statement
            )

        # 检查是否有 patched 镜像（优先使用）
        image_name = self.image_name
        if image_name and not image_name.endswith("-patched") and image_name.startswith("mswebench/"):
            patched_image_name = f"{image_name}-patched"
            import subprocess
            try:
                result = subprocess.run(
                    ["docker", "image", "inspect", patched_image_name],
                    capture_output=True,
                    timeout=2,
                    check=False
                )
                if result.returncode == 0:
                    image_name = patched_image_name
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                pass  # 如果检查失败，使用原始镜像名
        
        deployment.image = image_name  # type: ignore

        if isinstance(deployment, DockerDeploymentConfig):
            # Normalize empty string to None so standalone Python build is disabled.
            if deployment.python_standalone_dir == "":
                deployment.python_standalone_dir = None  # type: ignore
            if deployment.python_standalone_dir is None:
                # For Multi-SWE-bench images (mswebench/*), disable standalone Python
                # as these images may not support building standalone Python
                if image_name and image_name.startswith("mswebench/"):
                    deployment.python_standalone_dir = None  # type: ignore
                # For Multi-SWE-bench images, we need to ensure Python 3.10+ is available
                # for swe-rex installation. The swerex library will try to install swe-rex
                # using python3 (which is 3.8), but swe-rex requires Python >= 3.10.
                # 
                # Solution: We'll monkey-patch swerex's Docker deployment to use a custom
                # startup command that installs Python 3.10+ and swe-rex before starting.
                # This is done by patching the _get_swerex_start_cmd method in swerex.deployment.docker
                global _swerex_patched
                if not _swerex_patched:
                    try:
                        from swerex.deployment.docker import DockerDeployment
                        from swerex import REMOTE_EXECUTABLE_NAME, PACKAGE_NAME
                        original_get_swerex_start_cmd = DockerDeployment._get_swerex_start_cmd
                        
                        def patched_get_swerex_start_cmd(self, token: str) -> list[str]:
                            """Patched version that installs Python 3.10+ and swe-rex for Multi-SWE-bench images."""
                            if self._config.image and self._config.image.startswith("mswebench/"):
                                # Custom startup command for Multi-SWE-bench images
                                rex_args = f"--auth-token {token}"
                                install_cmd = """
                                if ! command -v swerex-remote >/dev/null 2>&1; then
                                    # Try to find Python 3.10+
                                    PYTHON_CMD=""
                                    for py in python3.11 python3.10 python3.12 python3; do
                                        if command -v $py >/dev/null 2>&1; then
                                            PYTHON_CMD=$py
                                            break
                                        fi
                                    done
                                    
                                    # If not found, try to install Python 3.10
                                    if [ -z "$PYTHON_CMD" ] && [ -f /etc/debian_version ]; then
                                        apt-get update -qq >/dev/null 2>&1 && \
                                        apt-get install -y -qq software-properties-common >/dev/null 2>&1 && \
                                        add-apt-repository -y ppa:deadsnakes/ppa >/dev/null 2>&1 && \
                                        apt-get update -qq >/dev/null 2>&1 && \
                                        apt-get install -y -qq python3.10 python3.10-venv python3.10-dev >/dev/null 2>&1 && \
                                        PYTHON_CMD=python3.10 || true
                                    fi
                                    
                                    # Install swe-rex using the found Python
                                    if [ -n "$PYTHON_CMD" ]; then
                                        $PYTHON_CMD -m venv /root/venv_swe >/dev/null 2>&1 || true
                                        /root/venv_swe/bin/pip install --quiet --upgrade pip >/dev/null 2>&1 || true
                                        # Try to install swe-rex and verify installation
                                        INSTALLED=0
                                        if /root/venv_swe/bin/pip install --quiet swe-rex >/dev/null 2>&1 && \
                                           /root/venv_swe/bin/python -c "import swerex" >/dev/null 2>&1; then
                                            INSTALLED=1
                                        elif /root/venv_swe/bin/pip install --quiet git+https://github.com/SWE-agent/SWE-ReX.git >/dev/null 2>&1 && \
                                             /root/venv_swe/bin/python -c "import swerex" >/dev/null 2>&1; then
                                            INSTALLED=1
                                        fi
                                        
                                        if [ $INSTALLED -eq 1 ]; then
                                            # Verify and create swerex-remote if it exists in venv
                                            if [ -f /root/venv_swe/bin/swerex-remote ]; then
                                                ln -sf /root/venv_swe/bin/swerex-remote /usr/local/bin/swerex-remote >/dev/null 2>&1 || true
                                            else
                                                # Create wrapper script using venv python
                                                cat > /usr/local/bin/swerex-remote << 'EOFWRAP'
#!/bin/sh
exec /root/venv_swe/bin/python -m swerex.runtime.remote "$@"
EOFWRAP
                                                chmod +x /usr/local/bin/swerex-remote >/dev/null 2>&1 || true
                                            fi
                                        fi
                                    fi
                                fi
                                """
                                # Combine install command with swerex-remote startup
                                # Try swerex-remote command first, fallback to venv python, then system python
                                venv_python_fallback = f"/root/venv_swe/bin/python -m swerex.runtime.remote {rex_args}"
                                system_python_fallback = f"python3 -m pip install --break-system-packages --quiet swe-rex >/dev/null 2>&1 && python3 -m swerex.runtime.remote {rex_args}"
                                cmd = f"{install_cmd.strip()} && (command -v swerex-remote >/dev/null 2>&1 && swerex-remote {rex_args} || ([ -f /root/venv_swe/bin/python ] && /root/venv_swe/bin/python -c 'import swerex' >/dev/null 2>&1 && {venv_python_fallback} || {system_python_fallback}))"
                                return ["/bin/sh", "-c", cmd]
                            else:
                                # Use original method for other images
                                return original_get_swerex_start_cmd(self, token)
                        
                        # Apply the patch
                        DockerDeployment._get_swerex_start_cmd = patched_get_swerex_start_cmd
                        _swerex_patched = True
                        logger.debug("Patched DockerDeployment._get_swerex_start_cmd for Multi-SWE-bench images")
                    except Exception as e:
                        logger.warning(f"Failed to patch DockerDeployment for Multi-SWE-bench: {e}")
                        # Continue without patch - will likely fail but at least we tried
            else:
                # Note: you can disable this by setting python_standalone_dir to null or ""
                deployment.python_standalone_dir = "/root"  # type: ignore

        return BatchInstance(
            env=EnvironmentConfig(deployment=deployment, repo=repo), problem_statement=problem_statement
        )

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_id(cls, data):
        # Handling compatibility with swe-agent <= 1.0.1
        if isinstance(data, dict):
            if "id" in data and "instance_id" not in data:
                data["instance_id"] = data["id"]
                data.pop("id")
        return data

    # todo: Maybe populate extra fields?
    @classmethod
    def from_swe_bench(cls, instance: dict[str, Any]) -> Self:
        """Convert instances from the classical SWE-bench dataset or Multi-SWE-bench to the `SimpleBatchInstance` format."""
        # Handle Multi-SWE-bench format (has org, repo, number fields)
        if "org" in instance and "repo" in instance and "number" in instance:
            # Multi-SWE-bench format: org, repo, number
            org = instance["org"]
            repo = instance["repo"]
            number = instance["number"]
            iid = f"{org}__{repo}-{number}"

            # Multi-SWE-bench image format: mswebench/{org}_m_{repo}:pr-{number}
            image_name = instance.get("image_name", None)
            if image_name is None:
                org_safe = org.replace("_", "-")
                repo_safe = repo.replace("_", "-")
                image_name = f"mswebench/{org_safe}_m_{repo_safe}:pr-{number}"
            
            # 无论 image_name 是否已存在，都检查是否有 patched 版本（优先使用）
            if image_name and not image_name.endswith("-patched"):
                patched_image_name = f"{image_name}-patched"
                # 使用 subprocess 检查镜像是否存在（更可靠，不依赖docker库）
                import subprocess
                try:
                    result = subprocess.run(
                        ["docker", "image", "inspect", patched_image_name],
                        capture_output=True,
                        timeout=2,
                        check=False
                    )
                    if result.returncode == 0:
                        image_name = patched_image_name
                except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                    pass  # 如果检查失败，使用原始镜像名

            # Extract problem statement from body or title
            problem_statement = instance.get("body") or instance.get("title") or instance.get("problem_statement", "")

            # Extract base commit
            if "base" in instance and isinstance(instance["base"], dict):
                base_commit = instance["base"].get("sha") or instance.get("base_commit", "HEAD")
            else:
                base_commit = instance.get("base_commit", "HEAD")
        elif instance.get("instance_id", "").startswith("instance_") and "repo" in instance:
            # Handle SWE-bench Pro format
            iid = instance["instance_id"]
            repo_full = instance["repo"]

            image_name = instance.get("image_name", None)
            if image_name is None and repo_full and "/" in repo_full:
                repo_base, repo_name_only = repo_full.lower().split("/", 1)
                hsh = iid.replace("instance_", "")

                # Handle special cases from mini-swe-agent-extension
                if iid == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
                    repo_name_only = 'element-web'
                elif 'element-hq' in repo_full.lower() and 'element-web' in repo_full.lower():
                    repo_name_only = 'element'
                    if hsh.endswith('-vnan'):
                        hsh = hsh[:-5]
                elif hsh.endswith('-vnan'):
                    hsh = hsh[:-5]

                tag = f"{repo_base}.{repo_name_only}-{hsh}"
                if len(tag) > 128:
                    tag = tag[:128]
                image_name = f"jefzda/sweap-images:{tag}"

            if image_name is None:
                # Docker doesn't allow double underscore, so we replace them with a magic token
                id_docker_compatible = iid.replace("__", "_1776_")
                image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
            problem_statement = instance.get("problem_statement", "")
            base_commit = instance.get("base_commit", "HEAD")
        else:
            # Standard SWE-bench format
            iid = instance["instance_id"]
            image_name = instance.get("image_name", None)
            if image_name is None:
                # Docker doesn't allow double underscore, so we replace them with a magic token
                id_docker_compatible = iid.replace("__", "_1776_")
                image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
            problem_statement = instance["problem_statement"]
            base_commit = instance["base_commit"]
        
        extra_fields = {}
        if "image_assets" in instance:
            issue_images = json.loads(instance["image_assets"])["problem_statement"]
            extra_fields["issue_images"] = issue_images
        
        # For Multi-SWE-bench, use the actual repo name (located at /home/{repo_name})
        # For other formats, use "testbed" (located at /testbed)
        if "org" in instance and "repo" in instance and "number" in instance:
            # Multi-SWE-bench format: repository is at /home/{repo_name}
            repo_name = instance["repo"]
        else:
            # Standard SWE-bench format: repository is at /testbed
            repo_name = "testbed"
        
        return cls(
            image_name=image_name,
            problem_statement=problem_statement,
            instance_id=iid,
            repo_name=repo_name,
            base_commit=base_commit,
            extra_fields=extra_fields,
        )


class InstancesFromFile(BaseModel, AbstractInstanceSource):
    """Load instances from a file."""

    path: Path
    filter: str = ".*"
    """Regular expression to filter the instances by instance id."""
    slice: str = ""
    """Select only a slice of the instances (after filtering by `filter`).
    Possible values are stop or start:stop or start:stop:step
    (i.e., it behaves exactly like python's list slicing `list[slice]`).
    """
    shuffle: bool = False
    """Shuffle the instances (before filtering and slicing)."""

    deployment: DeploymentConfig = Field(
        default_factory=lambda: DockerDeploymentConfig(image="python:3.11"),
        description="Deployment options.",
    )
    """Note that the image_name option is overwritten by the images specified in the task instances."""

    simple: Literal[True] = True
    """Convenience discriminator for (de)serialization/CLI. Do not change."""

    type: Literal["file"] = "file"
    """Discriminator for (de)serialization/CLI. Do not change."""

    def get_instance_configs(self) -> list[BatchInstance]:
        instance_dicts = load_file(self.path)
        simple_instances = [SimpleBatchInstance.model_validate(instance_dict) for instance_dict in instance_dicts]
        instances = [instance.to_full_batch_instance(self.deployment) for instance in simple_instances]
        return _filter_batch_items(instances, filter_=self.filter, slice_=self.slice, shuffle=self.shuffle)

    @property
    def id(self) -> str:
        return self.path.stem


class InstancesFromHuggingFace(BaseModel, AbstractInstanceSource):
    """Load instances from HuggingFace."""

    dataset_name: str
    """Name of the HuggingFace dataset. Same as when using `datasets.load_dataset`."""
    split: str = "dev"
    filter: str = ".*"
    """Regular expression to filter the instances by instance id."""
    slice: str = ""
    """Select only a slice of the instances (after filtering by `filter`).
    Possible values are stop or start:stop or start:stop:step.
    (i.e., it behaves exactly like python's list slicing `list[slice]`).
    """
    shuffle: bool = False
    """Shuffle the instances (before filtering and slicing)."""

    deployment: DeploymentConfig = Field(
        default_factory=lambda: DockerDeploymentConfig(image="python:3.11"),
    )
    """Deployment configuration. Note that the `image_name` option is overwritten by the images specified in the task instances.
    """
    type: Literal["huggingface"] = "huggingface"
    """Discriminator for (de)serialization/CLI. Do not change."""

    def get_instance_configs(self) -> list[BatchInstance]:
        from datasets import load_dataset

        ds: list[dict[str, Any]] = load_dataset(self.dataset_name, split=self.split)  # type: ignore
        simple_instances: list[SimpleBatchInstance] = [SimpleBatchInstance.model_validate(instance) for instance in ds]
        instances = [instance.to_full_batch_instance(self.deployment) for instance in simple_instances]
        return _filter_batch_items(instances, filter_=self.filter, slice_=self.slice, shuffle=self.shuffle)

    @property
    def id(self) -> str:
        ds_name = "".join(l for l in self.dataset_name if l.isalnum() or l in ["-", "_"])
        return f"{ds_name}_{self.split}"


class SWEBenchInstances(BaseModel, AbstractInstanceSource):
    """Load instances from SWE-bench."""

    subset: Literal["lite", "verified", "full", "multimodal", "multilingual", "pro"] = "lite"
    """Subset of swe-bench to use"""

    # IMPORTANT: Do not call this `path`, because then if people do not specify instance.type,
    # it might be resolved to ExpertInstancesFromFile or something like that.
    path_override: str | Path | None = None
    """Allow to specify a different huggingface dataset name or path to a huggingface
    dataset. This will override the automatic path set by `subset`.
    """

    split: Literal["dev", "test", "train"] = "dev"

    deployment: DeploymentConfig = Field(
        default_factory=lambda: DockerDeploymentConfig(image="python:3.11"),
    )
    """Deployment configuration. Note that the image_name option is overwritten by the images specified in the task instances.
    """

    type: Literal["swe_bench"] = "swe_bench"
    """Discriminator for (de)serialization/CLI. Do not change."""

    filter: str = ".*"
    """Regular expression to filter the instances by instance id."""
    slice: str = ""
    """Select only a slice of the instances (after filtering by `filter`).
    Possible values are stop or start:stop or start:stop:step.
    (i.e., it behaves exactly like python's list slicing `list[slice]`).
    """
    shuffle: bool = False
    """Shuffle the instances (before filtering and slicing)."""

    evaluate: bool = False
    """Run sb-cli to evaluate"""

    def _get_dataset_path(self) -> str:
        if self.path_override is not None:
            return str(self.path_override)
        dataset_mapping = {
            "full": "princeton-nlp/SWE-Bench",
            "verified": "princeton-nlp/SWE-Bench_Verified",
            "lite": "princeton-nlp/SWE-Bench_Lite",
            "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
            "multilingual": "swe-bench/SWE-Bench_Multilingual",
            "pro": "ScaleAI/SWE-bench_Pro",
        }

        if self.subset not in dataset_mapping:
            msg = f"Unsupported subset: {self.subset}"
            raise ValueError(msg)

        return dataset_mapping[self.subset]

    def get_instance_configs(self) -> list[BatchInstance]:
        from datasets import load_dataset
        import re

        dataset_path = self._get_dataset_path()
        
        # For Multi-SWE-bench, use streaming mode to avoid type casting issues
        # Multi-SWE-bench has complex nested structures that cause PyArrow issues
        is_multiswe = "Multi-SWE-bench" in dataset_path or "multi-swe-bench" in dataset_path.lower()
        
        if isinstance(self.deployment, DockerDeploymentConfig):
            self.deployment.platform = "linux/amd64"

        if is_multiswe:
            # Load image overrides if available
            image_overrides = {}
            # Try multiple possible paths for image_overrides.json
            possible_paths = [
                Path(__file__).parent.parent.parent.parent / "image_overrides.json",  # From sweagent-eval root
                Path(__file__).parent.parent.parent / "image_overrides.json",  # From SWE-agent root
                Path.cwd() / "image_overrides.json",  # Current working directory
            ]
            for override_file in possible_paths:
                if override_file.exists():
                    try:
                        with open(override_file, "r") as f:
                            image_overrides = json.load(f)
                        logger.debug(f"Loaded {len(image_overrides)} image overrides from {override_file}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load image overrides from {override_file}: {e}")
            
            # Use streaming mode for Multi-SWE-bench and process on-the-fly
            dataset = load_dataset(dataset_path, split=self.split, streaming=True)  # type: ignore
            instances = []
            filter_pattern = re.compile(self.filter) if self.filter != ".*" else None
            
            for instance_dict in dataset:
                # For Multi-SWE-bench, generate instance_id from org/repo/number before conversion
                org = instance_dict.get("org", "")
                repo = instance_dict.get("repo", "")
                number = instance_dict.get("number", "")
                if org and repo and number:
                    instance_id = f"{org}__{repo}-{number}"
                else:
                    # Fallback to standard format
                    instance_id = instance_dict.get("instance_id", "")
                
                # Apply filter early to avoid unnecessary processing
                if filter_pattern and not filter_pattern.search(instance_id):
                    continue
                
                # Apply image override if available
                if instance_id in image_overrides:
                    instance_dict["image_name"] = image_overrides[instance_id]
                    logger.debug(f"Applied image override for {instance_id}: {image_overrides[instance_id]}")
                
                # Convert to SimpleBatchInstance format
                simple_instance = SimpleBatchInstance.from_swe_bench(instance_dict)
                
                # Convert to full batch instance
                batch_instance = simple_instance.to_full_batch_instance(self.deployment)
                instances.append(batch_instance)
                
                # If we have a specific filter and found a match, we can break early
                # (but only if not shuffling and no slice)
                if filter_pattern and not self.shuffle and not self.slice:
                    break
            
            # Apply slice and shuffle if needed
            return _filter_batch_items(instances, filter_=".*", slice_=self.slice, shuffle=self.shuffle)
        else:
            # Standard non-streaming mode
            ds: list[dict[str, Any]] = load_dataset(dataset_path, split=self.split)  # type: ignore
            instances = [
                SimpleBatchInstance.from_swe_bench(instance).to_full_batch_instance(self.deployment) for instance in ds
            ]
            return _filter_batch_items(instances, filter_=self.filter, slice_=self.slice, shuffle=self.shuffle)

    @property
    def id(self) -> str:
        return f"swe_bench_{self.subset}_{self.split}"


class ExpertInstancesFromFile(BaseModel, AbstractInstanceSource):
    """Load instances from a file. The difference to `InstancesFromFile` is that the instances are configured as full
    `EnvironmentInstanceConfig` objects, i.e., we could specify separate deployment configurations etc.
    """

    path: Path
    filter: str = ".*"
    """Regular expression to filter the instances by instance id."""
    slice: str = ""
    """Select only a slice of the instances (after filtering by `filter`).
    Possible values are stop or start:stop or start:stop:step.
    (i.e., it behaves exactly like python's list slicing `list[slice]`).
    """
    shuffle: bool = False
    """Shuffle the instances (before filtering and slicing)."""

    type: Literal["expert_file"] = "expert_file"
    """Discriminator for (de)serialization/CLI. Do not change."""

    def get_instance_configs(self) -> list[BatchInstance]:
        instance_dicts = load_file(self.path)
        instances = [BatchInstance.model_validate(instance_dict) for instance_dict in instance_dicts]
        return _filter_batch_items(instances, filter_=self.filter, slice_=self.slice, shuffle=self.shuffle)

    @property
    def id(self) -> str:
        return self.path.stem


class SWESmithInstances(BaseModel, AbstractInstanceSource):
    """Load instances from SWE-smith."""

    path: Path

    deployment: DeploymentConfig = Field(
        default_factory=lambda: DockerDeploymentConfig(image="python:3.11"),
    )
    """Deployment configuration. Note that the image_name option is overwritten by the images specified in the task instances.
    """

    filter: str = ".*"
    """Regular expression to filter the instances by instance id."""
    slice: str = ""
    """Select only a slice of the instances (after filtering by `filter`).
    Possible values are stop or start:stop or start:stop:step.
    (i.e., it behaves exactly like python's list slicing `list[slice]`).
    """
    shuffle: bool = False
    """Shuffle the instances (before filtering and slicing)."""

    type: Literal["swesmith"] = "swesmith"
    """Discriminator for (de)serialization/CLI. Do not change."""

    def get_instance_configs(self) -> list[BatchInstance]:
        def convert_instance_dict(instance_dict: dict[str, Any]) -> dict[str, Any]:
            instance_dict["id"] = instance_dict["instance_id"]
            # todo: The base_commit is currently incorrect
            instance_dict["base_commit"] = instance_dict["id"]
            instance_dict["problem_statement"] = instance_dict.get("problem_statement", "")
            instance_dict["repo_name"] = "testbed"
            instance_dict["extra_fields"] = {"fail_to_pass": instance_dict["FAIL_TO_PASS"]}
            return instance_dict

        instance_dicts = load_file(self.path)
        instances = [
            SimpleBatchInstance.model_validate(convert_instance_dict(instance_dict)).to_full_batch_instance(
                self.deployment
            )
            for instance_dict in instance_dicts
        ]
        return _filter_batch_items(instances, filter_=self.filter, slice_=self.slice, shuffle=self.shuffle)

    @property
    def id(self) -> str:
        return f"swesmith_{self.path.stem}"


BatchInstanceSourceConfig = (
    InstancesFromHuggingFace | InstancesFromFile | SWEBenchInstances | ExpertInstancesFromFile | SWESmithInstances
)
