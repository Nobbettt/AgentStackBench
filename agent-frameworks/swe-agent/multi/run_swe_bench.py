#!/usr/bin/env python3
"""
SWE-Bench 统一运行脚本 - 忠实于原脚本逻辑

功能：
1. 单实例测试（完全兼容 run_single_test.py）
2. 批量运行（使用 multirun.py 的 ThreadPoolExecutor）
3. 自适应运行（完整的容器冲突检测逻辑）
4. 轨迹回放（完全兼容 run_replay.py）

用法示例：
  # 单实例测试
  python run_swe_bench.py --mode single --instance_id "repo__name-123" --model gpt4o
  
  # 批量运行
  python run_swe_bench.py --mode batch --data_file data/go.jsonl --model gpt4o --workers 30
  
  # 自适应运行（动态检测容器占用，支持重试）
  python run_swe_bench.py --mode adaptive --data_file data/go.jsonl --model gpt4o --workers 50
  
  # 轨迹回放
  python run_swe_bench.py --mode replay --traj_path trajectories/user/run_name/instance.traj --config config/default.yaml
"""

import os
import sys
import json
import copy
import time
import re
import argparse
import yaml
from pathlib import Path
from typing import Optional
from getpass import getuser
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 设置工作目录
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)
sys.path.insert(0, str(script_dir))

from sweagent import CONFIG_DIR
from sweagent.utils.log import get_logger
from sweagent.agent.agents import AgentArguments
from sweagent.agent.models import ModelArguments
from sweagent.environment.swe_env import EnvironmentArguments
from sweagent.environment.utils import get_instances
from multi_swe_bench.harness.build_dataset import CliArgs
from run import ScriptArguments, ActionsArguments, _ContinueLoop

# 从 multirun 中复制的 Main 类（专门用于单实例）
from pathlib import Path as PathLib
from getpass import getuser
import re
import json

# 需要导入的常量
try:
    from sweagent.environment.swe_env import INSTANCE_LOG_DIR
except ImportError:
    INSTANCE_LOG_DIR = "logs"

from sweagent.agent.agents import Agent
from sweagent.environment.swe_env import SWEEnv
from unidiff import PatchSet

logger = get_logger("swe-bench-unified")

# 自适应模式的全局状态（仅用于adaptive模式）
_lock = threading.Lock()
_completed_instances = set()
_failed_instances = {}
_skipped_count = 0


# 单实例 Main 类（从 multirun.py 复制）
class SingleInstanceMain:
    """用于运行单个实例的 Main 类（从 multirun.py/multirun_adaptive.py 复制）"""
    
    def __init__(self, args: ScriptArguments, filter_instance: str):
        if args.print_config:
            logger.info(f"📙 Arguments: {args.dumps_yaml()}")
        self.args = args
        self.instance_id = filter_instance
        self.traj_dir = PathLib("trajectories") / PathLib(getuser()) / args.run_name
        self.traj_dir.mkdir(parents=True, exist_ok=True)
        if self.should_skip(self.instance_id):
            raise _ContinueLoop
        log_dir = PathLib(INSTANCE_LOG_DIR) / args.run_name / self.instance_id
        if log_dir.exists():
            file_path = log_dir / "log"
            file_path.unlink(missing_ok=True)
        self.agent = Agent("primary", args.agent, log_dir)
        self.env = SWEEnv(args.environment, log_dir)
        
        self._save_arguments()
        # 注意：这里不使用 hooks，因为原 multirun 也不用
        
    def run(self):
        assert isinstance(self.instance_id, str)
        if self.should_skip(self.instance_id):
            raise _ContinueLoop
        logger.info("▶️  Beginning task " + self.instance_id)

        observation, info = self.env.reset(self.instance_id)
        if info is None:
            raise _ContinueLoop

        # Get info, patch information
        issue = getattr(self.env, "query", None)
        files = []
        if self.env.record.instance.pr.fix_patch:
            files = "\n".join([f"- {x.path}" for x in PatchSet(self.env.record.instance.pr.fix_patch).modified_files])
        test_files = []
        if self.env.record.instance.pr.test_patch:
            test_patch_obj = PatchSet(self.env.record.instance.pr.test_patch)
            test_files = "\n".join([f"- {x.path}" for x in test_patch_obj.modified_files + test_patch_obj.added_files])
        tests = ""

        setup_args = {"issue": issue, "files": files, "test_files": test_files, "tests": tests}
        info, trajectory = self.agent.run(
            setup_args=setup_args,
            env=self.env,
            observation=observation,
            traj_dir=self.traj_dir,
            return_type="info_trajectory",
        )
        self._save_predictions(self.instance_id, info)
        self._save_patch(self.instance_id, info)

    def main(self):
        logger.info(f'running the instance id {self.instance_id} now!')
        try:
            self.run()
        except _ContinueLoop:
            logger.info("Skipping instance")
        except KeyboardInterrupt:
            logger.info("Exiting...")
            self.env.close()
        except SystemExit:
            logger.critical("❌ Exiting because SystemExit was called")
            self.env.close()
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            if self.args.raise_exceptions:
                self.env.close()
                raise e
            if self.env.record:
                logger.warning(f"❌ Failed on {self.env.record.data['instance_id']}: {e}")
            else:
                logger.warning("❌ Failed on unknown instance")
            raise

    def _save_arguments(self) -> None:
        log_path = self.traj_dir / "args.yaml"
        if not log_path.exists():
            with log_path.open("w") as f:
                self.args.dump_yaml(f)

    def should_skip(self, instance_id: str) -> bool:
        if re.match(self.args.instance_filter, instance_id) is None:
            return True
        if not self.args.skip_existing:
            return False
        log_path = self.traj_dir / (instance_id + ".traj")
        if log_path.exists():
            with log_path.open("r") as f:
                data = json.load(f)
            exit_status = data["info"].get("exit_status", None)
            if exit_status == "early_exit" or exit_status is None:
                os.remove(log_path)
            else:
                logger.info(f"⏭️ Skipping existing trajectory: {log_path}")
                return True
        return False

    def _save_predictions(self, instance_id: str, info):
        from swebench import KEY_MODEL, KEY_INSTANCE_ID, KEY_PREDICTION
        output_file = self.traj_dir / "all_preds.jsonl"
        model_patch = info["submission"] if "submission" in info else None
        datum = {
            KEY_MODEL: PathLib(self.traj_dir).name,
            KEY_INSTANCE_ID: instance_id,
            KEY_PREDICTION: model_patch,
        }
        with open(output_file, "a+") as fp:
            print(json.dumps(datum), file=fp, flush=True)
        logger.info(f"Saved predictions to {output_file}")
    
    def _save_patch(self, instance_id: str, info):
        patch_output_dir = self.traj_dir / "patches"
        patch_output_dir.mkdir(exist_ok=True, parents=True)
        patch_output_file = patch_output_dir / f"{instance_id}.patch"
        if info.get("submission"):
            patch_output_file.write_text(info["submission"])
            logger.info(f"💾 Patch saved for {instance_id}")



def mark_instance_running(instance_id: str, all_datas: dict):
    """标记实例为运行中（adaptive模式）"""
    with _lock:
        if instance_id in all_datas:
            all_datas[instance_id]["_running"] = True


def mark_instance_done(instance_id: str, all_datas: dict, success: bool, error: str = ""):
    """标记实例完成（adaptive模式）"""
    with _lock:
        if instance_id in all_datas:
            all_datas[instance_id]["_running"] = False
            all_datas[instance_id]["_done"] = True
        
        if success:
            _completed_instances.add(instance_id)
        else:
            _failed_instances[instance_id] = error


def is_instance_available(instance_id: str, all_datas: dict) -> bool:
    """检查实例是否可用（adaptive模式）"""
    with _lock:
        if instance_id in _completed_instances or instance_id in _failed_instances:
            return False
        
        if instance_id in all_datas:
            if all_datas[instance_id].get("_running", False):
                return False
            if all_datas[instance_id].get("_done", False):
                return False
    
    # 检查Docker容器占用
    try:
        import docker
        client = docker.from_env(timeout=10)
        containers = client.containers.list()
        for container in containers:
            if instance_id in container.name:
                return False
    except Exception:
        pass
    
    return True


class SWEBenchRunner:
    """统一的 SWE-Bench 运行器"""
    
    def __init__(self, args):
        self.args = args
        
        # 应用部署配置
        if args.deployment:
            os.environ["AZURE_OPENAI_DEPLOYMENT"] = args.deployment
            logger.info(f"Set AZURE_OPENAI_DEPLOYMENT to: {args.deployment}")
    
    def get_script_args(self, data_file: Optional[str] = None, instance_filter: Optional[str] = None) -> ScriptArguments:
        """构建 ScriptArguments（完全兼容原脚本）"""
        if data_file is None:
            data_file = self.args.data_file
        
        return ScriptArguments(
            suffix=self.args.suffix,
            environment=EnvironmentArguments(
                cli_args=CliArgs(
                    workdir=Path(self.args.workdir),
                    repo_dir=None,
                    pr_file=data_file,
                    need_clone=self.args.need_clone,
                    max_workers_build_image=self.args.max_workers_build,
                    max_workers_run_instance=self.args.max_workers_run,
                    clear_env=False,
                    global_env=[],
                    instance_filter=instance_filter,
                ),
                verbose=self.args.verbose,
                install_environment=True,
                cache_task_images=self.args.cache_images,
            ),
            skip_existing=self.args.skip_existing,
            raise_exceptions=self.args.raise_exceptions,
            print_config=self.args.print_config,
            agent=AgentArguments(
                model=ModelArguments(
                    model_name=self.args.model,
                    total_cost_limit=0.0,
                    per_instance_cost_limit=self.args.cost_limit,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                ),
                config_file=self._resolve_config_path(self.args.config),
            ),
            actions=ActionsArguments(
                open_pr=False,
                skip_if_commits_reference_issue=True
            ),
        )
    
    def _resolve_config_path(self, config: str) -> Path:
        """解析配置文件路径"""
        if config.startswith("/"):
            return Path(config)
        return CONFIG_DIR / config
    
    def run_single(self):
        """单实例运行（完全兼容 run_single_test.py）"""
        logger.info(f"Running single instance: {self.args.instance_id}")
        
        # 为单实例创建临时 JSONL
        temp_file = Path("data/temp_single.jsonl")
        
        # 如果未指定 data_file 或文件不存在，自动搜索
        data_file = None
        if self.args.data_file and Path(self.args.data_file).exists():
            data_file = Path(self.args.data_file)
        else:
            logger.warning(f"Data file {self.args.data_file} not found, searching all JSONL files...")
            data_file = self._find_instance_in_data_files(self.args.instance_id)
            if not data_file:
                logger.error(f"Instance {self.args.instance_id} not found in any data files")
                return False
            logger.info(f"Found instance in: {data_file}")
        
        self._create_single_instance_jsonl(self.args.instance_id, temp_file, data_file)
        
        script_args = self.get_script_args(data_file=str(temp_file))
        
        try:
            # 使用单实例 Main 类
            main_runner = SingleInstanceMain(script_args, self.args.instance_id)
            main_runner.main()
            logger.info("✅ Single instance run completed successfully!")
            return True
        except _ContinueLoop:
            logger.info("⏭️ Instance skipped")
            return True
        except Exception as e:
            logger.error(f"❌ Single instance run failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # 清理临时文件
            if temp_file.exists():
                temp_file.unlink()
    
    def run_batch(self):
        """批量运行（使用 multirun.py 的逻辑）"""
        logger.info(f"Running batch on: {self.args.data_file}")
        
        workers = self.args.workers
        logger.info(f"🚀 启动批量运行，并发数: {workers}")
        
        # 获取所有实例
        cli_args = CliArgs(
            workdir=Path(self.args.workdir),
            repo_dir=None,
            pr_file=self.args.data_file,
            need_clone=self.args.need_clone,
            max_workers_build_image=self.args.max_workers_build,
            max_workers_run_instance=self.args.max_workers_run,
            clear_env=False,
            global_env=[],
        )
        
        # 预构建（如果需要）
        all_datas = get_instances(
            cli_args.pr_file,
            cli_args,
            prebuild=True,  # batch模式预构建
        )
        instance_ids = list(all_datas.keys())
        logger.info(f"📊 Total instances: {len(instance_ids)}")
        
        # 获取post_args（禁用预构建）
        post_args = self.get_script_args()
        post_args.environment.cli_args.instance_filter = None
        
        # ThreadPoolExecutor批量运行
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = [executor.submit(self._run_single_batch, post_args, instance_id) for instance_id in instance_ids]
        
        completed = 0
        failed = 0
        for future in as_completed(futures):
            try:
                success = future.result()
                if success:
                    completed += 1
                else:
                    failed += 1
                logger.info(f"📈 进度: {completed + failed}/{len(instance_ids)} (成功: {completed}, 失败: {failed})")
            except Exception as e:
                failed += 1
                logger.error(f"❌ Task failed with exception: {e}")
        
        logger.info(f"✅ Batch run completed! 成功: {completed}, 失败: {failed}")
        return True
    
    def _run_single_batch(self, script_args: ScriptArguments, instance_id: str) -> bool:
        """运行单个实例（batch模式）"""
        try:
            copy_args = copy.deepcopy(script_args)
            handler = SingleInstanceMain(copy_args, instance_id)
            handler.main()
            return True
        except _ContinueLoop:
            logger.info(f'⏭️ Instance {instance_id} skipped')
            return True
        except Exception as e:
            logger.error(f'❌ Instance {instance_id} failed: {e}')
            return False
    
    def run_adaptive(self):
        """自适应运行（完整的容器冲突检测逻辑，基于 multirun_adaptive.py）"""
        global _completed_instances, _failed_instances, _skipped_count
        
        workers = self.args.workers
        logger.info(f"🚀 启动自适应运行，并发数: {workers}")
        
        # 获取所有实例（预构建）
        cli_args = CliArgs(
            workdir=Path(self.args.workdir),
            repo_dir=None,
            pr_file=self.args.data_file,
            need_clone=self.args.need_clone,
            max_workers_build_image=self.args.max_workers_build,
            max_workers_run_instance=self.args.max_workers_run,
            clear_env=False,
            global_env=[],
        )
        
        all_datas = get_instances(
            cli_args.pr_file,
            cli_args,
            prebuild=True,
        )
        instance_ids = list(all_datas.keys())
        total_instances = len(instance_ids)
        logger.info(f"📊 总实例数: {total_instances}")
        
        # post_args（禁用预构建）
        post_args = self.get_script_args()
        
        # ThreadPoolExecutor批量运行
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = {
            executor.submit(self._run_single_adaptive, post_args, instance_id, all_datas): instance_id 
            for instance_id in instance_ids
        }
        
        # 进度跟踪
        start_time = time.time()
        completed = 0
        
        for future in as_completed(futures):
            instance_id = futures[future]
            try:
                result = future.result()
                completed += 1
                elapsed = time.time() - start_time
                remaining = total_instances - completed
                rate = completed / elapsed if elapsed > 0 else 0
                eta = remaining / rate if rate > 0 else 0
                
                logger.info(
                    f"📈 进度: {completed}/{total_instances} "
                    f"({100*completed/total_instances:.1f}%) "
                    f"| 成功: {len(_completed_instances)} "
                    f"| 失败: {len(_failed_instances)} "
                    f"| 跳过次数: {_skipped_count} "
                    f"| ETA: {eta/60:.1f}min"
                )
            except Exception as e:
                logger.error(f"❌ Instance {instance_id} 异常: {e}")
        
        # 最终统计
        logger.info("=" * 50)
        logger.info(f"🏁 自适应运行完成!")
        logger.info(f"   总实例: {total_instances}")
        logger.info(f"   成功: {len(_completed_instances)}")
        logger.info(f"   失败: {len(_failed_instances)}")
        logger.info(f"   冲突跳过次数: {_skipped_count}")
        
        if _failed_instances:
            logger.info("失败实例（前10个）:")
            for inst, err in list(_failed_instances.items())[:10]:
                logger.info(f"  - {inst}: {err}")
        
        return True
    
    def _run_single_adaptive(self, script_args: ScriptArguments, instance_id: str, all_datas: dict, max_retries: int = 3) -> bool:
        """运行单个实例（adaptive模式，支持重试和冲突检测）"""
        global _skipped_count
        
        for attempt in range(max_retries):
            # 检查是否可用
            if not is_instance_available(instance_id, all_datas):
                with _lock:
                    _skipped_count += 1
                logger.info(f"⏳ 实例 {instance_id} 被占用，跳过 (attempt {attempt + 1})")
                time.sleep(2)
                continue
            
            # 标记为运行中
            mark_instance_running(instance_id, all_datas)
            
            try:
                copy_args = copy.deepcopy(script_args)
                handler = SingleInstanceMain(copy_args, instance_id)
                handler.main()
                mark_instance_done(instance_id, all_datas, success=True)
                logger.info(f"✅ 完成实例 {instance_id}")
                return True
            except _ContinueLoop:
                mark_instance_done(instance_id, all_datas, success=True, error="skipped")
                logger.info(f"⏭️ 实例 {instance_id} 已跳过")
                return True
            except Exception as e:
                error_msg = str(e)
                # 检查是否是容器冲突错误
                if "container" in error_msg.lower() or "conflict" in error_msg.lower():
                    mark_instance_done(instance_id, all_datas, success=False, error="conflict")
                    logger.warning(f"🔄 实例 {instance_id} 遇到容器冲突，将重试")
                    time.sleep(5)
                    continue
                else:
                    mark_instance_done(instance_id, all_datas, success=False, error=error_msg[:100])
                    logger.error(f"❌ 实例 {instance_id} 失败: {error_msg[:100]}")
                    return False
        
        # 重试次数用完
        mark_instance_done(instance_id, all_datas, success=False, error="max_retries")
        return False
    
    def run_replay(self):
        """轨迹回放（完全兼容 run_replay.py）"""
        logger.info(f"Replaying trajectory: {self.args.traj_path}")
        
        traj_path = Path(self.args.traj_path)
        if not traj_path.exists():
            logger.error(f"Trajectory file not found: {traj_path}")
            return False
        
        replay_action_trajs_path = Path("temp_replay.jsonl")
        
        try:
            # 读取轨迹文件，提取actions
            if traj_path.suffix == ".yaml":
                with open(traj_path) as f:
                    traj_data = {"history": yaml.safe_load(f)}
            else:
                with open(traj_path) as f:
                    traj_data = json.load(f)
            
            actions = [x["content"] for x in traj_data["history"] if x["role"] == "assistant"]
            instance_id = traj_path.stem  # 文件名（不含扩展名）
            
            # 写入临时replay文件
            with open(replay_action_trajs_path, 'w') as f:
                json.dump({instance_id: actions}, f)
                f.write('\n')
            
            # 获取 data_path
            data_path = self.args.data_file
            if not data_path:
                # 从 args.yaml 读取
                args_yaml = traj_path.parent / "args.yaml"
                if args_yaml.exists():
                    with open(args_yaml) as f:
                        args_data = yaml.safe_load(f)
                    data_path = args_data.get("environment", {}).get("data_path")
            
            if not data_path:
                logger.error("Could not determine data_path. Please specify --data_file")
                return False
            
            # 创建临时任务实例文件
            replay_task_instances_path = self._create_replay_task_instance(data_path, instance_id)
            if not replay_task_instances_path:
                return False
            
            # 构建 ScriptArguments（使用 replay 模型）
            script_args = ScriptArguments(
                suffix=self.args.suffix or "replay",
                environment=EnvironmentArguments(
                    cli_args=CliArgs(
                        workdir=Path(self.args.workdir),
                        repo_dir=None,
                        pr_file=str(replay_task_instances_path),
                        need_clone=True,
                        max_workers_build_image=4,
                        max_workers_run_instance=4,
                        clear_env=False,
                        global_env=[],
                    ),
                    verbose=self.args.verbose,
                    install_environment=True,
                    cache_task_images=False,
                ),
                skip_existing=False,  # replay 总是执行
                raise_exceptions=self.args.raise_exceptions,
                print_config=self.args.print_config,
                agent=AgentArguments(
                    model=ModelArguments(
                        model_name="replay",
                        total_cost_limit=0.0,
                        per_instance_cost_limit=0.0,
                        temperature=0.0,
                        top_p=0.95,
                    ),
                    config_file=self._resolve_config_path(self.args.config),
                    # 添加replay_path参数
                ),
                actions=ActionsArguments(
                    open_pr=False,
                    skip_if_commits_reference_issue=True
                ),
            )
            
            # 设置replay_path（通过环境变量或其他方式）
            os.environ["SWE_AGENT_REPLAY_PATH"] = str(replay_action_trajs_path)
            
            # 运行
            main_runner = SingleInstanceMain(script_args, instance_id)
            main_runner.main()
            
            logger.info("✅ Replay completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Replay failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # 清理临时文件
            if replay_action_trajs_path.exists():
                replay_action_trajs_path.unlink()
            if replay_task_instances_path and Path(replay_task_instances_path).exists():
                if not str(data_path).startswith(("http://", "https://", "/")):
                    Path(replay_task_instances_path).unlink()
    
    def _create_replay_task_instance(self, data_path: str, instance_id: str) -> Optional[str]:
        """为replay创建任务实例文件"""
        try:
            if data_path.endswith(".jsonl"):
                data = [json.loads(x) for x in Path(data_path).read_text().splitlines(keepends=True)]
            elif data_path.endswith(".json"):
                with open(data_path) as f:
                    data = json.load(f)
            else:
                # GitHub URL 或本地URL，直接返回
                return data_path
            
            # 过滤目标实例
            filtered_data = [d for d in data if d["instance_id"] == instance_id]
            if not filtered_data:
                logger.error(f"Instance {instance_id} not found in {data_path}")
                return None
            
            # 创建临时文件
            tmp_path = f"{instance_id}_replay.jsonl"
            with open(tmp_path, 'w') as f:
                for d in filtered_data:
                    json.dump(d, f)
                    f.write('\n')
            
            return tmp_path
        except Exception as e:
            logger.error(f"Failed to create replay task instance: {e}")
            return None
    
    def _find_instance_in_data_files(self, instance_id: str) -> Optional[Path]:
        """在所有数据文件中查找实例"""
        data_dir = Path("data")
        if not data_dir.exists():
            return None
        
        for jsonl_file in data_dir.glob("*.jsonl"):
            try:
                with open(jsonl_file) as f:
                    for line in f:
                        data = json.loads(line)
                        if data.get("instance_id") == instance_id:
                            return jsonl_file
            except Exception as e:
                logger.debug(f"Error reading {jsonl_file}: {e}")
                continue
        
        return None
    
    def _create_single_instance_jsonl(self, instance_id: str, output_file: Path, data_file: Path):
        """为单实例创建 JSONL"""
        with open(data_file) as f:
            for line in f:
                data = json.loads(line)
                if data.get("instance_id") == instance_id:
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_file, 'w') as out:
                        json.dump(data, out)
                    return
        
        raise ValueError(f"Instance {instance_id} not found in {data_file}")


def main():
    parser = argparse.ArgumentParser(
        description="SWE-Bench Unified Runner - 严格遵循原脚本逻辑",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 运行模式
    parser.add_argument(
        "--mode", 
        choices=["single", "batch", "adaptive", "replay"],
        required=True,
        help="Run mode: single instance, batch, adaptive (with conflict detection), or replay trajectory"
    )
    
    # 数据相关
    parser.add_argument("--data_file", help="Data file (JSONL). For single mode, will auto-search if not provided")
    parser.add_argument("--instance_id", help="Instance ID for single mode")
    parser.add_argument("--traj_path", help="Trajectory file path for replay mode (e.g., trajectories/user/run/instance.traj)")
    
    # 模型相关
    parser.add_argument("--model", default="openai/gpt-4o", help="Model name (e.g., openai/gpt-4o, azure/gpt-4o, gpt4o)")
    parser.add_argument("--deployment", help="Azure OpenAI deployment name")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top P")
    parser.add_argument("--cost_limit", type=float, default=3.0, help="Per instance cost limit")
    
    # 配置相关
    parser.add_argument("--config", default="default.yaml", help="Config file (relative to CONFIG_DIR or absolute path)")
    parser.add_argument("--workdir", default="data_files", help="Work directory")
    parser.add_argument("--suffix", default="", help="Output suffix")
    
    # 运行参数
    parser.add_argument("--workers", type=int, default=30, help="Number of parallel workers (for batch/adaptive modes)")
    parser.add_argument("--max_workers_build", type=int, default=4, help="Max build workers")
    parser.add_argument("--max_workers_run", type=int, default=4, help="Max run workers")
    parser.add_argument("--cache_images", action="store_true", help="Cache task images")
    parser.add_argument("--need_clone", action="store_true", default=True, help="Need clone repos")
    parser.add_argument("--skip_existing", action="store_true", default=True, help="Skip existing results")
    parser.add_argument("--raise_exceptions", action="store_true", help="Raise exceptions for debugging")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--print_config", action="store_true", default=True, help="Print config to log")
    
    args = parser.parse_args()
    
    # 验证参数
    if args.mode == "single" and not args.instance_id:
        parser.error("--instance_id is required for single mode")
    
    if args.mode in ["batch", "adaptive"] and not args.data_file:
        parser.error(f"--data_file is required for {args.mode} mode")
    
    if args.mode == "replay" and not args.traj_path:
        parser.error("--traj_path is required for replay mode")
    
    # 设置默认值
    if not args.data_file:
        args.data_file = "data/go.jsonl"  # 仅作为 fallback
    
    # 创建运行器
    runner = SWEBenchRunner(args)
    
    # 执行对应模式
    mode_handlers = {
        "single": runner.run_single,
        "batch": runner.run_batch,
        "adaptive": runner.run_adaptive,
        "replay": runner.run_replay,
    }
    
    success = mode_handlers[args.mode]()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
