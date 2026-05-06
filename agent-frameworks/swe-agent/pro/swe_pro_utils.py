#!/usr/bin/env python3
"""
SWE-bench Pro 工具集 - 整合轨迹管理工具

功能：
1. 测试运行（test_run_pro.sh）
2. 清理轨迹（clean_traj_no_patch_context.py）
3. 移动已提交实例（move_submitted_to_no_context.py）

用法示例：
  # 运行测试实例（2个ansible实例）
  python swe_pro_utils.py test-run --count 2
  
  # 清理无 patch_context 的轨迹
  python swe_pro_utils.py clean-traj --traj-dir trajectories/gpt-5__missing_pro
  
  # 移动已提交的实例
  python swe_pro_utils.py move-submitted --src-dir trajectories/gpt-5__missing_pro --dst-dir trajectories/gpt-5__no-context
"""

import argparse
import json
import shutil
import subprocess
import sys
import yaml
from pathlib import Path
from typing import List, Optional


class SWEProUtils:
    """SWE-bench Pro 工具集"""
    
    def __init__(self, script_dir: Path):
        self.script_dir = script_dir
        self.swe_bench_pro_dir = Path(os.environ.get("SWE_BENCH_PRO_DIR", "../SWE-bench_Pro-os"))
        self.dockerhub_username = "jefzda"
    
    def test_run(self, count: int = 2, config: str = "config/azure_gpt5_multilingual.yaml"):
        """测试运行（基于 test_run_pro.sh）"""
        print(f"🧪 Running test with {count} ansible instances...")
        
        # 检查 SWE-bench Pro 目录
        if not self.swe_bench_pro_dir.exists():
            print(f"⚠️  SWE_BENCH_PRO_DIR not found: {self.swe_bench_pro_dir}")
            print("Running local smoke test instead...")
            return self._run_smoke_test()
        
        # 读取缺失实例，筛选 ansible
        missing_path = self.script_dir / "missing_pro.txt"
        if not missing_path.exists():
            print(f"❌ Missing file not found: {missing_path}")
            return False
        
        with open(missing_path) as f:
            all_missing = [line.strip() for line in f if line.strip()]
        
        # 筛选 ansible 实例
        ansible_ids = [id for id in all_missing if "ansible" in id.lower()][:count]
        
        if len(ansible_ids) < count:
            print(f"⚠️  Only found {len(ansible_ids)} ansible instances (requested {count})")
        
        print(f"Selected test instances: {ansible_ids}")
        
        # 创建测试实例文件
        test_instances_file = self.script_dir / "test_instances.txt"
        with open(test_instances_file, 'w') as f:
            f.write('\n'.join(ansible_ids))
        
        # 生成 instances.yaml
        full_instances_yaml = self._ensure_instances_yaml()
        if not full_instances_yaml:
            return False
        
        filtered_yaml = self._create_filtered_instances_yaml(
            full_instances_yaml, ansible_ids, "data/test_instances.yaml"
        )
        if not filtered_yaml:
            return False
        
        # 运行测试
        output_dir = self.script_dir / "trajectories" / "gpt-5__test_pro"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "python", "-m", "sweagent.run.run", "run-batch",
            "--config", str(self.script_dir / config),
            "--data", str(filtered_yaml),
            "--output_dir", str(output_dir),
        ]
        
        try:
            subprocess.run(cmd, check=True, cwd=self.script_dir)
            print("✅ Test run completed!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Test run failed: {e}")
            return False
    
    def _run_smoke_test(self):
        """运行本地 smoke test"""
        output_dir = self.script_dir / "trajectories" / "smoke_test_pro"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        simple_instances = self.script_dir / "tests" / "test_data" / "data_sources" / "simple_instances.yaml"
        if not simple_instances.exists():
            print(f"❌ Simple instances file not found: {simple_instances}")
            return False
        
        cmd = [
            "python", "-m", "sweagent.run.run", "run-batch",
            "--config", str(self.script_dir / "config" / "azure_gpt5_multilingual.yaml"),
            "--data", str(simple_instances),
            "--output_dir", str(output_dir),
        ]
        
        try:
            subprocess.run(cmd, check=True, cwd=self.script_dir)
            print("✅ Smoke test passed!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Smoke test failed: {e}")
            return False
    
    def clean_traj(self, traj_dir: str):
        """清理无 patch_context 的轨迹（基于 clean_traj_no_patch_context.py）"""
        base = self.script_dir / traj_dir
        if not base.exists():
            print(f"❌ Trajectory directory not found: {base}")
            return False
        
        traj_files = list(base.rglob("*.traj"))
        to_remove: List[Path] = []
        
        for traj_path in traj_files:
            if not self._has_patch_context(traj_path):
                to_remove.append(traj_path.parent)  # 实例目录
        
        # 去重
        to_remove = sorted(set(to_remove))
        
        print(f"📊 Found {len(traj_files)} .traj files, {len(to_remove)} instances without patch_context")
        
        if not to_remove:
            print("✅ No instances to remove")
            return True
        
        for inst_dir in to_remove:
            print(f"  Removing: {inst_dir.relative_to(base)}")
        
        # 确认删除
        response = input(f"\n⚠️  Delete {len(to_remove)} instance directories? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled")
            return False
        
        # 实例 ID（用于从 preds.json 删除）
        instance_ids = [d.name for d in to_remove]
        
        # 删除实例目录
        for inst_dir in to_remove:
            shutil.rmtree(inst_dir, ignore_errors=True)
        
        # 从 preds.json 删除记录
        preds_file = base / "preds.json"
        if preds_file.exists():
            try:
                with open(preds_file) as f:
                    preds = [json.loads(line) for line in f]
                
                before_count = len(preds)
                preds = [p for p in preds if p.get("instance_id") not in instance_ids]
                after_count = len(preds)
                
                with open(preds_file, 'w') as f:
                    for p in preds:
                        f.write(json.dumps(p) + '\n')
                
                print(f"✅ Updated preds.json: {before_count} → {after_count} entries")
            except Exception as e:
                print(f"⚠️  Failed to update preds.json: {e}")
        
        print(f"✅ Removed {len(to_remove)} instances")
        return True
    
    def move_submitted(self, src_dir: str, dst_dir: str):
        """移动已提交的实例（基于 move_submitted_to_no_context.py）"""
        src = self.script_dir / src_dir
        dst = self.script_dir / dst_dir
        status_file = src / "run_batch_exit_statuses.yaml"
        
        if not status_file.exists():
            print(f"❌ Status file not found: {status_file}")
            return False
        
        # 读取状态文件
        with open(status_file) as f:
            data = yaml.safe_load(f)
        
        by_status = data.get("instances_by_exit_status", {})
        
        # 收集所有 submitted* 的实例
        to_move = []
        for key, instances in by_status.items():
            if instances and isinstance(instances, list) and "submitted" in key.lower():
                to_move.extend(instances)
        
        to_move = sorted(set(to_move))
        print(f"📊 Found {len(to_move)} submitted instances to move")
        
        if not to_move:
            print("✅ No instances to move")
            return True
        
        # 确认移动
        response = input(f"\n⚠️  Move {len(to_move)} instances from {src_dir} to {dst_dir}? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled")
            return False
        
        dst.mkdir(parents=True, exist_ok=True)
        moved = 0
        
        for instance_name in to_move:
            src_inst_dir = src / instance_name
            dst_inst_dir = dst / instance_name
            
            if not src_inst_dir.is_dir():
                print(f"  [SKIP] Source not found: {src_inst_dir}")
                continue
            
            if dst_inst_dir.exists():
                print(f"  [SKIP] Destination exists, removing: {dst_inst_dir}")
                shutil.rmtree(dst_inst_dir, ignore_errors=True)
            
            shutil.move(str(src_inst_dir), str(dst_inst_dir))
            print(f"  ✓ Moved: {instance_name}")
            moved += 1
        
        print(f"✅ Moved {moved} instances to {dst}")
        return True
    
    def _has_patch_context(self, traj_path: Path) -> bool:
        """检查 .traj 文件是否有 patch_context"""
        try:
            data = json.loads(traj_path.read_text())
            info = data.get("info", {})
            pc = info.get("patch_context")
            return bool(pc and isinstance(pc, str) and pc.strip())
        except Exception as e:
            print(f"  [WARN] Failed to read {traj_path}: {e}", file=sys.stderr)
            return False
    
    def _ensure_instances_yaml(self) -> Optional[Path]:
        """确保 instances.yaml 存在"""
        full_yaml = self.swe_bench_pro_dir / "SWE-agent" / "data" / "instances.yaml"
        
        if full_yaml.exists():
            print(f"✓ Using existing instances.yaml: {full_yaml}")
            return full_yaml
        
        # 生成 instances.yaml
        print(f"⚠️  instances.yaml not found, generating...")
        helper_code = self.swe_bench_pro_dir / "helper_code"
        
        if not helper_code.exists():
            print(f"❌ Helper code directory not found: {helper_code}")
            return None
        
        cmd = [
            "python", "generate_instances_yaml.py",
            "--output", str(full_yaml),
        ]
        
        try:
            subprocess.run(cmd, check=True, cwd=helper_code)
            print(f"✓ Generated: {full_yaml}")
            return full_yaml
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to generate instances.yaml: {e}")
            return None
    
    def _create_filtered_instances_yaml(self, source_yaml: Path, instance_ids: List[str],
                                       output_file: str) -> Optional[Path]:
        """创建过滤后的 instances YAML"""
        try:
            with open(source_yaml) as f:
                all_instances = yaml.safe_load(f)
            
            if not isinstance(all_instances, list):
                print(f"❌ Invalid instances.yaml format")
                return None
            
            # 过滤实例
            filtered = [inst for inst in all_instances if inst.get("instance_id") in instance_ids]
            
            print(f"✓ Filtered: {len(filtered)}/{len(all_instances)} instances")
            
            # 写入过滤后的文件
            output_path = self.script_dir / output_file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                yaml.dump(filtered, f, default_flow_style=False)
            
            print(f"✓ Created: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Failed to create filtered yaml: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(
        description="SWE-bench Pro 工具集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # run-missing 命令
    run_missing = subparsers.add_parser("run-missing", help="Run missing Pro instances")
    run_missing.add_argument("--missing-file", default="missing_pro.txt", help="Missing instances file")
    run_missing.add_argument("--config", default="config/azure_gpt5_multilingual.yaml", help="Config file")
    run_missing.add_argument("--output-dir", help="Output directory")
    
    # test-run 命令
    test_run = subparsers.add_parser("test-run", help="Test run with ansible instances")
    test_run.add_argument("--count", type=int, default=2, help="Number of test instances")
    test_run.add_argument("--config", default="config/azure_gpt5_multilingual.yaml", help="Config file")
    
    # clean-traj 命令
    clean_traj = subparsers.add_parser("clean-traj", help="Clean trajectories without patch_context")
    clean_traj.add_argument("--traj-dir", default="trajectories/gpt-5__missing_pro", help="Trajectory directory")
    
    # move-submitted 命令
    move_submitted = subparsers.add_parser("move-submitted", help="Move submitted instances")
    move_submitted.add_argument("--src-dir", default="trajectories/gpt-5__missing_pro", help="Source directory")
    move_submitted.add_argument("--dst-dir", default="trajectories/gpt-5__no-context", help="Destination directory")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # 创建工具实例
    script_dir = Path(__file__).parent
    utils = SWEProUtils(script_dir)
    
    # 执行命令
    if args.command == "test-run":
        success = utils.test_run(args.count, args.config)
    elif args.command == "clean-traj":
        success = utils.clean_traj(args.traj_dir)
    elif args.command == "move-submitted":
        success = utils.move_submitted(args.src_dir, args.dst_dir)
    else:
        parser.print_help()
        sys.exit(1)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    import os
    main()
