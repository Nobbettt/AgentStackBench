#!/usr/bin/env python3
"""
数据预处理脚本：从 Multi-SWE-bench 数据集中提取 Multi.csv 指定的实例
并转换为 MopenHands 期望的格式。
"""

import argparse
import json
import os
import glob
import pandas as pd
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_instance(instance: dict, csv_row: pd.Series) -> dict:
    """将 Multi-SWE-bench 实例转换为 MopenHands 期望的格式"""
    converted = instance.copy()
    
    # 1. repo: 组合 org/repo
    org = instance.get('org', '')
    repo = instance.get('repo', '')
    if org and repo:
        converted['repo'] = f"{org}/{repo}"
    elif csv_row.get('repo') and pd.notna(csv_row.get('repo')):
        converted['repo'] = csv_row.get('repo')
    else:
        # 如果都没有，尝试从 instance_id 解析
        instance_id = instance.get('instance_id', '')
        if '__' in instance_id:
            parts = instance_id.split('__')
            if len(parts) >= 2:
                converted['repo'] = f"{parts[0]}/{parts[1]}"
            else:
                converted['repo'] = instance_id
        else:
            converted['repo'] = instance_id
    
    # 2. version: 使用 base.sha 或 commit
    base_sha = instance.get('base', {}).get('sha', '')
    csv_commit = csv_row.get('commit', '')
    if pd.notna(csv_commit) and csv_commit:
        converted['version'] = str(csv_commit)
    elif base_sha:
        converted['version'] = base_sha
    else:
        converted['version'] = ''
        logger.warning(f"No version found for instance {instance.get('instance_id')}")
    
    # 3. base_commit: 优先使用 base.sha
    if base_sha:
        converted['base_commit'] = base_sha
    elif csv_commit and pd.notna(csv_commit):
        converted['base_commit'] = str(csv_commit)
    else:
        converted['base_commit'] = converted.get('version', '')
        logger.warning(f"No base_commit found for instance {instance.get('instance_id')}")
    
    # 4. problem_statement: 从 resolved_issues 提取
    resolved_issues = instance.get('resolved_issues', [])
    if resolved_issues and len(resolved_issues) > 0:
        issue = resolved_issues[0]
        issue_body = issue.get('body', '')
        issue_title = issue.get('title', '')
        # 组合 title 和 body
        if issue_title and issue_body:
            converted['problem_statement'] = f"{issue_title}\n\n{issue_body}".strip()
        elif issue_body:
            converted['problem_statement'] = issue_body
        elif issue_title:
            converted['problem_statement'] = issue_title
        else:
            converted['problem_statement'] = instance.get('body', '')
    else:
        # 如果没有 resolved_issues，使用 PR body
        converted['problem_statement'] = instance.get('body', '')
    
    # 5. language: 从 Multi.csv 获取
    csv_language = csv_row.get('language', '')
    if pd.notna(csv_language) and csv_language:
        converted['language'] = str(csv_language).lower()
    else:
        converted['language'] = ''
        logger.warning(f"No language found for instance {instance.get('instance_id')}")
    
    # 6. 保留 Multi.csv 的额外字段
    for col in ['status', 'patch_files', 'patch_blocks', 'patch_span', 
                'gold_context_length', 'num_agents']:
        if col in csv_row:
            val = csv_row.get(col)
            if pd.notna(val):
                converted[col] = val
    
    # 7. 确保 instance_id 使用 Multi.csv 的格式
    csv_instance_id = csv_row.get('instance_id', '')
    if pd.notna(csv_instance_id) and csv_instance_id:
        converted['instance_id'] = str(csv_instance_id)
    else:
        converted['instance_id'] = instance.get('instance_id', '')
    
    # 8. 删除动态字典字段（run_infer.py 不使用，且会导致 datasets 库 schema 错误）
    # 这些字段每个实例的字典键都不同，无法创建统一的 schema
    dynamic_dict_fields = ['fixed_tests', 'p2p_tests', 'f2p_tests', 's2p_tests', 'n2p_tests']
    for field in dynamic_dict_fields:
        converted.pop(field, None)  # 安全删除，不存在也不报错
    
    return converted


def load_multi_csv_instances(csv_path: str, dataset_root: str) -> pd.DataFrame:
    """从 Multi-SWE-bench 数据集中加载 Multi.csv 指定的实例"""
    # 1. 读取 Multi.csv
    logger.info(f"Loading Multi.csv from {csv_path}")
    multi_csv = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(multi_csv)} rows from Multi.csv")
    
    # 2. 构建 original_inst_id -> row 映射
    csv_map = {}
    for idx, row in multi_csv.iterrows():
        original_inst_id = row.get('original_inst_id', '')
        if pd.notna(original_inst_id) and original_inst_id:
            csv_map[str(original_inst_id)] = row
    
    logger.info(f"Built mapping for {len(csv_map)} instances")
    
    # 3. 遍历所有 JSONL 文件
    instances = []
    language_dirs = ['c', 'cpp', 'go', 'java', 'js', 'python', 'rust', 'ts']
    
    for lang_dir in language_dirs:
        lang_path = os.path.join(dataset_root, lang_dir)
        if not os.path.exists(lang_path):
            logger.warning(f"Language directory {lang_path} does not exist, skipping")
            continue
        
        jsonl_files = glob.glob(os.path.join(lang_path, "*.jsonl"))
        logger.info(f"Found {len(jsonl_files)} JSONL files in {lang_path}")
        
        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            instance = json.loads(line.strip())
                            instance_id = instance.get('instance_id', '')
                            
                            if instance_id in csv_map:
                                # 4. 字段映射和转换
                                converted = convert_instance(instance, csv_map[instance_id])
                                instances.append(converted)
                                logger.debug(f"Converted instance {instance_id}")
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON at {jsonl_file}:{line_num}: {e}")
                            continue
            except Exception as e:
                logger.error(f"Error reading {jsonl_file}: {e}")
                continue
    
    logger.info(f"Found {len(instances)} matching instances")
    
    # 检查是否有缺失的实例
    found_ids = {inst['instance_id'] for inst in instances}
    expected_ids = set(csv_map.keys())
    missing_ids = expected_ids - found_ids
    
    if missing_ids:
        logger.warning(f"Missing {len(missing_ids)} instances: {list(missing_ids)[:10]}...")
    
    # 5. 转换为 DataFrame
    if not instances:
        logger.error("No instances found! Check your paths and data.")
        return pd.DataFrame()
    
    df = pd.DataFrame(instances)
    
    # 验证必需字段
    required_fields = ['instance_id', 'repo', 'version', 'problem_statement', 
                       'base_commit', 'language']
    missing_fields = []
    for field in required_fields:
        if field not in df.columns:
            missing_fields.append(field)
        elif df[field].isna().any():
            logger.warning(f"Field {field} has missing values")
    
    if missing_fields:
        logger.error(f"Missing required fields: {missing_fields}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Prepare Multi-SWE-bench dataset from Multi.csv index'
    )
    parser.add_argument(
        '--multi-csv',
        type=str,
        required=True,
        help='Path to Multi.csv file'
    )
    parser.add_argument(
        '--dataset-root',
        type=str,
        required=True,
        help='Path to Multi-SWE-bench dataset root directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 验证输入文件存在
    if not os.path.exists(args.multi_csv):
        logger.error(f"Multi.csv file not found: {args.multi_csv}")
        return 1
    
    if not os.path.exists(args.dataset_root):
        logger.error(f"Dataset root directory not found: {args.dataset_root}")
        return 1
    
    # 加载和转换数据
    try:
        df = load_multi_csv_instances(args.multi_csv, args.dataset_root)
        
        if df.empty:
            logger.error("No data to save!")
            return 1
        
        # 创建输出目录
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 保存为 JSONL
        logger.info(f"Saving {len(df)} instances to {args.output}")
        df.to_json(args.output, orient='records', lines=True, force_ascii=False)
        
        logger.info(f"Successfully saved {len(df)} instances to {args.output}")
        
        # 打印统计信息
        logger.info("\n=== Statistics ===")
        logger.info(f"Total instances: {len(df)}")
        logger.info(f"Languages: {df['language'].value_counts().to_dict()}")
        logger.info(f"Required fields present: {all(f in df.columns for f in ['instance_id', 'repo', 'version', 'problem_statement', 'base_commit', 'language'])}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error processing data: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
