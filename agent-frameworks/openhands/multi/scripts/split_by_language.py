#!/usr/bin/env python3
"""
按语言拆分 JSONL 数据集文件
将 multi_subset.jsonl 按 language 字段拆分成多个文件
"""

import argparse
import json
import os
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def split_jsonl_by_language(input_file: str, output_dir: str = None):
    """
    按语言拆分 JSONL 文件
    
    Args:
        input_file: 输入的 JSONL 文件路径
        output_dir: 输出目录，如果为 None 则使用输入文件所在目录
    """
    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        return 1
    
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_file))
        if not output_dir:
            output_dir = '.'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 按语言分组
    instances_by_language = defaultdict(list)
    total_count = 0
    
    logger.info(f"读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                instance = json.loads(line.strip())
                language = instance.get('language', 'unknown')
                if not language:
                    language = 'unknown'
                language = str(language).lower()
                instances_by_language[language].append(instance)
                total_count += 1
            except json.JSONDecodeError as e:
                logger.warning(f"第 {line_num} 行 JSON 解析失败: {e}")
                continue
    
    logger.info(f"总共读取 {total_count} 个实例")
    logger.info(f"发现 {len(instances_by_language)} 种语言: {list(instances_by_language.keys())}")
    
    # 为每种语言创建文件
    input_basename = os.path.splitext(os.path.basename(input_file))[0]
    
    for language, instances in instances_by_language.items():
        output_file = os.path.join(output_dir, f"{input_basename}_{language}.jsonl")
        
        logger.info(f"写入 {len(instances)} 个 {language} 实例到 {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for instance in instances:
                f.write(json.dumps(instance, ensure_ascii=False) + '\n')
        
        logger.info(f"✓ 已创建: {output_file} ({len(instances)} 个实例)")
    
    # 打印统计信息
    logger.info("\n" + "=" * 60)
    logger.info("拆分统计")
    logger.info("=" * 60)
    for language in sorted(instances_by_language.keys()):
        count = len(instances_by_language[language])
        logger.info(f"{language:15s}: {count:3d} 个实例")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='按语言拆分 JSONL 数据集文件'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入的 JSONL 文件路径'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录（默认为输入文件所在目录）'
    )
    
    args = parser.parse_args()
    
    return split_jsonl_by_language(args.input, args.output_dir)


if __name__ == '__main__':
    exit(main())
