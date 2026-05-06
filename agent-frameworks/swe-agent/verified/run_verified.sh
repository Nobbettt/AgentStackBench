#!/bin/bash
# SWE-bench Verified 批量运行脚本
# 使用方法: ./run_verified.sh [num_workers]
# 例如: ./run_verified.sh 4

set -e

# 配置参数
NUM_WORKERS=${1:-4}  # 默认 4 个并发
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config/azure_gpt5.yaml"
OUTPUT_DIR="${SCRIPT_DIR}/output"
VERIFIED_FILE="${VERIFIED_FILE:-../missing_verified}"
LOG_FILE="${SCRIPT_DIR}/run_verified_$(date +%Y%m%d_%H%M%S).log"

# 切换到 SWE-agent 目录
cd "$SCRIPT_DIR"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 读取实例 ID，构建严格匹配的 filter 参数
INSTANCE_IDS=$(grep -v '^[[:space:]]*$' "$VERIFIED_FILE" | tr '\n' '|' | sed 's/|$//')
INSTANCE_FILTER="^(${INSTANCE_IDS})$"

echo "=========================================="
echo "SWE-bench Verified 批量运行"
echo "=========================================="
echo "并发数: $NUM_WORKERS"
echo "配置文件: $CONFIG_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "实例数量: $(wc -l < "$VERIFIED_FILE")"
echo "日志文件: $LOG_FILE"
echo "=========================================="

# 运行 SWE-agent（强制使用当前仓库源码）
# 注意: SWE-agent 会自动处理仓库隔离，每个实例使用独立的 docker 容器
# remove_container=True 会在实例完成后自动清理容器
PYTHONPATH="$SCRIPT_DIR" python3 -m sweagent.run.run_batch \
    --instances.type swe_bench \
    --instances.subset verified \
    --instances.split test \
    --instances.filter="$INSTANCE_FILTER" \
    --config "$CONFIG_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_workers "$NUM_WORKERS" \
    2>&1 | tee "$LOG_FILE"

echo "运行完成！输出保存在: $OUTPUT_DIR"
