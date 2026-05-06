# OpenHands Multi（MopenHands）- 本地修改记录

## 修改日期
- 2026-02-06

## 背景
本目录下的脚本 `scripts/prepare_multi_csv_dataset.py` 用于把 Multi-SWE-bench（按 `Multi.csv` 选取的实例）转换为 MopenHands 推理/评测所需的 JSONL 格式。

在当前工作区的“裁剪版数据”（只含少量 `js/test.jsonl`）上做冒烟运行时发现：脚本会把输入实例里已经存在、且更可信的字段覆盖掉，导致输出 JSONL 出现 `repo` 误解析、`problem_statement`/`base_commit`/`version` 为空等问题，影响后续 checkout 与任务描述构建。

## 修改内容

### 1) 修复字段映射优先级（核心改动）
修改文件：
- `Context-Bench/agent-frameworks/openhands/multi/scripts/prepare_multi_csv_dataset.py`

修改位置：
- 函数 `convert_instance(instance, csv_row)`

调整后的字段优先级：

- `repo`
  1. 如果存在 `instance["org"]` + `instance["repo"]` → 组合为 `org/repo`
  2. 否则如果 `instance["repo"]` 本身已是 `org/repo`（包含 `/`）→ 直接保留
  3. 否则如果 CSV 行里有 `repo` → 使用 CSV 的 `repo`
  4. 最后才从 `instance_id` 推断（仅兜底，可能不可靠）

- `problem_statement`
  - 若实例原本有 `problem_statement` → 直接保留
  - 否则回退到从 `resolved_issues`（title/body）或 `body` 组装

- `version`
  - 仍优先用 CSV 的 `commit`，其次 `base.sha`
  - 若二者都缺失，则回退使用实例原 `base_commit`

- `base_commit`
  - 仍优先用 `base.sha`
  - 若缺失则回退使用实例原 `base_commit`
  - 再不行才用 CSV `commit` 或 `version`

- `language`
  - 仍优先使用 CSV 的 `language`（用于与 `Multi.csv` 对齐）
  - 若 CSV 缺失，则回退到实例原 `language` 或 `repo_language`

### 2) 冒烟运行验证
使用当前工作区自带的最小输入进行验证：

输入：
- Multi.csv：`Context-Bench/agent-frameworks/agentless/multi/data/Multi.csv`
- dataset-root：`Context-Bench/agent-frameworks/agentless/multi/data/multi-swe-bench/`（当前只包含 `js/test.jsonl`）

输出：
- `Context-Bench/agent-frameworks/openhands/multi/temp_multi_subset.jsonl`

验证点：输出 JSONL 能正确保留关键字段（不再被置空或误解析），例如：
- `repo: iamkun/dayjs`
- `base_commit: test_commit_hash`
- `problem_statement: Test problem statement`
- `version` 回退为 `test_commit_hash`

## 运行命令（参考）
在 `Context-Bench/agent-frameworks/openhands/multi/` 下运行：

```bash
python scripts/prepare_multi_csv_dataset.py \
  --multi-csv /home/yukino/Desktop/anonymous/Context-Bench/agent-frameworks/agentless/multi/data/Multi.csv \
  --dataset-root /home/yukino/Desktop/anonymous/Context-Bench/agent-frameworks/agentless/multi/data/multi-swe-bench \
  --output /home/yukino/Desktop/anonymous/Context-Bench/agent-frameworks/openhands/multi/temp_multi_subset.jsonl
```

## 备注
- 由于当前 `dataset-root` 只包含 `js/` 子集，脚本只会匹配到极少数实例（日志中提示缺失大量实例是预期现象）。
- 若换成完整 Multi-SWE-bench 数据集路径（`c/cpp/go/java/js/python/rust/ts` 目录齐全），脚本会扫描全部语言目录下的 `*.jsonl` 并生成更完整的子集。
