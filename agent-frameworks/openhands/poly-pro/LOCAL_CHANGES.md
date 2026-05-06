# OpenHands（poly-pro）- 本地变更/验证记录

## 日期
- 2026-02-06

## 说明
本目录用于验证 `Context-Bench/agent-frameworks/openhands/poly-pro/` 这套 OpenHands 代码在当前机器上的“最小可运行性”（不涉及完整前端构建与评测数据集跑通）。

## 本次实际做过的动作
- 使用当前工作区的 venv（Python 3.13.3）执行了可编辑安装：
  - `python -m pip install -U pip setuptools wheel`
  - `python -m pip install -e .`

说明：这会把 `openhands-ai==1.2.1`（以及其依赖）安装到工作区 venv 中。

## 冒烟验证结果
### 1) Python 导入正常
- `python -c "import openhands; print(getattr(openhands, '__version__', 'unknown'))"`
  - 输出：`1.2.1`

### 2) 可运行的后端入口存在
- `python -m openhands.agent_server --help` 正常输出 help
- `python -m openhands.agent_server --host 127.0.0.1 --port 8010` 可以启动
  - 启动后 API docs：`http://127.0.0.1:8010/docs`

### 3) UI server 入口当前不可直接跑通（预期）
- `python -m openhands.server --help` 会尝试启动 uvicorn 并加载前端静态目录
- 当前报错：`RuntimeError: Directory './frontend/build' does not exist`

结论：后端能启动，但 UI server 需要先构建前端产物（`frontend/build`）。

## 运行时告警（不阻塞启动，但会降级能力）
- 未设置 `OH_SECRET_KEY`：secrets 不会跨重启持久化
- VSCode server binary 缺失：VSCode 集成会被禁用
- Chromium 未安装：browser 工具 preload 失败（agent server 仍能启动）

## 若要完整跑通（前端 + UI server + 评测）
- 需要 Node.js 22.x+（仓库 Makefile 里有版本要求）以及 npm
- 需要按仓库说明执行 `make build` 来生成 `frontend/build`
- 若要启用 browser 能力，需要安装 Playwright Chromium（或系统 Chromium）

## 备注
- 本文件记录的是“本地安装/验证动作与结果”，本次未对 poly-pro 目录下代码做功能性修改。
