# Deep Research Agent

Deep Research Agent 是一个用于长链路研究任务的 Python 实验框架。它把问题拆成 DAG，
并发调用检索与分析 Agent，保留来源和证据关系，再生成带引用的 Markdown 报告。

这个项目目前更适合作为研究与评测代码库，而不是开箱即用的生产服务。我们关心的重点是：
研究过程能否追踪、预算能否控制、失败能否复现，以及一项策略是否真的通过了对照实验。

> 当前状态：Alpha。实时搜索结果会受供应商、时间和网络状态影响；实验性策略默认关闭。
> 仓库不会把“代码已实现”等同于“质量已经提升”。

## 主要能力

- **DAG 研究编排**：Planner 将问题拆成有依赖关系的子任务，Orchestrator 按拓扑层并发执行；失败、超时和证据不足可以触发有界重规划。
- **证据链路**：检索结果写入 lossless Evidence Ledger，使用稳定 source ID 连接来源、claim 和最终引用；模型侧只接收有数量与字符上限的证据目录。
- **搜索预算控制**：区分单任务限额和整次运行限额，记录 query rewrite、provider fallback、重复率和调用量等 telemetry。
- **报告校验与修复**：可选的 claim-level 证据验证会标记 supported、unsupported 和 conflict；Red/Blue 流程用于发现问题并做受限修复。
- **多后端模型路由**：Planner、Researcher、Summarizer、Judge 等模块可以分别选择 DeepSeek、MiMo、OpenAI 兼容接口或本地 vLLM。
- **评测与消融**：包含 ResearchBench、HotpotQA 适配、规则指标、LLM Judge、paired evaluation、bootstrap 置信区间和模块消融脚本。

## 最近加入的实验能力

### Research State shadow controller

`ResearchStateGraph` 在执行 DAG 之外维护另一层研究状态：facet、claim、未解决问题、
来源多样性和冲突。它能对下面这些动作做确定性评分：

- `search_new_facet`
- `deepen_claim`
- `open_primary_source`
- `cross_validate`
- `resolve_conflict`
- `stop`

状态可以序列化和重放，frontier 的候选分数与 stop gate 会写入 telemetry。

目前该能力处于 **shadow 阶段**：默认关闭；启用后只观察现有执行过程，不改写 DAG、
不增加搜索调用，也不会提前终止研究。这样可以先验证决策轨迹，再考虑开放主动控制。

```bash
python scripts/run_eval.py \
  --benchmark research_bench \
  --num_questions 5 \
  --research-state-shadow
```

### 可审计的 Harness Evolution

仓库提供两条离线演化轨道：

- `search_control_policy`：搜索决策、阈值和动作；
- `verify_numeric_claim_skill`：数字、百分比、金额、单位和日期验证。

候选产物采用版本化 YAML，并分别维护 `production`、`candidate` 和 `canary` registry。
评测支持冻结 fixture、record/replay、held-out paired evaluation 和晋升门控。即使候选通过门控，
也只有显式执行 `promote` 才会改变 production 指针。

这套机制的目标是让策略变更可比较、可回滚，而不是宣称系统已经可以自主提升质量。
现有实验尚未证明策略一或 Research State shadow 相对 baseline 有稳定收益，详见
[实验记录](docs/2026-09-14.md)。

## 处理流程

```text
Query
  -> Planner 生成 DAG
  -> Orchestrator 分层并发执行 Worker
  -> Web / Paper / Browser 工具收集来源
  -> Evidence Ledger 固化 provenance
  -> Compressor 控制上下文预算
  -> Summarizer 生成报告与引用
  -> Claim-Evidence 验证；必要时局部 replan
  -> 可选 Red/Blue 审查与修复
  -> Report + telemetry
```

Research State shadow、Search Controller 和 Harness Registry 都围绕这条主链路工作，
但不会绕过现有的预算和晋升边界。

## 快速开始

### 1. 安装

要求 Python 3.10 或更高版本。

```bash
git clone https://github.com/LINYUNLinYun/Deep-Research-Agent.git
cd Deep-Research-Agent

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

也可以使用 `uv venv` 创建环境。

### 2. 配置模型和搜索服务

```bash
cp .env.template .env.local
```

在 `.env.local` 中填写实际使用的模型与搜索 provider 凭证。默认配置使用 DeepSeek 作为
主要研究后端，并为部分模块配置 MiMo；搜索默认使用 Bocha。若要调整组合，请修改
[`configs/default.yaml`](configs/default.yaml) 中的 `model.backend_mapping` 和 `tools` 配置。

`.env.local` 已被 Git 忽略，不要把 API key 写入 YAML 或提交到仓库。

### 3. 运行一次研究

```bash
python scripts/run_single.py \
  --query "比较三种长上下文 Agent 的证据管理方法" \
  --config configs/default.yaml
```

报告默认保存到 `outputs/reports/`。默认配置会调用多个外部服务，并可能产生 API 费用；
正式批量运行前建议先检查并发数、搜索上限、模型映射和对抗轮数。

### 4. 使用 REPL

```bash
python scripts/run_repl.py --config configs/default.yaml
```

REPL 会在同一 session 中复用模块和记忆，支持 `sessions`、`ls`、`save` 和 `help`。

## 配置入口

主要配置集中在 [`configs/default.yaml`](configs/default.yaml)：

| 配置段 | 用途 |
| --- | --- |
| `model.backend_mapping` | 为 planner、solver、summarizer、judge 等模块分配后端 |
| `orchestrator` | 并发数、全局超时、子任务数量和重规划上限 |
| `planner` | 规划重试、证据增益阈值和 Research State 开关 |
| `compressor` | 上下文长度、输出保留量和多级压缩阈值 |
| `memory` | SQLite 路径、session 隔离、去重和检索参数 |
| `adversarial` | Red/Blue 轮数、超时、修复范围和证据验证 |
| `summarizer` | 模型侧来源目录的数量与字符预算 |
| `tools.search_control` | query 去重、rewrite 和 run-level provider 上限 |
| `harness_evolution` | registry 位置与线上使用的策略版本 |

## 评测

先运行单元测试：

```bash
python -m pytest -q
```

使用 HotpotQA 内置 mock 数据检查评测链路，不需要下载数据集：

```bash
python scripts/run_eval.py \
  --benchmark hotpotqa \
  --use_mock \
  --num_questions 3
```

运行 ResearchBench 或消融实验：

```bash
python scripts/run_eval.py --benchmark research_bench --num_questions 10
python scripts/run_ablation.py --mode module --questions 10
```

实时搜索的 A/B 结果容易受到 provider 熔断和时间顺序影响。需要比较策略时，优先使用冻结
fixture，或者交替运行同题 baseline/candidate，并单独记录 provider failure。

## Harness Evolution 工作流

```bash
# 从评测产物提取失败经验
python scripts/run_harness_evolution.py mine \
  --input outputs/evaluation/results.json

# 生成受约束候选
python scripts/run_harness_evolution.py propose \
  --track policy \
  --use-llm

# 在 held-out 数据上评测；Policy 评测建议提供 --fixture-dir
python scripts/run_harness_evolution.py evaluate \
  --track skill \
  --split held_out

# 生成晋升建议
python scripts/run_harness_evolution.py report \
  --experiment outputs/harness_evolution/<experiment_id>

# 人工确认后显式晋升
python scripts/run_harness_evolution.py promote \
  --decision <promotion_decision.json>
```

没有匹配 fixture 的 Policy replay 会 fail-fast，不会偷偷回退到真实网络。

## 仓库结构

```text
configs/                    默认配置、Agent 配置、版本化 Policy/Skill
evaluation/                 benchmark、指标和统计分析
scripts/                    单次运行、评测、消融和演化 CLI
src/agents/                 Researcher 与 Summarizer
src/orchestrator/           状态机、DAG 调度和结果汇总
src/planner/                Planner、预算追踪和 ResearchStateGraph
src/evidence/               Evidence Ledger、schema 和 claim verifier
src/adversarial/            Red/Blue 审查与修复
src/harness_evolution/      registry、replay、candidate 和 promotion gate
src/memory/                 session memory 与向量检索
src/models/                 模型路由和 vLLM 策略
src/tools/                  搜索、浏览、论文和本地工具
tests/                      单元与回归测试
```

更细的模块说明见 [`ARCHITECTURE.md`](ARCHITECTURE.md)。近期实验与设计取舍记录在
[`docs/`](docs/) 目录。

## 已知限制

- 这是研究代码，配置和接口仍可能变化。
- 引用与 claim verifier 能提高可追踪性，但不能保证来源本身正确，也不能替代人工核查。
- Red/Blue 修复会增加模型调用，是否提升最终质量取决于模型、问题和证据质量。
- Research State 目前只做 shadow 观测，尚未接管 follow-up DAG 或停止决策。
- Harness Evolution 是离线、受门控的策略迭代，不是在线自动训练系统。
- 使用实时 provider 的实验无法天然复现；严肃比较需要固定数据、fixture 和运行顺序。

## 参与开发

欢迎提交 Issue 或 PR。对于行为变更，请同时提供回归测试；对于声称提升质量的策略，
请附 baseline、candidate、数据切分、失败样本和统计结果。
