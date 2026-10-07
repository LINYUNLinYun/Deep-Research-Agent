<div align="center">

# 🏗️ DeepResearch Agent 架构说明

### 面向代码的模块梳理 —— 每个模块做什么、如何协作、数据如何流动

</div>

> **本文与其他文档的关系**
> [README.md](README.md) 面向使用与宣传；本文面向**代码阅读与二次开发**，逐模块说明职责、关键实现与依赖。
> 本文内容全部对照当前仓库源码整理，并在文末单列一节「[与 README / 注释不一致之处](#10-与-readme--注释不一致之处)」，避免文档美化误导维护者。
>
> 目录规模统计：`src/`（核心源码，约 10k 行）+ `evaluation/`（评测，约 2.5k 行）+ `scripts/` + `configs/` ≈ 1.7 万行。

> **Harness 改造（2026-09）**：当前主线已加入共享 `SearchController`、工具
> retry/fallback/circuit breaker、claim-level `EvidenceVerifier`、状态驱动 replan、
> evidence-gain stopping、自适应 Memory/Compression 和可选动态模型路由。
> 核心决策会写入 `ResearchReport.decision_trace`，便于 paired ablation。

---

## 1. 系统总览

### 1.1 一句话定位

一个 **Query → 结构化 Markdown 研究报告** 的全链路自动化 Agent 系统：`Planner` 把复杂问题拆成 DAG，`Orchestrator`（自研 asyncio 状态机）并发调度多个 Worker Agent 检索信息，`Summarizer` 合成报告，`Critic/Repairer` 对抗降噪，`Memory`/`Compressor` 做记忆与长上下文管理，另有一套评测体系 + 预留的自进化框架。

### 1.2 分层视图

```raw
┌──────────────────────────────────────────────────────────────────────┐
│ 入口层   scripts/            run_single / run_repl / run_eval / …     │
├──────────────────────────────────────────────────────────────────────┤
│ 装配层   src/core/runner.py   load_config → initialize_modules → run  │
│                             （唯一的"组装根"：把所有模块拼起来）      │
├──────────────────────────────────────────────────────────────────────┤
│ 编排层   src/orchestrator/    9 状态状态机 Orchestrator               │
│          src/planner/         调度单元：任务 DAG + 预算               │
│          src/agents/          Worker（Researcher）与 Summarizer      │
├──────────────────────────────────────────────────────────────────────┤
│ 支撑能力 src/memory/          跨 Agent 共享记忆（去重/矛盾/淘汰）     │
│          src/compressor/      L1/L2/L3 语义压缩                      │
│          src/adversarial/     Critic-Repairer 对抗降噪               │
│          src/evolution/       在线自进化框架（默认关闭）             │
├──────────────────────────────────────────────────────────────────────┤
│ 基础设施 src/tools/           7 个检索/计算/读写工具                  │
│          src/models/          ModelRouter 多后端路由 + VLLMPolicy     │
│          src/utils/           .env 加载 / LangSmith 追踪              │
├──────────────────────────────────────────────────────────────────────┤
│ 横切     configs/ YAML 配置 · evaluation/ 评测 · tests/              │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.3 六大模块（README 视角）→ 实际目录对照

| 代号 | 名称 | 实际代码位置 | 当前状态 |
|---|---|---|---|
| M1 | 多智能体编排器 | [src/orchestrator/](src/orchestrator/) | ✅ 已接入主线 |
| M2 | 自适应规划器 | [src/planner/](src/planner/) | ✅ 已接入主线 |
| M3 | 上下文压缩器 | [src/compressor/](src/compressor/) | ✅ 已接入（降级路径） |
| M4 | 跨 Agent 共享记忆 | [src/memory/](src/memory/) | ✅ 已接入主线 |
| M5 | 对抗降噪循环 | [src/adversarial/](src/adversarial/) | ✅ 已接入（按置信度门控） |
| M6 | 在线自进化 | [src/evolution/](src/evolution/) | ⚠️ 框架完整但 GRPO 训练为占位、默认关闭 |

---

## 2. 完整调用链（数据流）

### 2.1 一次研究的端到端流程

```raw
用户 Query
   │
   ▼
scripts/run_single.py ──► runner.load_config() ──► runner.initialize_modules()   # 装配（见 §4.1）
   │        # 初始化顺序：ModelRouter → Planner → Compressor → Memory → Tools
   │        #            → Critic/Repairer/AdversarialLoop → AgentPool → Orchestrator
   ▼
runner.run_research(query, config, modules)                  # src/core/runner.py
   │  构造 RunConfig（并发度/超时/重规划上限/开关）
   ▼
Orchestrator.run(query, config)                              # src/orchestrator/orchestrator.py
   │
   │  ┌─────────── 9 状态状态机主循环 while state not in (DONE, FAILED) ───────────┐
   │  │                                                                            │
   │  │  IDLE ─► PLANNING      Planner.generate_plan() ─► LLM ─► JSON ─► DAG 无环校验
   │  │         PLANNING ─► DISPATCHING
   │  │  DISPATCHING          DAG.get_parallel_groups() 按层；每层 Semaphore+gather 并发
   │  │                       AgentPool.get_agent() ─► ResearcherAgent 多轮 tool-calling
   │  │                       asyncio.wait_for(agent.run(), timeout) 单任务超时
   │  │                       每层 barrier 后立即提交结果；依赖失败则下游短路
   │  │  DISPATCHING ─► COLLECTING
   │  │  COLLECTING           结果写入运行时 dict + M4 持久化；判断是否需重规划
   │  │   ├─ 失败比例/依赖影响/证据增益/预算要求补救 ─► REPLANNING
   │  │   │     Planner.replan(累计结果+动态保留阈值) ─► 新 DAG ─► 回 DISPATCHING
   │  │   └─ 否则 ─► SYNTHESIZING
   │  │  SYNTHESIZING         压缩预算管理 ─► SummarizerAgent ─► ResearchReport
   │  │                       EvidenceVerifier 做 claim-source 对齐；缺口可补规划
   │  │   ├─ enable_adversarial ─► ADVERSARIAL
   │  │   └─ 否则 ─► DONE
   │  │  ADVERSARIAL          配置化 gate；Critic/Repairer 修复后重新评分
   │  │                       ─► DONE
   │  │  DONE / FAILED        终态
   │  └────────────────────────────────────────────────────────────────────────────┘
   ▼
runner._format_report() ─► Markdown（正文 + 元信息 + 参考来源）      # src/core/runner.py
   ▼
runner.save_report() ─► outputs/reports/report_YYYYMMDD_HHMMSS_<query前20字>.md
```

> ⚠️ **注意**：上图是**代码实际顺序**。README 的数据流图把「对抗」画在「合成」之前，与代码相反——实际是 **先 Summarizer 合成整篇报告，再对整篇做 Critic/Repairer 对抗**（[orchestrator.py:429](src/orchestrator/orchestrator.py#L429) `_do_synthesizing` 返回 `ADVERSARIAL` 态）。

### 2.2 Worker Agent 内部的多轮 tool-calling

```raw
ResearcherAgent.run(subtask, context)          # src/agents/researcher.py
   │
   ├─ _is_non_searchable(question)?            # 私人/主观问题 → 跳过检索直接分析
   │
   ├─ policy.set_tools(所有工具 schema)         # 注册 OpenAI function-calling
   ├─ loop 每轮: policy(messages) ─► 解析 tool_calls
   │     └─ _execute_tool(name, args) ─► 对应工具 .execute(**args)  # 结果追加回 messages
   ├─ 无 tool_call → 提取自评 confidence，返回 AgentResult
   └─ 达到 max_turns(默认10) → AgentStatus.TIMEOUT
```

---

## 3. 仓库结构速览（对照实际代码）

```raw
├── configs/                       # YAML 配置中心
│   ├── default.yaml               # 全局唯一生效配置：模型路由/采样/各模块开关与阈值
│   ├── agents/                    # researcher/critic_agent/repairer_agent Prompt 规范（⚠️ 未被代码加载）
│   ├── planner/planner.yaml       # Planner Prompt 与 DAG 约束（⚠️ 未被代码加载）
│   ├── tools/                     # 各工具的声明式参数参考（⚠️ 未被代码加载）
│   └── evolution/                 # grpo_online / reward_shaping（⚠️ 与代码 key 不匹配，参考用）
│
├── src/                           # 核心源码
│   ├── core/                      # 装配与运行框架
│   ├── orchestrator/              # M1 编排器 + AgentPool + 共享数据结构
│   ├── planner/                   # M2 规划器（DAG）
│   ├── agents/                    # Worker（Researcher）/ Summarizer / 基类
│   ├── compressor/                # M3 上下文压缩
│   ├── memory/                    # M4 共享记忆
│   ├── adversarial/               # M5 对抗降噪
│   ├── evolution/                 # M6 自进化（框架）
│   ├── tools/                     # 7 个检索/计算/读写工具
│   ├── models/                    # 多后端 LLM 路由
│   └── utils/                     # env / 追踪
│
├── evaluation/                    # 评测体系
│   ├── benchmarks/                # ResearchBench(35题) / HotpotQA
│   ├── metrics/                   # rule_based / judge_based / composite / stats
│   ├── run_baseline.py            # 旧版基线（与 scripts/run_ablation.py 重叠）
│   ├── analyze_ablation.py        # 消融结果可视化
│   └── report.py                  # EvaluationReport 数据容器
│
├── scripts/                       # 可执行入口（CLI）
├── tests/                         # demo / 环境校验 / 持续测试脚本
└── outputs/                       # 报告输出目录（gitignore 只留目录）
```

---

## 4. 核心模块详解

### 4.1 `src/core/` — 装配层与运行框架

| 文件 | 核心类/函数 | 职责 |
|---|---|---|
| [runner.py](src/core/runner.py) | `load_config` / `initialize_modules` / `run_research` / `save_report` | **装配根**：加载 YAML → 按 M1~M6 顺序实例化所有模块 → 构造 `RunConfig` 调 `Orchestrator.run` → 把 `ResearchReport` 格式化成带元信息的 Markdown 并落盘 |
| [judge.py](src/core/judge.py) | `LLMJudge` | 评测用的统一 LLM-as-Judge：`score_single`（单报告 5 维 0-10 分）、`compare_two`（两报告 4 维 head-to-head），JSON 三级解析容错 |
| [ablation.py](src/core/ablation.py) | `AblationStudy` | 消融实验通用框架：4 个有效模块配置（full / no_adversarial / no_compressor / no_memory）+ Harness 专项消融 + 对抗轮数消融 |

`runner.py` 是唯一的"上帝装配点"——所有模块在此按序构造并互相注入，子模块之间基本不直接 import（用延迟导入规避循环依赖）。理解整个项目**从这里开始**。关键装配代码：[runner.py:127-269](src/core/runner.py#L127-L269)。

`initialize_modules` 的初始化顺序与分工（模块 → 配置注入）：

| 顺序 | 初始化对象 | 使用的后端 policy |
|---|---|---|
| 1 | `default_policy` + 各模块专用 policy（按 `backend_mapping`） | `deepseek` / `mimo` / `vllm` / `openai` |
| 2 | Planner + BudgetTracker | `planner_policy` |
| 3 | ContextCompressor | `compressor_policy` |
| 4 | SharedMemoryStore（session 隔离） | — |
| 5 | 7 个 Tools（mock/真实由 `tools.web_search.mock_mode` 决定） | — |
| 6 | CriticAgent / RepairerAgent / AdversarialLoop | `critic/repairer/judge_policy` |
| 7 | AgentPool + Orchestrator（注入上面全部） | `solver_policy` |

---

### 4.2 `src/orchestrator/` — M1 多智能体编排器（状态机）

**职责**：接收 query，驱动「规划 → 执行 → 收集 → 合成 → 对抗」全流程，管并发、超时与重规划。

| 文件 | 类 | 职责 |
|---|---|---|
| [orchestrator.py](src/orchestrator/orchestrator.py) | `Orchestrator` | 9 状态状态机主循环；`_state_handlers` 字典把状态映射到 handler 方法 |
| [agent_pool.py](src/orchestrator/agent_pool.py) | `AgentPool` | Worker Agent 对象池：按 task_type 复用/新建/健康检查回收（`policy.was_truncated` 判定） |
| [schemas.py](src/orchestrator/schemas.py) | 枚举 + dataclass | 全项目共享数据结构（见 §7） |

**状态机**（[schemas.py:29](src/orchestrator/schemas.py#L29)）：9 态 = `IDLE, PLANNING, DISPATCHING, COLLECTING, SYNTHESIZING, ADVERSARIAL, REPLANNING, DONE, FAILED`。

**关键调度逻辑**（[orchestrator.py:226-301](src/orchestrator/orchestrator.py#L226-L301)）：
- 用 DAG 的 `get_parallel_groups()` 得到**执行层**，层间串行、层内 `asyncio.gather` + `Semaphore(max_concurrent)` 并发。
- 单任务 `asyncio.wait_for(timeout)`，超时/异常包装成 `AgentResult(status=TIMEOUT/FAILED)` 不中断整体。
- 依赖任务结果通过 `_build_task_context` 注入子任务上下文。

**状态驱动降级策略**（实际实现）：
1. 单任务超时/失败 → 标记状态继续跑；
2. 综合失败比例、依赖影响、可用证据、证据新颖度和剩余预算决定是否 `REPLANNING`；
3. **全局超时** → 使用已完成的成功结果生成降级报告，并记录 `force_synthesize` 决策。

**对抗门控**：`_do_synthesizing` 完成后若 `enable_adversarial`，进入 `ADVERSARIAL` 态；`_do_adversarial` 仅在**合成报告 `confidence < 0.8`** 时真正跑对抗循环（[orchestrator.py:446](src/orchestrator/orchestrator.py#L446)），分够高就直接跳过。

---

### 4.3 `src/planner/` — M2 自适应规划器

**职责**：把模糊 query 拆解为带依赖关系的原子子任务 DAG；任务大面积失败时做增量 replan。

| 文件 | 类/函数 | 职责 |
|---|---|---|
| [planner.py](src/planner/planner.py) | `Planner` | `generate_plan`（LLM → JSON → DAG 无环校验）；`replan`（保留 `confidence≥0.6` 的成功结果，针对失败任务生成新 DAG）；Prompt 硬编码于 `INITIAL_PLAN_PROMPT` / `REPLAN_PROMPT` |
| [dag.py](src/planner/dag.py) | `DAG` | 图结构 + Kahn `topological_sort` + `get_parallel_groups`（按层分组供并行调度）；加边检测环 |
| [budget_tracker.py](src/planner/budget_tracker.py) | `BudgetTracker` | Token 预算追踪（默认 100K）：`track` / `get_usage_ratio` / `is_over_budget`，供后续压缩/截断决策 |

**解析健壮性**：`_parse_plan` 容忍 markdown 代码块、尾逗号、`//` 注释；非法 `task_type` 降级为 SEARCH。

**生成的子任务单元** `SubTask`（[schemas.py:66](src/orchestrator/schemas.py#L66)）：`task_id / task_type(search|analyze|verify) / description / dependencies / context_keys / timeout_seconds / priority / expected_type(factual|analytical|comparative|temporal) / search_hints`。

---

### 4.4 `src/agents/` — Worker Agent 与合成 Agent

| 文件 | 类 | 职责 |
|---|---|---|
| [base_agent.py](src/agents/base_agent.py) | `BaseAgent(ABC)` | 抽象基类：持有 `policy`（LLM）与 `tools`，定义 `async run(task, context) -> AgentResult` |
| [researcher.py](src/agents/researcher.py) | `ResearcherAgent` | **检索 Worker**：多轮 tool-calling 循环执行子任务（见 §2.2），维护搜索→抽取→置信度 的逻辑 |
| [summarizer.py](src/agents/summarizer.py) | `SummarizerAgent` | **合成器**：把全部子结果按置信度降序拼 prompt，单轮生成整篇报告；计算综合置信度 `llm_confidence × success_rate^0.5`，从 trajectory 提取 sources 并统计 `num_searches` |

> 设计点：规划出的 `search / analyze / verify` 三类子任务在 AgentPool 中**都由 ResearcherAgent 执行**（只是 prompt 侧重不同）；合成的 SummarizerAgent 由 Orchestrator 在 SYNTHESIZING 态单独构造（[orchestrator.py:389-393](src/orchestrator/orchestrator.py#L389-L393)）。Agent 的 Prompt 硬编码在代码内，`configs/agents/*.yaml` 并未被加载。

---

### 4.5 `src/compressor/` — M3 上下文压缩（语义级，非截断）

**职责**：长上下文按预算逐级压缩，尽量保留语义关键信息。

**三级渐进**（[compressor.py](src/compressor/compressor.py)，按 `usage_ratio = 当前token/预算` 触发）：

| 级别 | 触发 | 算法 | 实现文件 |
|---|---|---|---|
| L1 | > 0.60 | **Cosine 语义过滤**：embedding 相似度阈值自适应放宽 0.25→0.20→0.15，滤到 ≤80% 预算 | [compressor.py](src/compressor/compressor.py) `_l1_filter` |
| L2 | > 0.80 | **TextRank + query-biased 关键句提取**：句子 embedding → PageRank(阻尼0.85) → 融合 query 相关性与高价值句(数字/URL/引用 bonus)，按原文顺序取 top-k | [extractive.py](src/compressor/extractive.py) |
| L3 | > 0.95 | **LLM 抽象摘要**：逐文档摘要 → 聚合摘要（Prompt 要求保留数字/引用/矛盾观点） | [summarizer.py](src/compressor/summarizer.py) `LLMSummarizer` |
| 兜底 | 仍超限 | **FIFO 滑动窗口截断**：保留 system，从旧消息丢弃，保底最近 3 条，末条内容级截断 | [sliding_window.py](src/compressor/sliding_window.py) |

**接入点**：Orchestrator 的 `_build_memory_context` 回退分支中，上下文 >6000 字符时调用 `compressor.compress(...)`（[orchestrator.py:574-584](src/orchestrator/orchestrator.py#L574-L584)）。注意这是**降级路径**，正常路径优先用 M4 语义检索。

---

### 4.6 `src/memory/` — M4 跨 Agent 共享记忆

**职责**：跨 Agent 沉淀中间结论，供后续规划/合成做语义检索；写入时自动去重、检测矛盾；超限淘汰。

**分层结构**：

```raw
SharedMemoryStore（高层语义接口：去重 / 矛盾 / 检索 / 淘汰 / 上下文组装）
    ├─ LongTermMemory（SQLite 持久层，只管 CRUD，无向量计算）
    └─ Embedder（sentence-transformers all-MiniLM-L6-v2，384 维；失败时 MD5 种子确定性向量兜底）
ShortTermMemory（独立对话历史，⚠️ 当前全项目无调用方）
```

| 文件 | 类 | 职责 |
|---|---|---|
| [memory_store.py](src/memory/memory_store.py) | `SharedMemoryStore` | 唯一语义入口；维护 numpy 内存向量索引（加速相似度）与 SQLite 双层 |
| [long_term.py](src/memory/long_term.py) | `LongTermMemory` + `MemoryEntry`/`ConflictRecord` | `entries` / `conflicts` 两表；会话隔离（`session_id`）；淘汰评分 SQL |
| [embedder.py](src/memory/embedder.py) | `Embedder` | 文本→归一化向量（自动设 HF 国内镜像） |
| [short_term.py](src/memory/short_term.py) | `ShortTermMemory` | 会话内多轮对话历史（预留，未接入） |

**关键行为**：
- **写入** `put`：垃圾过滤（<30 字符 / confidence<0.3 / 正则）→ 补 embedding → **去重**（cosine>0.92，新条目置信度更高则合并覆盖）→ 持久化 → 矛盾检测。
- **矛盾检测**：cosine 落在 **(0.65, 0.92)** 区间 **且** 语义对立（否定词不对称+Jaccard，或反义词对命中）→ 生成 `ConflictRecord`。
- **矛盾消解三策略**：`majority_vote`（多数票）/ `source_weight`（evidence_type×confidence 加权）/ `llm_judge`（LLM 二选一，失败降级 source_weight）。
- **淘汰**：综合分 `confidence × evidence_weight × recency(30天半衰期) × conflict_bonus(有 open conflict 翻倍)`，取最低分删；有 open conflict 的条目受保护。
- **Session 隔离**：构造时注入 `session_id`，索引只加载该会话数据；`session_id=""` 表示全局不隔离。

**接入点**：Orchestrator COLLECTING 态把成功子结果 `_sync_result_to_memory_store` 写入（[orchestrator.py:342](src/orchestrator/orchestrator.py#L342)），DONE 态把最终报告也存一份；规划/重规划前用 `get_context_for_query(query, max_tokens=2000)` 做语义检索喂给 Planner。

---

### 4.7 `src/adversarial/` — M5 Critic-Repairer 对抗降噪

**职责**：对**已合成的整篇报告**做「攻击-修复-评分」迭代，主动压低幻觉与逻辑问题（灵感来自 GAN，应用于文本质量）。

| 文件 | 类 | 职责 |
|---|---|---|
| [critic_agent.py](src/adversarial/critic_agent.py) | `CriticAgent` | 五维度攻击方 |
| [repairer_agent.py](src/adversarial/repairer_agent.py) | `RepairerAgent` | 修复方 + `self_verify` |
| [verdict.py](src/adversarial/verdict.py) | `VerdictEngine` + `Issue`/`CriticVerdict`/`FixOperation` | 评分模型、优先级、JSON 序列化 |
| [loop.py](src/adversarial/loop.py) | `AdversarialLoop` | Critic→Repairer→评分 主循环 + 收敛/震荡控制 |

**Critic 攻击五维度与权重**（[verdict.py:188-194](src/adversarial/verdict.py#L188-L194)）：
`FACTUAL 事实性 0.30 / HALLUCINATION 幻觉 0.25 / LOGICAL 逻辑 0.20 / SOURCE_CREDIBILITY 来源可信 0.15 / COVERAGE 覆盖 0.10`，每维独立 LLM Prompt，JSON 四层容错解析。

**Repairer 修复操作**（按优先级 `severity_weight × dimension_weight × fix_difficulty` 降序逐个处理）：
- `IN_PLACE`（难度1.0）：数字/日期/人名与来源不一致 → 直接替换；
- `SUPPLEMENTARY`（难度0.6）：无来源 claim → 先搜索补证，无法证实则标注"未经证实"；
- `REMOVAL`（难度0.8）：高置信幻觉 → 删除段落；
- 每个修复后跑 `self_verify` 检测是否引入新矛盾。

**循环终止三选一**（[loop.py:161-176](src/adversarial/loop.py#L161-L176)）：轮数达上限（默认配置 `max_rounds=10`）／综合分 ≥ `score_threshold`（配置 9.5）／相邻两轮五维欧氏距离 Δ < `delta_threshold`（0.2）——三选一即停。
**震荡检测**：已修复 issue（按 severity+dimension+description+location+fix_type 判等）重新出现 → 判定震荡优雅终止。

**接入点**：Orchestrator `_do_adversarial`（仅当合成报告 `confidence<0.8`）。运行轮数写回 `ResearchReport.adversarial_rounds` 进入输出元信息。

---

### 4.8 `src/evolution/` — M6 旧 GRPO 进化骨架（默认未启用）

**设计目标**（MAE 三角）：`Proposer` 出题 → `Solver`（即整套 DeepResearch Agent）执行 → `Judge` 五维评分 → 产出 `parquet` 喂 **veRL GRPO** 训练，并把经验写回记忆实现越用越强。

| 文件 | 类 | 职责 | 接线状态 |
|---|---|---|---|
| [engine.py](src/evolution/engine.py) | `SelfEvolutionEngine` | 编排一轮轮出题-求解-评分-入库 | ✅ 引擎本身可跑 |
| [proposer.py](src/evolution/proposer.py) | `Proposer` | 按 L1/L2/L3 难度与成功率生成研究问题，embedding 去重 | ✅ |
| [judge.py](src/evolution/judge.py) | `Judge` | Ensemble(3视角) 评分 + `shape_reward` 转单值 reward + 校准 | ✅ |
| [collector.py](src/evolution/collector.py) | `TrajectoryCollector` | 轨迹收集并转 veRL 标准 parquet 行 | ✅ |
| [experience_memory.py](src/evolution/experience_memory.py) | `ExperienceMemory` | SQLite 存"问题-轨迹-得分"，embedding 检索 + 综合分淘汰 | ✅ |
| [symbolic_learning.py](src/evolution/symbolic_learning.py) | `SymbolicLearner` | 从失败轨迹提取错误模式 → LLM 改进 prompt → 变差自动回滚 | ⚠️ 已实现但 `run_evolution.py` 未实例化 |

**关键事实**：
- `evolution.enabled=false`（[default.yaml:180](configs/default.yaml#L180)），主流程默认不进入进化路径。
- 唯一入口 [scripts/run_evolution.py](scripts/run_evolution.py)；veRL GRPO **训练调用仍未接入**，命令默认拒绝运行，只有显式 `--prepare-data-only` 才执行轨迹与 parquet 数据准备，且不会创建伪 checkpoint。
- 五维评分（`factual_accuracy .30 / coverage .25 / logical_coherence .20 / citation_quality .15 / efficiency .10`）+ 规则式 `efficiency`（sigmoid 防 reward hacking）。
- `configs/evolution/grpo_online.yaml` 的 `training:` 已接入 legacy 数据准备参数；`reward_shaping.yaml` 仍无代码消费。

---

### 4.9 `src/tools/` — 工具层

所有工具实现同一接口约定：`name`、`get_openai_tool_schema()`（OpenAI function-calling schema）、`async execute(...)`。

| 工具 | 用途 | 关键点 |
|---|---|---|
| [web_search.py](src/tools/web_search.py) | 网页搜索 | 按 `SEARCH_BACKEND` 选 SerpAPI/Bing/博查/秘塔；aiohttp 共享连接池；URL 规范化去重 |
| [browser.py](src/tools/browser.py) | 抓取网页正文 | web_search 下游；BeautifulSoup 优先 article/main；超长截断加标记 |
| [arxiv_reader.py](src/tools/arxiv_reader.py) | 读论文元数据 | 三后端：ArXiv / Semantic Scholar / OpenAlex；OpenAlex 倒排摘要还原 |
| [calculator.py](src/tools/calculator.py) | 数学计算 | AST 白名单 `_safe_eval`（禁 import/黑名单函数），轻量安全 |
| [code_sandbox.py](src/tools/code_sandbox.py) | 受限代码执行 | 静态扫描禁 `Import`/黑名单 + `eval(code, {"__builtins__":{}})` + asyncio 超时；仅单表达式（文档注明生产可换 Docker） |
| [file_reader.py](src/tools/file_reader.py) | 读本地文件 | 四重校验：目录/存在性/扩展名白名单/大小上限(10MB)；txt/md/pdf/csv/json/docx |
| [notepad.py](src/tools/notepad.py) | Agent 草稿纸 | 会话级非持久化；write/read/list/search/clear，帮长时程研究记住早期结论 |

> mock 开关：`configs/default.yaml → tools.web_search.mock_mode`（默认 true）→ `runner` 据此注入 `MockWebSearchTool/MockBrowserTool`，无 API key 也能全流程演示。

---

### 4.10 `src/models/` — 多后端 LLM 路由

**职责**：让"模块 ↔ 后端模型 ↔ 采样参数"三者解耦，改配置不改代码即可换模型/热切换。

| 文件 | 类 | 职责 |
|---|---|---|
| [model_router.py](src/models/model_router.py) | `ModelRouter` | 从 `.env` 读 `{PREFIX}_API_KEY/_BASE_URL/_MODEL` 建后端；实例缓存；支持任意 OpenAI 兼容后端 |
| [vllm_policy.py](src/models/vllm_policy.py) | `VLLMPolicy` / `OpenAICompatibleDict` | 所有后端落到的**统一 LLM 封装**：消息清洗（修复元组/防 task 泄露/合并连续同角色）、35000 字符截断、原生 tool_calls + `<tool_call>` 标签正则兜底解析、错误分类（context length 错误中止轨迹，网络抖动返回假 assistant 继续） |

**模块级分工**（`configs/default.yaml:99-107`）：`solver/planner/summarizer → deepseek`（强推理/大输出），`judge/critic_agent/repairer_agent/compressor → mimo`（稳定低成本）。未在映射中的模块回退 `default_policy`。

**采样参数集中管理**（[default.yaml:58-94](configs/default.yaml#L58)）：`model.backend_sampling` = 后端全局默认（deepseek: temp0.7/max_tokens4096，mimo: temp0.3）+ `modules` 级覆盖（如 judge temp0.1 保评分一致、summarizer max_tokens 16384 容纳长报告）。优先级：模块覆盖 > YAML 后端默认 > 内置默认。

**运行时热切换**：`create_backend(name, **override_kwargs)` 按配置名实例化，相同配置走缓存。

---

### 4.11 `src/utils/` — 基础设施

| 文件 | 职责 |
|---|---|
| [env_config.py](src/utils/env_config.py) | `.env` → `.env.local`（后者优先）幂等加载；类型化读取 `get_env_int/bool/float` |
| [tracing.py](src/utils/tracing.py) | LangSmith 可观测性：`maybe_wrap_openai_client` 包装 OpenAI client、`@trace_agent/trace_tool/trace_chain/trace_retriever` 装饰器、`trace_block` 上下文管理器；`LANGSMITH_TRACING` 关闭时零开销 |

---

### 4.12 `src/harness_evolution/` — 冻结模型的 Policy / Skill 自进化 V1

这是与 `src/evolution/` 完全隔离的离线外循环。它不训练模型，也不修改 Python 源码，只进化两类经过白名单校验的 YAML 工件：

- `search_control_policy`：控制 `SearchController.after_search` 的 accept/rewrite/switch/verify/stop 决策；
- `verify_numeric_claim_skill`：组合固定 primitive，保守验证数字、百分比、金额单位、日期和财年 claim。

```text
运行轨迹 → Failure Miner → LLM 修改假设（可选）
        → 程序枚举最多 5 个单变量候选 → immutable candidate
        → frozen replay / held-out paired evaluation
        → Promotion Gate → 人工 promote 或 reject
```

关键隔离边界：

- `production.yaml`、`candidate.yaml`、`canary.yaml` 只保存版本指针；工件采用排他创建，读取时校验 SHA-256；
- 默认配置 `harness_evolution.enabled=false`，启用时仍只读 production，candidate 必须由离线 CLI 显式选择；
- 每个 paired run 重建 modules、Model、AgentPool、Memory session 和 SearchController；
- replay 未覆盖请求会硬失败，已录制 timeout/429/5xx 则作为冻结实验条件正常回放；
- Manifest 固化 config、Git、dataset、fixture、evaluator 与工件版本/hash，不写入最终 Markdown；
- held-out 数据不能进入 `mine` 或 `propose`，且晋升必须再次校验 decision hash、candidate hash 和 production parent。

入口是 [run_harness_evolution.py](scripts/run_harness_evolution.py)，而不是旧 [run_evolution.py](scripts/run_evolution.py)。当前示例保留 production Skill `v0001`，candidate Skill `v0002`；held-out 结果为 24 tie、无显著提升，因此正确地被拒绝晋升。

---

## 5. 配置体系（`configs/`）

**真正生效的只有 [configs/default.yaml](configs/default.yaml)**，由 `runner.load_config` 加载。顶层结构：

| 顶层段 | 内容 |
|---|---|
| `model` | `backend` 默认后端 + `backend_sampling`（后端/模块采样）+ `backend_mapping`（模块→后端分工） |
| `orchestrator` | max_concurrent / global_timeout / max_replan_rounds / max_sub_questions |
| `planner` | 每子 Agent 搜索轮数上限、replan 开关、完整性检查 |
| `compressor` | max_context_length=16000、L1/L2/L3 阈值、embedding_model |
| `memory` | db_path=`data/memory.db`、去重/矛盾阈值、淘汰间隔、向量索引开关 |
| `adversarial` | enabled / max_rounds=10 / score_threshold=9.5 / delta_threshold=0.2 |
| `evolution` | enabled(**false**) / 各训练超参 |
| `tools` | web_search(mock_mode)/arxiv_reader/code_sandbox 的开关与参数 |

**⚠️ 只读参考、未被代码加载的 YAML**（代码中 Prompt 硬编码，这些文件更像"规范文档"）：
`configs/agents/researcher.yaml`、`critic_agent.yaml`、`repairer_agent.yaml`（含很详细的攻击维度/评分框架/修复模板）、`configs/planner/planner.yaml`、`configs/tools/*.yaml`。其中 critic_agent 的权重（0.30/0.20/0.20/0.15/0.15）与代码 `verdict.py` 的权重定义**并不一致**，修改时以代码为准。

**环境变量**：见 [.env.template](.env.template)——LLM 各后端 `{PREFIX}_API_KEY/_BASE_URL/_MODEL/_TEMPERATURE/_MAX_TOKENS`、搜索后端（SerpAPI/Bing/博查/秘塔）、`SEARCH_BACKEND`、arxiv 三后端、LangSmith、文件/沙箱限制等。

---

## 6. 评测体系与运行入口

### 6.1 评测分四层（`evaluation/`）

| 层 | 位置 | 指标/方法 |
|---|---|---|
| 📏 规则指标 | [metrics/rule_based.py](evaluation/metrics/rule_based.py) | `fact_accuracy`（字符串）/ `semantic_fact_accuracy`（embedding cosine，阈值0.65）/ `hallucination_rate` / `citation_coverage` / `logical_consistency` / `comprehensiveness` / `composite_score`（加权0.25/0.20/0.20/0.20/0.15）/ `efficiency_score`（sigmoid）——全部本地可复现、零 API 成本 |
| 👨‍⚖️ LLM-as-Judge | [metrics/judge_based.py](evaluation/metrics/judge_based.py) + [core/judge.py](src/core/judge.py) | MiMo 评 5 维 0-10（factual/logical/citation/comprehensiveness/overall）；`compare_two` 做 head-to-head |
| 🧮 综合 | [metrics/composite.py](evaluation/metrics/composite.py) | `rule 0.6 + judge 0.4` 加权归一 |
| 📈 统计显著性 | [metrics/stats.py](evaluation/metrics/stats.py) | 配对/双样本 Bootstrap 95% CI、Cohen's d、paired t-test（无 scipy 降级 bootstrap） |

**评测集**：
- [benchmarks/research_bench.py](evaluation/benchmarks/research_bench.py)：**35 题 × 11 领域**（文件头注释"20 道"已过时），每题含 `query / expected_topics / ground_truth`。
- [benchmarks/hotpotqa.py](evaluation/benchmarks/hotpotqa.py)：标准 EM/F1/pass@1 + **深度研究变体**（`gold_entity_coverage`、`semantic_gold_coverage`，评测"整篇报告质量"而非短答案）；数据可 mock/本地/HuggingFace 三源。

### 6.2 运行入口（`scripts/`）

| 脚本 | 用途 |
|---|---|
| [run_single.py](scripts/run_single.py) | 单条 query 跑全流程，输出报告（另写 `run_*.log`） |
| [run_repl.py](scripts/run_repl.py) | 交互式 REPL；`ls/sessions/save/q`；单进程复用 Orchestrator + 记忆，session 可继承/新建 |
| [run_eval.py](scripts/run_eval.py) | 标准评测：ResearchBench 或 HotpotQA，汇总平均综合分 |
| [run_ablation.py](scripts/run_ablation.py) | 模块消融（4 个有效配置）、Harness 专项消融与对抗轮数消融，带 bootstrap 显著性 |
| [run_benchmark.py](scripts/run_benchmark.py) | **Agent vs 单轮 LLM** head-to-head（A=baseline 单轮 deepseek，B=agent 全流程），4 维配对显著性 |
| [run_judge.py](scripts/run_judge.py) | 对已有报告文件跑单次 MiMo Judge 深度评分 |
| [run_all_experiments.py](scripts/run_all_experiments.py) | 批量编排：subprocess 串行跑 7 类实验（约 165 次研究运行），汇总 `SUMMARY.md` |
| [run_evolution.py](scripts/run_evolution.py) | Legacy M6 数据准备入口；需显式 `--prepare-data-only`，当前不执行模型训练 |
| [run_harness_evolution.py](scripts/run_harness_evolution.py) | 离线 Policy/Skill mine、propose、evaluate、report、人工 promote/rollback |
| [run_repl.py](scripts/run_repl.py) | 见上 |

> `pyproject.toml` 把这些脚本注册为 CLI（`run-research` / `run-eval` 等），依赖分 `serve`(vllm) / `train` / `dev` / `all` 四级 optional。

---

## 7. 关键共享数据结构（[schemas.py](src/orchestrator/schemas.py)）

跨模块传递的所有对象都集中在这一个文件，保证类型一致：

| 对象 | 关键字段 | 用途 |
|---|---|---|
| `SubTask` | task_id / task_type / description / dependencies / timeout_seconds / expected_type / search_hints | Planner 产出的原子任务 |
| `AgentResult` | task_id / status(SUCCESS\|FAILED\|TIMEOUT) / output / trajectory / confidence | Worker 执行结果 |
| `ResearchReport` | query / content / sources / confidence / num_searches / num_replan / adversarial_rounds / final_score | 最终交付物 |
| `RunConfig` | max_concurrent / global_timeout_seconds / max_replan_rounds / enable_adversarial / enable_evolution | 单次运行全局配置 |

---

## 8. 模块依赖关系

```raw
runner.py（唯一装配根，函数内延迟导入，避免循环依赖）
  ├─► models.model_router ─► models.vllm_policy ─► utils.env_config / utils.tracing
  ├─► planner.{planner,dag,budget_tracker} ─► orchestrator.schemas（SubTask）◄── 共享
  ├─► compressor.compressor ─► {extractive, summarizer, sliding_window} + models(policy)
  ├─► memory.memory_store ─► {long_term, embedder}
  ├─► adversarial.{loop,critic_agent,repairer_agent} + verdict（依赖 orchestrator.schemas.ResearchReport）
  └─► orchestrator.orchestrator ─► {schemas, agent_pool}
        agent_pool ─► agents.{base,researcher,summarizer}
        orchestrator（SYNTHESIZING 态）─► agents.summarizer

tools/         ← 被 AgentPool(agent 构造) 与 runner 注入
core.judge  ─► models.model_router（评测用 LLM-Judge）
evaluation/ ─► core.runner（run_research）+ core.judge + memory.embedder
evolution/  ─► orchestrator(as Solver) + memory.embedder + orchestrator.schemas
harness_evolution/ ─► SearchController / EvidenceVerifier + immutable Registry + strict Replay + paired Evaluation
```

**隐式接口契约**（跨模块鸭子类型，改模块前先对齐）：
- **policy**（任何 LLM 后端）必须实现 `__call__(messages) -> resp`、`set_tools(schemas)`、属性 `tools`、`was_truncated`。
- **tool** 必须实现 `name`、`get_openai_tool_schema()`、`async execute(**args)`。

---

## 9. 设计要点（为什么这么设计）

1. **自研编排，不依赖 LangGraph/AutoGen**：深度研究需要完全可控的调度（超时/重规划/降级），状态机字典映射（`_state_handlers`）+ DAG 分层并发比通用框架更透明易调。
2. **编排器 = 装配 + 调度分离**：`runner.initialize_modules` 做全部依赖注入，子模块互不直接 import，便于替换/消融单个模块（消融实验正是靠关掉某个注入实现）。
3. **记忆走"语义"而非"KV"**：写入去重、矛盾检测、按质量分淘汰，都是为了让长时程研究的跨 Agent 共享**不是简单的丢进 prompt**，而是可检索的结论库。
4. **压缩从"过滤"到"提取"到"抽象"渐进**：先用便宜的 cosine 过滤，再 TextRank 提取，最后才动用 LLM 摘要；越贵的操作触发越靠后。
5. **对抗用于"事后纠偏"**：Critic-Repairer 放在整篇报告合成之后，而不是每个子任务内，避免成本爆炸；用置信度门控 + 收敛/震荡检测控制预算。
6. **多后端路由 + mock 开关**：模块级选模型（推理/成本/稳定性分流）+ 模块级采样参数集中管理；`mock_mode` 让无 key 也能跑通全链路 demo 与评测。

---

## 10. 与 README / 注释不一致之处

> 阅读代码时的"避坑清单"。前几项是**文档与实现不符**，后几项是**疑似缺陷/接线缺口**（按重要性排序，尚未修，如需可直接当作改造清单）。

**A. 文档 vs 实现**

| # | 不一致 | 以代码为准的真相 |
|---|---|---|
| A1 | README 数据流把对抗画在合成之前 | 实际是 **先 Summarizer 合成 → 再对整篇 Critic/Repairer 对抗**（[orchestrator.py:429](src/orchestrator/orchestrator.py#L429)、[orchestrator.py:435](src/orchestrator/orchestrator.py#L435)） |
| A2 | README 称对抗"评分达标 ≥8.0 / 3 轮上限" | 那是 AdversarialLoop **类默认值**；实际注入的是 `configs/default.yaml` 的 **max_rounds=10 / score_threshold=9.5 / delta=0.2**（[runner.py:233-235](src/core/runner.py#L233-L235)）；且进不进对抗另由合成置信度 <0.8 门控 |
| A3 | `configs/agents/*.yaml`、`planner/*.yaml`、`tools/*.yaml`、`reward_shaping.yaml` | 均**无代码加载**（Prompt 硬编码在 .py 内），属参考文档；改 Prompt 需改源码 |
| A4 | research_bench 文件头注释"20 道" | 实际 **35 题 / 11 领域** |
| A5 | README 说 9 状态机 | 属实（含 REPLANNING/FAILED，正常流只有 7 个） |

**B. 疑似缺陷 / 接线缺口**

| # | 位置 | 问题 |
|---|---|---|
| B1 | [orchestrator.py](src/orchestrator/orchestrator.py) | **已修复**：全局超时时基于已有结果生成降级报告，并记录 `force_synthesize` 决策 |
| B2 | [evolution/judge.py](src/evolution/judge.py) | **已修复**：`shape_reward` 按 `composite/5-1` 从 `[0,10]` 映射到 `[-1,1]` |
| B3 | [run_evolution.py](scripts/run_evolution.py) vs [grpo_online.yaml](configs/evolution/grpo_online.yaml) | **已修复**：读取 `training:`，同时保留旧 `trainer:` alias；未接入训练器时默认 fail-fast |
| B4 | [schemas.py:48](src/orchestrator/schemas.py#L48) + [agent_pool.py](src/orchestrator/agent_pool.py) | `TaskType` 无 `synthesize`，agent_pool 中 `elif type_key=="synthesize"` 是死代码（合成实际由 Orchestrator 手动 new SummarizerAgent） |
| B5 | [planner.py](src/planner/planner.py) | **已修复**：`max_sub_questions` 注入 Planner，并拒绝重复/空 ID 和未知依赖 |
| B6 | [researcher.py](src/agents/researcher.py) | 已增加配置化 `max_tool_calls` 硬上限；Prompt 中“2 次”仍是建议值 |
| B7 | [agent_pool.py](src/orchestrator/agent_pool.py) | **已修复**：Agent 记录原始 pool type，释放时不再全部归入 search |
| B8 | [short_term.py](src/memory/short_term.py) | `ShortTermMemory` 全项目无调用方（预留） |
| B9 | [base_agent.py:13](src/agents/base_agent.py#L13) | `TYPE_CHECKING` 下 `from orchestrator.schemas import …` 缺前导点（应为相对导入），仅类型检查报错、运行无碍 |
| B10 | [runner.py:330](src/core/runner.py#L330) | 无论是否 mock 都调 `WebSearchTool.close_session()`（mock 时用的是 Mock 版） |

---

## 附：读懂本项目的推荐阅读顺序

1. [README.md](README.md)（产品视角，**避开** §A1/A2 两条表述）
2. [configs/default.yaml](configs/default.yaml)（看一遍所有可调旋钮）
3. [runner.py](src/core/runner.py) → [orchestrator.py](src/orchestrator/orchestrator.py)（装配 + 状态机 = 骨架）
4. [schemas.py](src/orchestrator/schemas.py)（所有数据长什么样）
5. [planner.py](src/planner/planner.py) → [researcher.py](src/agents/researcher.py)（任务怎么拆、怎么搜）
6. [summarizer.py](src/agents/summarizer.py) → [adversarial/](src/adversarial/)（报告怎么来、怎么被挑刺修复）
7. [memory/](src/memory/) + [compressor/](src/compressor/)（记忆与长上下文）
8. 最后再碰 [evolution/](src/evolution/)（半成品框架，默认关闭）
