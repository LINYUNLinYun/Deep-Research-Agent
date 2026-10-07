#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/core/runner.py
================================================================================
DeepResearch Agent 核心运行逻辑。

本模块包含初始化所有模块和执行完整研究流程的核心函数，
供 scripts/ 和 evaluation/ 统一调用，避免 evaluation/ 反向依赖 scripts/。

对外接口:
    - load_config(config_path) -> dict
    - initialize_modules(config) -> dict
    - run_research(query, config, modules) -> str
    - save_report(report, query, output_dir) -> str
================================================================================
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# 将项目根目录加入 sys.path，确保 src 包可导入
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
def setup_logging(log_level: str = "INFO") -> None:
    """配置全局日志格式与级别。"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------
def load_config(config_path: str | None = None) -> dict:
    """
    加载 YAML 配置文件。

    若未指定路径，默认加载 configs/default.yaml。
    """
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


# ---------------------------------------------------------------------------
# 工具工厂
# ---------------------------------------------------------------------------
def _create_tools_factory(config: dict):
    """创建工具工厂函数，返回 Agent 可用的工具列表。"""
    tools_cfg = config.get("tools", {})
    mock_mode = tools_cfg.get("web_search", {}).get("mock_mode", True)

    from src.tools import (
        WebSearchTool,
        MockWebSearchTool,
        ArxivReaderTool,
        BrowserTool,
        MockBrowserTool,
        FileReaderTool,
        CodeSandboxTool,
        CalculatorTool,
        NotepadTool,
        SearchController,
    )

    tools = {}

    # 1. web_search
    if mock_mode:
        tools["web_search"] = MockWebSearchTool()
    else:
        tools["web_search"] = WebSearchTool()

    # 2. browser
    if mock_mode:
        tools["browser"] = MockBrowserTool()
    else:
        tools["browser"] = BrowserTool()

    # 3. arxiv_reader
    tools["arxiv_reader"] = ArxivReaderTool(use_mock=mock_mode)

    # 4. file_reader（不限制目录）
    tools["file_reader"] = FileReaderTool(allowed_base_dir=None)

    # 5. code_sandbox
    tools["code_sandbox"] = CodeSandboxTool(use_mock=mock_mode)

    # 6. calculator
    tools["calculator"] = CalculatorTool()

    # 7. notepad
    tools["notepad"] = NotepadTool()

    search_cfg = tools_cfg.get("search_control", {})
    search_policy = _load_search_control_policy(config)
    controller = SearchController(
        enabled=search_cfg.get("enabled", True),
        cache_ttl_seconds=search_cfg.get("cache_ttl_seconds", 3600),
        max_cache_entries=search_cfg.get("max_cache_entries", 512),
        query_similarity_threshold=search_cfg.get("query_similarity_threshold", 0.82),
        novelty_threshold=search_cfg.get("novelty_threshold", 0.15),
        max_rewrites=search_cfg.get("max_rewrites", 1),
        max_backend_calls=search_cfg.get("max_backend_calls_per_run", 64),
        policy=search_policy,
    )
    if "web_search" in tools:
        tools["web_search"].search_controller = controller

    # 返回列表形式（AgentPool 和 Agent 构造函数需要 list）
    return list(tools.values())


def _load_search_control_policy(config: dict):
    """Resolve the immutable search policy selected by a registry.

    Harness evolution is opt-in.  Missing configuration therefore returns
    ``None`` and preserves the legacy SearchController behavior.  Candidate
    registries are never selected implicitly.
    """

    harness_cfg = config.get("harness_evolution")
    if not isinstance(harness_cfg, dict) or not harness_cfg.get("enabled", False):
        return None

    from src.harness_evolution.policy import SearchControlPolicy
    from src.harness_evolution.registry import VersionRegistry

    registry_dir_value = harness_cfg.get(
        "registry_dir", "configs/harness_evolution/registries"
    )
    registry_dir = Path(str(registry_dir_value))
    if not registry_dir.is_absolute():
        registry_dir = PROJECT_ROOT / registry_dir

    # ``registry`` is a logical name (production/candidate/canary) by
    # default, but an explicit YAML path is also accepted for experiment
    # manifests. Relative paths remain rooted at registry_dir as required by
    # VersionRegistry.
    registry_name = str(harness_cfg.get("registry", "production"))
    manager = VersionRegistry(registry_dir)
    reference = manager.resolve(registry_name, "search_control_policy")
    policy = SearchControlPolicy.from_file(reference.path, sha256=reference.sha256)
    if policy.version != reference.version or policy.artifact_id != reference.artifact_id:
        raise ValueError("resolved search policy identity does not match its registry entry")
    return policy


def _load_numeric_verification_skill(config: dict):
    """Resolve the immutable numeric Skill selected by the same registry."""
    harness_cfg = config.get("harness_evolution")
    if not isinstance(harness_cfg, dict) or not harness_cfg.get("enabled", False):
        return None
    from src.harness_evolution.numeric_skill import NumericVerificationSkill
    from src.harness_evolution.registry import VersionRegistry

    registry_dir = Path(str(harness_cfg.get("registry_dir", "configs/harness_evolution/registries")))
    if not registry_dir.is_absolute():
        registry_dir = PROJECT_ROOT / registry_dir
    manager = VersionRegistry(registry_dir)
    reference = manager.resolve(str(harness_cfg.get("registry", "production")), "verify_numeric_claim_skill")
    with Path(reference.path).open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle) or {}
    return NumericVerificationSkill(
        spec,
        artifact={"id": reference.artifact_id, "version": reference.version, "sha256": reference.sha256},
    )


# ---------------------------------------------------------------------------
# 模块初始化
# ---------------------------------------------------------------------------
def initialize_modules(config: dict, session_id: str = "") -> dict[str, Any]:
    """
    根据配置初始化所有核心模块。

    Args:
        config: 全局配置字典。
        session_id: 会话 ID，用于 memory store 的 session 隔离。

    返回一个包含各模块实例的字典。
    """
    logger = logging.getLogger("runner")
    logger.info("正在初始化核心模块...")

    modules: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 多后端 LLM 初始化（从 .env + configs/default.yaml 读取配置）
    # ------------------------------------------------------------------
    from src.models.model_router import ModelRouter, AdaptiveModelPolicy

    model_cfg = config.get("model", {})
    default_backend = model_cfg.get("backend", "vllm")
    backend_mapping = model_cfg.get("backend_mapping", {})
    backend_sampling = model_cfg.get("backend_sampling", {})
    # Canonical module-level configuration lives in
    # model.backend_sampling.modules.

    # 辅助函数：根据模块名获取采样参数覆盖
    def _get_sampling_kwargs(module_name: str, backend_name: str) -> dict:
        """合并后端全局默认 + 模块级覆盖参数。"""
        kwargs = {}
        # 1. 后端全局默认
        if backend_name in backend_sampling:
            kwargs.update(backend_sampling[backend_name])
        # 2. 模块级覆盖（优先级更高）
        module_overrides = backend_sampling.get("modules", {}).get(module_name, {})
        kwargs.update(module_overrides)
        return kwargs

    # 默认后端（所有模块共用）
    default_kwargs = _get_sampling_kwargs("default", default_backend)
    try:
        default_policy = ModelRouter.create_backend(default_backend, module_name="default", **default_kwargs)
    except ValueError as exc:
        logger.warning("[LLM] 默认后端不可用 (%s)，回退到本地 vLLM", exc)
        default_backend = "vllm"
        default_policy = ModelRouter.create_backend("vllm", module_name="default", **_get_sampling_kwargs("default", "vllm"))
    modules["default_policy"] = default_policy
    logger.info(f"[LLM] 默认后端已加载: {default_backend} ({default_kwargs})")

    # 多后端分工：不同模块用不同后端 + 不同采样参数
    for module_name, backend_name in backend_mapping.items():
        kwargs = _get_sampling_kwargs(module_name, backend_name)
        try:
            modules[f"{module_name}_policy"] = ModelRouter.create_backend(
                backend_name, module_name=module_name, **kwargs
            )
        except ValueError as exc:
            modules[f"{module_name}_policy"] = default_policy
            logger.warning("[LLM] %s 后端不可用，回退默认后端: %s", module_name, exc)
        logger.info(
            f"[LLM] {module_name} → backend={backend_name}, "
            f"thinking={kwargs.get('thinking', None)}, sampling={kwargs}"
        )

    # 若未配置分工，所有模块回退到 default_policy
    # ------------------------------------------------------------------

    # M2: Adaptive Planner（Orchestrator 依赖 Planner，先初始化）
    from src.planner.planner import Planner
    from src.planner.budget_tracker import BudgetTracker

    planner_policy = modules.get("planner_policy", default_policy)
    budget_tracker = BudgetTracker()
    modules["budget_tracker"] = budget_tracker
    planner = Planner(
        policy=planner_policy,
        budget_tracker=budget_tracker,
        max_tasks=config.get("orchestrator", {}).get("max_sub_questions", 8),
        max_attempts=config.get("planner", {}).get("max_parse_attempts", 3),
        facet_planning_enabled=config.get("planner", {}).get("research_state", {}).get("enabled", False),
    )
    modules["planner"] = planner
    logger.info("[M2] Planner 模块已初始化")

    # M3: Context Compressor
    from src.compressor.compressor import ContextCompressor

    compressor_policy = modules.get("compressor_policy", default_policy)
    compressor_cfg = config.get("compressor", {})
    compressor = None
    if compressor_cfg.get("enabled", True):
        compressor = ContextCompressor(
            llm_policy=compressor_policy,
            budget=compressor_cfg.get("max_context_length", 16000),
            output_reserve=compressor_cfg.get("output_reserve_tokens", 2048),
            l1_threshold=compressor_cfg.get("l1_threshold", 0.6),
            l2_threshold=compressor_cfg.get("l2_threshold", 0.8),
            l3_threshold=compressor_cfg.get("l3_threshold", 0.95),
            enable_multilevel=compressor_cfg.get("enable_multilevel", True),
            chars_per_token=compressor_cfg.get("chars_per_token", 3.5),
        )
    modules["compressor"] = compressor
    logger.info("[M3] Compressor 模块已初始化")

    # M4: Shared Memory Store
    from src.memory.memory_store import SharedMemoryStore

    memory_cfg = config.get("memory", {})
    memory_store = None
    resolved_session_id = session_id or f"run-{uuid.uuid4().hex}"
    if memory_cfg.get("enabled", True):
        memory_store = SharedMemoryStore(
            db_path=memory_cfg.get("db_path", "data/memory.db"),
            session_id=resolved_session_id,
            dedup_threshold=memory_cfg.get("similarity_threshold_dup", 0.92),
            conflict_low=memory_cfg.get("similarity_threshold_conflict", 0.65),
            conflict_high=memory_cfg.get("similarity_threshold_dup", 0.92),
            max_entries=memory_cfg.get("max_entries", 10000),
            evict_interval=memory_cfg.get("evict_interval", 100),
            retrieval_top_k=memory_cfg.get("retrieval_top_k", 10),
            retrieval_min_sim=memory_cfg.get("retrieval_min_sim", 0.55),
            recency_half_life_days=memory_cfg.get("recency_half_life_days", 30),
        )
    modules["memory_store"] = memory_store
    logger.info(f"[M4] Memory Store 模块已初始化 (session={resolved_session_id}, enabled={memory_store is not None})")

    # Tools（真实工具或 Mock 工具）
    tools_list = _create_tools_factory(config)
    modules["tools"] = tools_list
    logger.info(f"Tools 模块已初始化（共 {len(tools_list)} 个工具）")

    # M5: Critic-Repairer Adversarial Loop（先创建，再注入 Orchestrator）
    from src.adversarial.loop import AdversarialLoop
    from src.adversarial.critic_agent import CriticAgent
    from src.adversarial.repairer_agent import RepairerAgent
    from src.evidence import EvidenceVerifier

    critic_policy = modules.get("critic_agent_policy", default_policy)
    repairer_policy = modules.get("repairer_agent_policy", default_policy)
    adversarial_cfg = config.get("adversarial", {})

    max_adversarial_issues = adversarial_cfg.get("max_issues_per_round", 5)
    shared_search_controller = next(
        (
            getattr(tool, "search_controller", None)
            for tool in tools_list
            if getattr(tool, "name", "") == "web_search"
        ),
        None,
    )
    critic_agent = CriticAgent(
        policy=critic_policy,
        max_tokens=adversarial_cfg.get("critic_max_tokens", 4096),
        max_issues=max_adversarial_issues,
        max_issues_per_dimension=adversarial_cfg.get("max_issues_per_dimension", 2),
        max_sources=adversarial_cfg.get("max_sources", 20),
        context_chars=adversarial_cfg.get("critic_context_chars", 8000),
        dimension_parse_retries=adversarial_cfg.get("critic_dimension_parse_retries", 1),
    )
    repairer_agent = RepairerAgent(
        policy=repairer_policy,
        tools=tools_list,
        max_tokens=adversarial_cfg.get("repairer_max_tokens", 4096),
        max_issues=max_adversarial_issues,
        max_sources=adversarial_cfg.get("max_sources", 20),
        max_consecutive_failures=adversarial_cfg.get("max_consecutive_failures", 2),
        max_candidate_sources=adversarial_cfg.get("max_candidate_sources", 3),
        repair_context_chars=adversarial_cfg.get("repairer_context_chars", 4000),
        self_verify_context_chars=adversarial_cfg.get("self_verify_context_chars", 6000),
        search_controller=shared_search_controller,
    )
    evidence_verifier = None
    numeric_skill = _load_numeric_verification_skill(config)
    modules["numeric_verification_skill"] = numeric_skill
    if adversarial_cfg.get("evidence_verification_enabled", True):
        evidence_verifier = EvidenceVerifier(
            policy=numeric_skill,
            max_claims=adversarial_cfg.get("max_verified_claims", 50),
            min_overlap=adversarial_cfg.get("evidence_min_overlap", 0.32),
        )
    modules["evidence_verifier"] = evidence_verifier
    adversarial_loop = AdversarialLoop(
        critic_agent=critic_agent,
        repairer_agent=repairer_agent,
        policy=modules.get("judge_policy", default_policy),
        max_rounds=adversarial_cfg.get("max_rounds", 3),
        score_threshold=adversarial_cfg.get("score_threshold", 8.0),
        delta_threshold=adversarial_cfg.get("delta_threshold", 0.3),
        evidence_verifier=evidence_verifier,
    )
    modules["adversarial"] = adversarial_loop
    logger.info("[M5] Adversarial 模块已初始化")

    # M1: Multi-Agent Orchestrator
    from src.orchestrator.orchestrator import Orchestrator
    from src.orchestrator.agent_pool import AgentPool
    from src.tools import ToolExecutionPolicy

    routing_cfg = model_cfg.get("dynamic_routing", {})

    def _solver_policy_factory(_task_type: str = "search"):
        if not routing_cfg.get("enabled", False):
            # A per-agent instance prevents mutable tool/truncation state from
            # leaking across concurrent workers.
            backend_name = backend_mapping.get("solver", default_backend)
            kwargs = _get_sampling_kwargs("solver", backend_name)
            try:
                return ModelRouter.create_backend(
                    backend_name, module_name="solver", use_cache=False, **kwargs
                )
            except ValueError:
                return ModelRouter.create_backend(
                    default_backend, module_name="solver", use_cache=False,
                    **_get_sampling_kwargs("solver", default_backend),
                )
        candidates = routing_cfg.get("candidates", [default_backend])
        factories = []
        for candidate in candidates:
            kwargs = _get_sampling_kwargs("solver", candidate)
            factories.append((candidate, lambda name=candidate, kw=kwargs: ModelRouter.create_backend(
                name, module_name="solver", use_cache=False, **kw
            )))
        return AdaptiveModelPolicy(
            factories,
            max_retries=model_cfg.get("max_retries", 1),
            retry_delay=model_cfg.get("retry_delay", 0.0),
            failure_cooldown=routing_cfg.get("failure_cooldown_seconds", 60),
        )

    agent_pool = AgentPool(
        policy_factory=_solver_policy_factory,
        tools_factory=lambda: list(modules["tools"]),
        max_idle=3,
        agent_kwargs={
            "compressor": compressor,
            "max_turns": config.get("planner", {}).get("max_search_rounds_per_subagent", 5) + 2,
            "tool_policy": ToolExecutionPolicy(
                max_retries=config.get("orchestrator", {}).get("max_subagent_retries", 2),
                retry_delay=config.get("tools", {}).get("execution", {}).get("retry_delay", 0.5),
                max_retry_delay=config.get("tools", {}).get("execution", {}).get("max_retry_delay", 8.0),
                circuit_failure_threshold=config.get("tools", {}).get("execution", {}).get("circuit_failure_threshold", 3),
                circuit_cooldown=config.get("tools", {}).get("execution", {}).get("circuit_cooldown_seconds", 30),
                timeout_seconds=config.get("tools", {}).get("execution", {}).get("timeout_seconds"),
            ),
            "evidence_gain_threshold": config.get("planner", {}).get("evidence_gain_threshold", 0.15),
            "evidence_patience": config.get("planner", {}).get("evidence_gain_patience", 1),
            "max_tool_calls": config.get("planner", {}).get("max_tool_calls_per_subagent", 6),
        },
    )
    modules["agent_pool"] = agent_pool

    orchestrator = Orchestrator(
        planner=planner,
        agent_pool=agent_pool,
        budget_tracker=budget_tracker,
        compressor=compressor,
        adversarial_loop=adversarial_loop,
        memory_store=memory_store,
        summarizer_policy=modules.get("summarizer_policy", default_policy),
        evidence_verifier=evidence_verifier,
        max_catalog_sources=config.get("summarizer", {}).get("max_catalog_sources", 24),
        max_catalog_chars=config.get("summarizer", {}).get("max_catalog_chars", 12000),
    )
    modules["orchestrator"] = orchestrator
    logger.info("[M1] Orchestrator 模块已初始化")

    # M6: Self-Evolution Engine（预留，默认禁用）
    if config.get("evolution", {}).get("enabled", False):
        logger.info("[M6] Evolution 模块已启用（预留接口）")
    else:
        logger.info("[M6] Evolution 模块已禁用")

    return modules


# ---------------------------------------------------------------------------
# 研究流程主函数
# ---------------------------------------------------------------------------
async def run_research(query: str, config: dict, modules: dict[str, Any]) -> str:
    """
    执行完整的研究流程。

    流程：
        1. Orchestrator 调用 Planner 拆解问题为子任务 DAG
        2. Orchestrator 调度 AgentPool 中的子 Agent 并行/串行执行
        3. 子 Agent 调用 Tools 检索信息并生成子报告
        4. Compressor 管理长上下文
        5. Memory 存储中间结果
        6. Adversarial Loop 对报告进行多轮对抗优化（若启用）
        7. 输出最终研究报告

    Args:
        query: 用户输入的研究问题。
        config: 全局配置字典。
        modules: 已初始化的模块实例字典。

    Returns:
        最终研究报告文本（Markdown 格式）。
    """
    import asyncio

    logger = logging.getLogger("runner")
    logger.info(f"开始研究，查询: {query[:80]}...")

    start_time = time.time()

    # Search deduplication is shared across workers in one run, not across
    # unrelated queries/evaluation examples.
    seen_controllers: set[int] = set()
    for tool in modules.get("tools", []):
        controller = getattr(tool, "search_controller", None)
        if controller is not None and id(controller) not in seen_controllers:
            controller.reset()
            seen_controllers.add(id(controller))

    # Step 1-3: Orchestrator 内部完成规划、调度、收集、合成
    orchestrator = modules["orchestrator"]
    from src.orchestrator.schemas import RunConfig

    run_cfg = RunConfig(
        max_concurrent=config.get("orchestrator", {}).get("max_concurrent", 5),
        global_timeout_seconds=config.get("orchestrator", {}).get("global_timeout_seconds", 600),
        max_replan_rounds=config.get("orchestrator", {}).get("max_replan_rounds", 3),
        max_sub_questions=config.get("orchestrator", {}).get("max_sub_questions", 8),
        enable_adversarial=config.get("adversarial", {}).get("enabled", True),
        enable_evolution=config.get("evolution", {}).get("enabled", False),
        enable_replan=config.get("planner", {}).get("enable_replan", True),
        replan_failure_ratio=config.get("planner", {}).get("replan_failure_ratio", 0.35),
        replan_min_novelty=config.get("planner", {}).get("replan_min_novelty", 0.08),
        replan_novelty_patience=config.get("planner", {}).get("replan_novelty_patience", 1),
        min_usable_confidence=config.get("planner", {}).get("min_usable_confidence", 0.45),
        adversarial_confidence_threshold=config.get("adversarial", {}).get("confidence_gate", 0.8),
        adversarial_timeout_seconds=config.get("adversarial", {}).get("timeout_seconds", 180),
        token_budget=config.get("orchestrator", {}).get("token_budget", 100000),
        evidence_replan_threshold=config.get("planner", {}).get("evidence_replan_threshold", 0.6),
        evidence_replan_max_tasks=config.get("planner", {}).get("evidence_replan_max_tasks", 3),
        research_state_enabled=config.get("planner", {}).get("research_state", {}).get("enabled", False),
        research_state_active=config.get("planner", {}).get("research_state", {}).get("active", False),
        frontier_marginal_gain_threshold=config.get("planner", {}).get("research_state", {}).get("marginal_gain_threshold", 0.15),
        frontier_max_consecutive_action=config.get("planner", {}).get("research_state", {}).get("max_consecutive_action", 2),
    )

    report = await orchestrator.run(query, config=run_cfg)
    modules["last_report"] = report
    logger.info(
        f"[Orchestrator] 报告生成完成 | 置信度={report.confidence:.2f} | "
        f"搜索轮数={report.num_searches} | 重规划={report.num_replan} | 对抗轮数={report.adversarial_rounds}"
    )

    # Step 4/5: 进化优化（如启用且已训练）
    if run_cfg.enable_evolution:
        logger.info("[Evolution] 进化优化已启用（预留接口）")
    else:
        logger.info("[Evolution] 进化优化已跳过")

    # 关闭 WebSearchTool 连接池
    from src.tools.web_search import WebSearchTool
    await WebSearchTool.close_session()

    elapsed = time.time() - start_time
    logger.info(f"研究完成，耗时: {elapsed:.2f} 秒")

    # 组装最终输出
    final_report = _format_report(report, elapsed)
    return final_report


def collect_harness_telemetry(modules: dict[str, Any]) -> dict[str, Any]:
    """Return JSON-serialisable signals used by paired Harness ablations."""
    report = modules.get("last_report")
    search = {}
    seen: set[int] = set()
    for tool in modules.get("tools", []):
        controller = getattr(tool, "search_controller", None)
        if controller is not None and id(controller) not in seen:
            search = controller.snapshot()
            seen.add(id(controller))
    tool_policy = getattr(modules.get("agent_pool"), "agent_kwargs", {}).get("tool_policy")
    compressor = modules.get("compressor")
    budget = modules.get("budget_tracker")
    budget_snapshot = budget.snapshot() if budget is not None else None
    numeric_skill = modules.get("numeric_verification_skill")
    skill_artifact = dict(getattr(numeric_skill, "artifact", {}) or {})
    return {
        "search": search,
        "tools": tool_policy.snapshot() if tool_policy is not None else {},
        "compression": compressor.get_stats() if compressor is not None else {"enabled": False},
        "agent_pool": modules["agent_pool"].get_stats() if modules.get("agent_pool") else {},
        "decision_trace": list(getattr(report, "decision_trace", []) or []),
        "evidence_verification": dict(getattr(report, "evidence_verification", {}) or {}),
        "research_state": dict(getattr(report, "research_state", {}) or {}),
        "skills": {
            "verify_numeric_claim_skill": {
                **skill_artifact,
                "triggered": bool(
                    numeric_skill is not None
                    and getattr(report, "evidence_verification", {}).get("total_claims", 0)
                ),
            }
        } if numeric_skill is not None else {},
        "num_searches": getattr(report, "num_searches", 0),
        "num_replan": getattr(report, "num_replan", 0),
        "adversarial_rounds": getattr(report, "adversarial_rounds", 0),
        "budget": {
            "total_tokens": getattr(budget_snapshot, "total_tokens", 0),
            "budget_limit": getattr(budget_snapshot, "budget_limit", 0),
            "usage_ratio": getattr(budget_snapshot, "usage_ratio", 0.0),
        },
    }


def _format_report(report, elapsed: float) -> str:
    """将 ResearchReport 格式化为 Markdown 文本。"""
    content = report.content or ""
    verification = getattr(report, "evidence_verification", {}) or {}
    verification_summary = {
        key: verification.get(key)
        for key in (
            "total_claims", "supported", "contradicted", "unknown",
            "support_rate", "unsupported_rate",
        )
        if key in verification
    }

    # 统一置信度：如果正文中有 LLM 自评的"整体置信度"，替换为实际计算值，避免不一致
    content = re.sub(
        r"(整体置信度|Overall Confidence|置信度)[:：]\s*0?\.\d+",
        f"\\1: {report.confidence:.2f}",
        content,
        flags=re.I,
    )

    lines = [
        f"# 研究报告：{report.query}",
        "",
        "---",
        "",
        content,
        "",
        "---",
        "",
        "## 元信息",
        "",
        f"- **置信度**: {report.confidence:.2f}",
        f"- **搜索轮数**: {report.num_searches}",
        f"- **重规划次数**: {report.num_replan}",
        f"- **对抗轮数**: {report.adversarial_rounds}",
        f"- **对抗状态**: {getattr(report, 'adversarial_status', 'not_run')}",
        f"- **对抗原因**: {getattr(report, 'adversarial_reason', '') or '无'}",
        f"- **Harness 决策数**: {len(getattr(report, 'decision_trace', []) or [])}",
        f"- **证据验证**: {verification_summary or '未启用'}",
        f"- **总耗时**: {elapsed:.2f} 秒",
        "",
    ]

    history = getattr(report, "adversarial_history", [])
    if history:
        lines.extend(["## 对抗审查记录", "", "| 轮次 | 维度 | 修复前 | 修复后 | 采纳 |", "|---|---|---:|---:|---|"])
        for record in history:
            before = record.get("pre_fix_dimension_scores", record.get("dimension_scores", {}))
            after = record.get("dimension_scores", {})
            for dimension in dict.fromkeys([*before, *after]):
                lines.append(f"| {record['round']} | {dimension} | {before.get(dimension, '—')} | {after.get(dimension, '—')} | {record.get('accepted', False)} |")
        lines.append("")

    if report.sources:
        lines.append("## 参考来源")
        lines.append("")
        for i, src in enumerate(report.sources, 1):
            citation_id = src.get("citation_id", i)
            title = src.get("title", "未知标题")
            url = src.get("url", "")
            snippet = str(src.get("snippet", ""))[:240]
            relevance = src.get("temporal_relevance")
            date_note = f" | source_date={src.get('source_date')} | temporal_relevance={relevance}" if relevance else ""
            lines.append(f"[{citation_id}] [{title}]({url}) — {snippet}{date_note}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 报告保存
# ---------------------------------------------------------------------------
def save_report(report: str, query: str, output_dir: str = "outputs/reports") -> str:
    """
    将研究报告保存到文件。

    文件名格式：report_YYYYMMDD_HHMMSS_<query前20字>.md
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c if c.isalnum() or c in "_-" else "_" for c in query[:20])
    filename = f"report_{timestamp}_{safe_query}.md"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    return filepath
