"""Fresh-module paired replay runner for the search-policy track."""
from __future__ import annotations

import copy
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from evaluation.benchmarks.research_bench import ResearchBench
from evaluation.metrics.rule_based import RuleBasedMetrics
from src.core.runner import collect_harness_telemetry, initialize_modules, load_config, run_research
from src.harness_evolution.evaluation import DatasetManifest, PairedEvolutionEvaluator
from src.harness_evolution.manifest import RunManifest
from src.harness_evolution.registry import VersionRegistry
from src.harness_evolution.replay import RecordReplayStore, ReplayMismatchError, attach_replay_tools
from src.harness_evolution.numeric_skill import NumericVerificationSkill
from src.evidence.schemas import Claim, Evidence
from src.models.model_router import ModelRouter


def _policy_behavior_signature(detail: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    telemetry = detail.get("telemetry", {})
    if not isinstance(telemetry, Mapping):
        return ()
    search = telemetry.get("search", {})
    if not isinstance(search, Mapping):
        return ()
    decisions = search.get("policy_decisions", [])
    if not isinstance(decisions, list):
        return ()
    return tuple(
        (str(item.get("rule_id", "")), str(item.get("action", "")))
        for item in decisions
        if isinstance(item, Mapping)
    )


def _mark_policy_mechanism_changes(details: list[dict[str, Any]]) -> None:
    """Mark only candidate runs whose observable policy behavior changed."""

    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for detail in details:
        by_pair.setdefault(str(detail.get("pair_id", "")), {})[str(detail.get("group", ""))] = detail
        detail["mechanism_triggered"] = False
    for groups in by_pair.values():
        baseline = groups.get("baseline")
        candidate = groups.get("candidate")
        if baseline is None or candidate is None:
            continue
        candidate["mechanism_triggered"] = (
            _policy_behavior_signature(candidate) != _policy_behavior_signature(baseline)
        )


def _frozen_eval_config(config: Mapping[str, Any], registry_dir: Path, registry: str) -> dict[str, Any]:
    cfg = copy.deepcopy(dict(config))
    model = cfg.setdefault("model", {})
    model["temperature"] = 0
    model["top_p"] = 1.0
    model.setdefault("dynamic_routing", {})["enabled"] = False
    sampling = model.setdefault("backend_sampling", {})
    for name, value in sampling.items():
        if name == "modules" or not isinstance(value, dict):
            continue
        value["temperature"] = 0
        value["top_p"] = 1.0
    modules_sampling = sampling.get("modules", {})
    if isinstance(modules_sampling, dict):
        for value in modules_sampling.values():
            if isinstance(value, dict):
                value["temperature"] = 0
                value["top_p"] = 1.0
    cfg["harness_evolution"] = {
        "enabled": True,
        "registry_dir": str(registry_dir),
        "registry": registry,
    }
    cfg.setdefault("tools", {}).setdefault("web_search", {})["mock_mode"] = True
    return cfg


async def _ensure_policy_claim_verification(modules: Mapping[str, Any]) -> Mapping[str, Any]:
    """Run deterministic claim verification when the main path did not do so."""

    report = modules.get("last_report")
    current = getattr(report, "evidence_verification", {}) or {}
    if isinstance(current, Mapping):
        try:
            if int(current.get("total_claims", 0) or 0) > 0:
                return current
        except (TypeError, ValueError):
            pass
    verifier = modules.get("evidence_verifier")
    if report is None or verifier is None:
        return current if isinstance(current, Mapping) else {}
    try:
        results = await verifier.verify(report)
        summary = verifier.summary(results)
    except Exception as exc:
        summary = {"error": f"{type(exc).__name__}: {exc}"}
    report.evidence_verification = dict(summary)
    return summary


async def run_policy_replay_experiment(
    *,
    config_path: str | None,
    registry_dir: str | Path,
    baseline_registry: str,
    candidate_registry: str,
    dataset_manifest: str | Path,
    fixture_dir: str | Path,
    output_dir: str | Path,
    split: str = "held_out",
    repeats: int = 2,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    registry_root = Path(registry_dir).resolve()
    dataset = DatasetManifest.load(dataset_manifest)
    ids = list(dataset.data["splits"].get(split, []))
    questions_by_id = {item["id"]: item for item in ResearchBench.DEFAULT_QUESTIONS}
    missing = [qid for qid in ids if qid not in questions_by_id]
    if missing:
        raise ValueError(f"dataset manifest refers to unknown queries: {missing}")
    questions = [questions_by_id[qid] for qid in ids]
    fixture_root = Path(fixture_dir).resolve()
    output_root = Path(output_dir).resolve()
    exp_id = experiment_id or f"policy-{uuid.uuid4().hex[:12]}"
    base_config = load_config(config_path)

    async def run_one(**kwargs: Any) -> Mapping[str, Any]:
        group = kwargs["group"]
        question = kwargs["question"]
        pair_id = kwargs["pair_id"]
        registry_name = kwargs["registry"]
        fixture_path = fixture_root / f"{question['id']}.json"
        if not fixture_path.is_file():
            raise FileNotFoundError(f"missing strict replay fixture: {fixture_path}")
        store = RecordReplayStore.load(fixture_path)
        cfg = _frozen_eval_config(base_config, registry_root, registry_name)
        run_id = f"{exp_id}-{group}-{question['id']}-r{kwargs['repeat'] + 1}"
        session_id = f"isolated:{run_id}"
        ModelRouter.clear_cache()
        modules = initialize_modules(cfg, session_id=session_id)
        attach_replay_tools(modules, store)
        manager = VersionRegistry(registry_root)
        artifacts = {
            artifact_id: manager.resolve(registry_name, artifact_id)
            for artifact_id in ("search_control_policy", "verify_numeric_claim_skill")
        }
        manifest = RunManifest.create(
            run_id=run_id,
            experiment_id=exp_id,
            pair_id=pair_id,
            query_id=question["id"],
            group=group,
            registry=registry_root / f"{registry_name}.yaml",
            artifacts=artifacts,
            config=cfg,
            executor={
                "backend": cfg.get("model", {}).get("backend", ""),
                "temperature": 0,
                "top_p": 1.0,
                "seed_controlled": False,
            },
            budgets={
                "timeout": cfg.get("orchestrator", {}).get("global_timeout_seconds"),
                "tool_calls": cfg.get("planner", {}).get("max_tool_calls_per_subagent"),
                "max_tokens": cfg.get("model", {}).get("max_tokens"),
            },
            memory_session=session_id,
            dataset={"id": dataset.data.get("dataset_id"), "version": dataset.data.get("version"), "split": split, "sha256": dataset.sha256},
            tool={"mode": "replay", "fixture_path": str(fixture_path), "fixture_sha256": store.sha256()},
            evaluator={"id": "rule_based_v1", "weights": "RuleBasedMetrics.composite_score"},
            created_at=datetime.now(timezone.utc).isoformat(),
            repo_root=project_root,
        )
        manifest_path = output_root / exp_id / "manifests" / f"{run_id}.json"
        manifest.write(manifest_path)
        started = time.perf_counter()
        try:
            report = await run_research(question["query"], cfg, modules)
            elapsed = time.perf_counter() - started
            metrics = {
                "factual_accuracy": RuleBasedMetrics.fact_accuracy(report, question.get("ground_truth", {})),
                "logical_consistency": RuleBasedMetrics.logical_consistency(report),
                "citation_coverage": RuleBasedMetrics.citation_coverage(report),
                "bias": max(0.0, 1.0 - RuleBasedMetrics.hallucination_rate(report)),
                "comprehensiveness": RuleBasedMetrics.comprehensiveness(report, question.get("expected_topics", [])),
            }
            metrics["composite_score"] = RuleBasedMetrics.composite_score(metrics)
            verification = await _ensure_policy_claim_verification(modules)
            telemetry = collect_harness_telemetry(modules)
            replay_mismatch = any(
                "ReplayMismatchError" in str(attempt.get("error", ""))
                for call in telemetry.get("tools", {}).get("recent", [])
                for attempt in call.get("attempts", [])
            )
            verification = telemetry.get("evidence_verification", verification)
            stats = telemetry.get("search", {}).get("stats", {})
            metrics.update({
                "unsupported_claim_rate": float(verification.get("unsupported_rate", 0.0) or 0.0),
                "critical_factual_errors": int(verification.get("contradicted", 0) or 0),
                "total_tokens": int(telemetry.get("budget", {}).get("total_tokens", 0) or 0),
                "latency_seconds": elapsed,
                "evidence_yield": float(stats.get("new_results", 0) or 0) / max(int(stats.get("backend_calls", 0) or 0), 1),
                "claim_macro_f1": 0.0,
            })
            false_supported_rate = RuleBasedMetrics.false_supported_rate(
                dict(verification) if isinstance(verification, Mapping) else None
            )
            if false_supported_rate is not None:
                metrics["false_supported_rate"] = false_supported_rate
            return {
                "query_type": str(question.get("domain", "unknown")),
                "metrics": metrics,
                "mechanism_triggered": bool(telemetry.get("search", {}).get("policy_decisions")),
                "telemetry": telemetry,
                "manifest": str(manifest_path),
                "replay_mismatch": replay_mismatch,
                "isolation_error": "strict replay request was not covered" if replay_mismatch else "",
            }
        except ReplayMismatchError as exc:
            return {
                "query_type": str(question.get("domain", "unknown")),
                "metrics": {"composite_score": 0.0},
                "replay_mismatch": True,
                "isolation_error": str(exc),
                "manifest": str(manifest_path),
            }

    evaluator = PairedEvolutionEvaluator(run_one, repeats=repeats)
    result = await evaluator.evaluate(
        questions,
        baseline_registry=baseline_registry,
        candidate_registry=candidate_registry,
        fixture=str(fixture_root),
        split=split,
    )
    _mark_policy_mechanism_changes(result["details"])
    result.update({
        "experiment_id": exp_id,
        "dataset": {
            "id": dataset.data.get("dataset_id", "researchbench_v1"),
            "version": dataset.data.get("version", ""),
            "split": split,
            "sha256": dataset.sha256,
        },
        "dataset_sha256": dataset.sha256,
    })
    return result


def run_numeric_skill_experiment(
    *,
    registry_dir: str | Path,
    baseline_registry: str,
    candidate_registry: str,
    dataset_path: str | Path,
    output_dir: str | Path,
    split: str = "held_out",
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate immutable numeric skills without an LLM or network access."""
    import yaml

    registry_root = Path(registry_dir).resolve()
    dataset_file = Path(dataset_path).resolve()
    rows = [json.loads(line) for line in dataset_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("split") == split]
    if not rows:
        raise ValueError(f"numeric dataset contains no {split!r} rows")
    exp_id = experiment_id or f"skill-{uuid.uuid4().hex[:12]}"
    output_root = Path(output_dir).resolve()
    dataset_sha256 = __import__("hashlib").sha256(dataset_file.read_bytes()).hexdigest()
    manager = VersionRegistry(registry_root)
    predictions: dict[str, dict[str, str]] = {"baseline": {}, "candidate": {}}
    normalization: dict[str, dict[str, bool]] = {"baseline": {}, "candidate": {}}
    raw_details: list[dict[str, Any]] = []
    refs: dict[str, Any] = {}

    for group, registry_name in (("baseline", baseline_registry), ("candidate", candidate_registry)):
        ref = manager.resolve(registry_name, "verify_numeric_claim_skill")
        refs[group] = ref
        spec = yaml.safe_load(Path(ref.path).read_text(encoding="utf-8"))
        skill = NumericVerificationSkill(spec, artifact={"id": ref.artifact_id, "version": ref.version, "sha256": ref.sha256})
        for row in rows:
            claim = Claim(str(row["id"]), str(row["claim"]), metadata={"claim_type": row["claim_type"]})
            # Keep the original single ``source_span`` fixture format while
            # accepting challenge fixtures with multiple attributed sources.
            source_rows = row.get("sources")
            if isinstance(source_rows, list):
                evidence = []
                for source in source_rows:
                    if not isinstance(source, Mapping):
                        continue
                    metadata = dict(source.get("metadata", {}) or {})
                    for key in ("source_tier", "source_id"):
                        if key in source:
                            metadata.setdefault(key, source[key])
                    evidence.append(
                        Evidence(
                            source_url=str(source.get("source_url", source.get("url", "")) or ""),
                            source_span=str(
                                source.get("source_span", source.get("snippet", source.get("content", source.get("text", ""))))
                                or ""
                            ),
                            source_date=str(source.get("source_date", source.get("date", "")) or ""),
                            domain=str(source.get("domain", "") or ""),
                            title=str(source.get("title", "") or ""),
                            metadata=metadata,
                        )
                    )
            else:
                evidence = [Evidence(source_span=str(row.get("source_span", "")))]
            skill_result = skill(claim, evidence)
            predicted = str(skill_result["status"])
            expected = str(row["label"])
            predictions[group][str(row["id"])] = predicted
            expected_normalized = row.get("expected_normalized", {})
            if isinstance(expected_normalized, list):
                normalization[group][str(row["id"])] = bool(expected_normalized) and all(
                    isinstance(expected_item, Mapping)
                    and _matches_expected_normalized(skill.extract_facts(claim.text), expected_item)
                    for expected_item in expected_normalized
                )
            else:
                normalization[group][str(row["id"])] = _matches_expected_normalized(
                    skill.extract_facts(claim.text), expected_normalized
                )
            run_id = f"{exp_id}-{group}-{row['id']}"
            manifest = RunManifest.create(
                run_id=run_id,
                experiment_id=exp_id,
                pair_id=str(row["id"]),
                query_id=str(row["id"]),
                group=group,
                registry=registry_root / f"{registry_name}.yaml",
                artifacts={ref.artifact_id: ref},
                config={"track": "skill", "network": False},
                executor={"type": "deterministic_numeric_skill", "seed_controlled": True},
                budgets={"tool_calls": 0, "max_tokens": 0},
                memory_session=f"isolated:{run_id}",
                dataset={"id": dataset_file.stem, "split": split, "sha256": dataset_sha256},
                tool={"mode": "disabled"},
                evaluator={"id": "numeric_exact_label_v1"},
                created_at=datetime.now(timezone.utc).isoformat(),
                repo_root=Path(__file__).resolve().parents[2],
            )
            manifest_path = output_root / exp_id / "manifests" / f"{run_id}.json"
            manifest.write(manifest_path)
            raw_details.append({
                "pair_id": str(row["id"]),
                "query_id": str(row["id"]),
                "group": group,
                "query_type": str(row.get("failure_type", row["claim_type"])),
                "claim_type": str(row["claim_type"]),
                "dataset_split": split,
                "expected": expected,
                "predicted": predicted,
                "reason": str(skill_result.get("reason", "")),
                "primitive_trace": list(skill_result.get("primitive_trace", [])),
                "normalization_correct": normalization[group][str(row["id"])],
                "manifest": str(manifest_path),
            })

    macro = {group: _macro_f1(rows, values) for group, values in predictions.items()}
    normalization_accuracy = {
        group: sum(values.values()) / max(len(values), 1)
        for group, values in normalization.items()
    }
    for detail in raw_details:
        correct = detail["predicted"] == detail["expected"]
        false_supported = detail["predicted"] == "supported" and detail["expected"] != "supported"
        unsupported = detail["expected"] == "supported" and detail["predicted"] != "supported"
        # For the deterministic Skill track, a candidate is considered to have
        # exercised its changed mechanism only when the immutable artifact
        # changed *and* its final label differs from the paired baseline.  A
        # same-version audit therefore cannot accidentally pass this gate just
        # because the parser emitted a normal ``numeric_skill:...`` reason.
        detail["mechanism_triggered"] = bool(
            detail["group"] == "candidate"
            and refs["candidate"].sha256 != refs["baseline"].sha256
            and detail["predicted"] != predictions["baseline"].get(detail["query_id"])
        )
        detail["metrics"] = {
            "composite_score": 1.0 if correct else 0.0,
            "claim_macro_f1": macro[detail["group"]],
            "numeric_normalization_accuracy": normalization_accuracy[detail["group"]],
            "false_supported_rate": 1.0 if false_supported else 0.0,
            "unsupported_claim_rate": 1.0 if unsupported else 0.0,
            "critical_factual_errors": 1 if false_supported else 0,
            "total_tokens": 0,
            "latency_seconds": 1.0,
            "evidence_yield": 0.0,
        }
    return {
        "experiment_id": exp_id,
        "split": split,
        "dataset": {"id": dataset_file.stem, "split": split, "sha256": dataset_sha256},
        "dataset_sha256": dataset_sha256,
        "details": raw_details,
        "claim_macro_f1": macro,
        "numeric_normalization_accuracy": normalization_accuracy,
        "artifacts": {
            group: {"artifact_id": ref.artifact_id, "version": ref.version, "sha256": ref.sha256, "parent": ref.parent}
            for group, ref in refs.items()
        },
    }


def _matches_expected_normalized(facts: list[Any], expected: Mapping[str, Any]) -> bool:
    """Check fixture normalization independently from final verdict labels."""
    if isinstance(expected, Mapping) and "facts" in expected:
        expected_facts = expected.get("facts")
        return isinstance(expected_facts, list) and bool(expected_facts) and all(
            isinstance(expected_item, Mapping)
            and _matches_expected_normalized(facts, expected_item)
            for expected_item in expected_facts
        )
    if not expected or "value" not in expected:
        return False
    try:
        expected_value = float(expected["value"])
    except (TypeError, ValueError):
        # The V1 parser currently normalises years but not complete ISO dates.
        # Unsupported structured values are a failed normalisation check, not
        # a reason to abort an otherwise valid held-out evaluation.
        return False
    expected_unit = str(expected.get("unit", ""))
    unit_aliases = {
        "count": ("number", ""),
        "percent": ("percentage", "%"),
        "year": ("date", "year"),
        "fiscal_year": ("fiscal_year", "fiscal_year"),
    }
    expected_kind, canonical_unit = unit_aliases.get(expected_unit, ("", expected_unit))
    for fact in facts:
        if expected_kind and str(fact.kind) != expected_kind:
            continue
        if canonical_unit and str(fact.unit) != canonical_unit:
            continue
        tolerance = max(1e-9, abs(expected_value) * 1e-12)
        if abs(float(fact.value) - expected_value) <= tolerance:
            return True
    return False


def _macro_f1(rows: list[Mapping[str, Any]], predictions: Mapping[str, str]) -> float:
    scores = []
    for label in ("supported", "contradicted", "unknown"):
        tp = fp = fn = 0
        for row in rows:
            expected = str(row["label"])
            predicted = predictions.get(str(row["id"]), "unknown")
            tp += int(expected == label and predicted == label)
            fp += int(expected != label and predicted == label)
            fn += int(expected == label and predicted != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores.append(2 * precision * recall / max(precision + recall, 1e-12))
    return sum(scores) / len(scores)
