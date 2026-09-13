"""Paired evaluation and multi-objective promotion gates."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from evaluation.metrics.stats import bootstrap_ci_paired


def canonical_split(value: Any) -> str:
    """Return the canonical spelling used by the evolution data protocol."""

    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {"heldout", "held_out", "test"}:
        return "held_out"
    return normalized


def is_held_out_split(value: Any) -> bool:
    """Whether ``value`` denotes a protected evaluation split.

    The CLI accepts ``held_out`` explicitly, but accepting common aliases here
    prevents a caller from bypassing the isolation check with ``heldout`` or
    ``held-out`` in an input JSONL file.
    """

    return canonical_split(value) == "held_out"


def _aggregate_details(details: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Produce a public, row-identity-free summary of paired run details.

    This function intentionally does not retain query IDs, claim text, source
    spans, expected/predicted labels, reasons, telemetry, or report content.
    It is safe to persist for a protected held-out evaluation while the full
    details remain available to :class:`PromotionGate` in the current process.
    """

    groups: dict[str, list[Mapping[str, Any]]] = {}
    pair_groups: dict[str, set[str]] = {}
    for item in details:
        group = str(item.get("group", "unknown"))
        groups.setdefault(group, []).append(item)
        pair_id = str(item.get("pair_id", ""))
        if pair_id:
            pair_groups.setdefault(pair_id, set()).add(group)

    aggregate: dict[str, Any] = {
        "detail_count": len(details),
        "pair_count": sum(1 for values in pair_groups.values() if {"baseline", "candidate"} <= values),
        "groups": {},
    }
    for group, items in sorted(groups.items()):
        metric_values: dict[str, list[float]] = {}
        mechanism_count = 0
        replay_mismatch_count = 0
        isolation_error_count = 0
        for item in items:
            if bool(item.get("mechanism_triggered")):
                mechanism_count += 1
            if bool(item.get("replay_mismatch")):
                replay_mismatch_count += 1
            if item.get("isolation_error"):
                isolation_error_count += 1
            metrics = item.get("metrics", {})
            if not isinstance(metrics, Mapping):
                continue
            for name, value in metrics.items():
                if str(name).lower() in {
                    "claim", "content", "expected", "expected_normalized", "predicted",
                    "reason", "report", "source", "source_span", "text",
                }:
                    continue
                # bool is an implementation detail, not a metric value.
                if isinstance(value, bool):
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(numeric):
                    continue
                metric_values.setdefault(str(name), []).append(numeric)
        aggregate["groups"][group] = {
            "count": len(items),
            "metrics": {
                name: statistics.fmean(values)
                for name, values in sorted(metric_values.items())
                if values
            },
            "mechanism_trigger_rate": mechanism_count / max(len(items), 1),
            "replay_mismatch_count": replay_mismatch_count,
            "isolation_error_count": isolation_error_count,
        }
    return aggregate


def redact_held_out_evaluation(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Return the public representation of a held-out evaluation.

    The evaluator must call :meth:`PromotionGate.decide` using the original
    mapping *before* invoking this function.  This keeps per-row information in
    memory for statistical gates but makes it impossible for the persisted
    evaluation/decision/report artifacts to become a source of held-out labels.
    Development evaluations are deliberately unchanged by this helper.
    """

    if not is_held_out_split(evaluation.get("split", "")):
        return dict(evaluation)

    public: dict[str, Any] = {}
    # Only these scalar/identity fields are allowed through.  In particular,
    # unknown top-level fields from a custom results file are not copied.
    for key in ("experiment_id", "split", "repeats", "dataset_sha256", "evaluator_sha256"):
        if key in evaluation:
            public[key] = evaluation[key]
    for key in ("dataset", "evaluator"):
        value = evaluation.get(key)
        if isinstance(value, Mapping):
            public[key] = {
                name: value[name]
                for name in ("id", "version", "split", "sha256")
                if name in value and isinstance(value[name], (str, int, float, bool, type(None)))
            }
    artifacts = evaluation.get("artifacts")
    if isinstance(artifacts, Mapping):
        public["artifacts"] = {}
        for group, value in artifacts.items():
            if not isinstance(value, Mapping):
                continue
            public["artifacts"][str(group)] = {
                name: value[name]
                for name in ("artifact_id", "version", "sha256", "parent")
                if name in value and isinstance(value[name], (str, int, float, bool, type(None)))
            }
    public["split"] = "held_out"
    public["details_redacted"] = True
    public["aggregate"] = _aggregate_details(
        [item for item in evaluation.get("details", []) if isinstance(item, Mapping)]
    )
    dataset = public.get("dataset", {})
    evaluator = public.get("evaluator", {})
    public["traceability"] = {
        "dataset_sha256": str(
            evaluation.get("dataset_sha256") or dataset.get("sha256", "")
        ),
        "evaluator_sha256": str(
            evaluation.get("evaluator_sha256") or evaluator.get("sha256", "")
        ),
        "split": "held_out",
    }
    return public


@dataclass(frozen=True)
class PromotionCriteria:
    non_regression_rate: float = 0.60
    max_token_increase: float = 0.15
    max_p95_latency_increase: float = 0.20
    min_improved_query_types: int = 2


class DatasetManifest:
    def __init__(self, data: Mapping[str, Any]) -> None:
        self.data = dict(data)
        self._validate()

    @classmethod
    def load(cls, path: str | Path) -> "DatasetManifest":
        import yaml
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    def _validate(self) -> None:
        splits = self.data.get("splits", {})
        if not isinstance(splits, Mapping):
            raise ValueError("dataset manifest requires splits")
        seen: set[str] = set()
        for name in ("miner", "development", "held_out"):
            ids = splits.get(name, [])
            if not isinstance(ids, list) or len(ids) != len(set(ids)):
                raise ValueError(f"invalid or duplicate IDs in split {name}")
            overlap = seen & set(ids)
            if overlap:
                raise ValueError(f"dataset split leakage: {sorted(overlap)}")
            seen.update(ids)

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PairedEvolutionEvaluator:
    """Evaluate baseline and candidate in alternating order with fresh runs."""

    def __init__(self, run_one: Callable[..., Mapping[str, Any] | Awaitable[Mapping[str, Any]]], repeats: int = 2) -> None:
        self.run_one = run_one
        self.repeats = max(int(repeats), 1)

    async def evaluate(
        self,
        questions: Sequence[Mapping[str, Any]],
        *,
        baseline_registry: str,
        candidate_registry: str,
        fixture: str = "",
        split: str = "held_out",
    ) -> dict[str, Any]:
        details: list[dict[str, Any]] = []
        for q_index, question in enumerate(questions):
            qid = str(question["id"])
            for repeat in range(self.repeats):
                pair_id = f"{qid}:r{repeat + 1}"
                order = ["baseline", "candidate"] if (q_index + repeat) % 2 == 0 else ["candidate", "baseline"]
                for group in order:
                    registry = baseline_registry if group == "baseline" else candidate_registry
                    result = self.run_one(
                        group=group,
                        question=dict(question),
                        repeat=repeat,
                        registry=registry,
                        fixture=fixture,
                        split=split,
                        pair_id=pair_id,
                    )
                    if inspect.isawaitable(result):
                        result = await result
                    details.append({"pair_id": pair_id, "query_id": qid, "group": group, **dict(result)})
        return {"repeats": self.repeats, "split": split, "details": details}


class PromotionGate:
    def __init__(self, criteria: PromotionCriteria | None = None) -> None:
        self.criteria = criteria or PromotionCriteria()

    def decide(self, evaluation: Mapping[str, Any], *, track: str) -> dict[str, Any]:
        paired = self._pairs(evaluation.get("details", []))
        if not paired:
            return self._decision(False, track, {}, [{"gate": "paired_results", "passed": False, "reason": "no complete pairs"}])
        baseline = [pair["baseline"] for pair in paired]
        candidate = [pair["candidate"] for pair in paired]
        quality_diffs = [self._metric(c, "composite_score") - self._metric(b, "composite_score") for b, c in zip(baseline, candidate)]
        # Repeats of the same query are correlated observations, not additional
        # independent samples.  Collapse them before significance and W/T/L
        # calculations so increasing ``--repeats`` cannot manufacture a narrow
        # confidence interval.
        clustered_quality_diffs = self._cluster_diffs(paired, quality_diffs)
        stats = bootstrap_ci_paired(clustered_quality_diffs, seed=0, alternative="greater")
        stats["cohens_d"] = self._paired_effect_size(clustered_quality_diffs)
        non_regression = sum(diff >= 0 for diff in clustered_quality_diffs) / len(clustered_quality_diffs)
        unsupported_delta = self._mean_delta(candidate, baseline, "unsupported_claim_rate")
        critical_delta = self._mean_delta(candidate, baseline, "critical_factual_errors")
        false_supported_available = all(
            self._has_metric(item, "false_supported_rate") for item in candidate + baseline
        )
        false_supported_delta = (
            self._mean_delta(candidate, baseline, "false_supported_rate")
            if false_supported_available
            else None
        )
        token_increase = self._relative_change(candidate, baseline, "total_tokens")
        latency_increase = self._relative_p95_change(candidate, baseline, "latency_seconds")
        improved_types = self._improved_types(paired)
        triggered = any(bool(item.get("mechanism_triggered")) for item in candidate)
        isolation_ok = all(not item.get("isolation_error") and not item.get("replay_mismatch") for item in candidate + baseline)
        gates = [
            self._gate("quality_ci", stats.get("ci_lower", 0) > 0, stats),
            self._gate("non_regression_rate", non_regression >= self.criteria.non_regression_rate, non_regression),
            self._gate("unsupported_claims", unsupported_delta <= 0, unsupported_delta),
            self._gate("critical_factual_errors", critical_delta <= 0, critical_delta),
            self._gate(
                "false_supported",
                false_supported_available and false_supported_delta is not None and false_supported_delta <= 0,
                {"available": false_supported_available, "delta": false_supported_delta},
            ),
            self._gate("token_budget", token_increase <= self.criteria.max_token_increase, token_increase),
            self._gate("p95_latency", latency_increase <= self.criteria.max_p95_latency_increase, latency_increase),
            self._gate("query_type_transfer", len(improved_types) >= self.criteria.min_improved_query_types, improved_types),
            self._gate("isolation", isolation_ok, isolation_ok),
            self._gate("mechanism_triggered", triggered, triggered),
        ]
        if track == "policy":
            gates.append(self._gate("evidence_yield", self._mean_delta(candidate, baseline, "evidence_yield") >= 0,
                                    self._mean_delta(candidate, baseline, "evidence_yield")))
        elif track == "skill":
            gates.append(self._gate("claim_macro_f1", self._mean_delta(candidate, baseline, "claim_macro_f1") >= 0,
                                    self._mean_delta(candidate, baseline, "claim_macro_f1")))
        else:
            raise ValueError(f"unsupported track: {track}")
        metrics = {
            "quality": stats,
            "non_regression_rate": non_regression,
            "unsupported_claim_delta": unsupported_delta,
            "critical_error_delta": critical_delta,
            "false_supported_delta": false_supported_delta,
            "token_increase": token_increase,
            "p95_latency_increase": latency_increase,
            "improved_query_types": improved_types,
            "win_tie_loss": {
                "win": sum(value > 0 for value in clustered_quality_diffs),
                "tie": sum(value == 0 for value in clustered_quality_diffs),
                "loss": sum(value < 0 for value in clustered_quality_diffs),
            },
        }
        return self._decision(all(item["passed"] for item in gates), track, metrics, gates)

    @staticmethod
    def _pairs(details: Sequence[Mapping[str, Any]]) -> list[dict[str, Mapping[str, Any]]]:
        grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
        for item in details:
            grouped.setdefault(str(item.get("pair_id", "")), {})[str(item.get("group", ""))] = item
        return [value for _, value in sorted(grouped.items()) if "baseline" in value and "candidate" in value]

    @staticmethod
    def _metric(item: Mapping[str, Any], name: str) -> float:
        metrics = item.get("metrics", {})
        value = metrics.get(name, item.get(name, 0.0)) if isinstance(metrics, Mapping) else item.get(name, 0.0)
        return float(value or 0.0)

    @staticmethod
    def _has_metric(item: Mapping[str, Any], name: str) -> bool:
        metrics = item.get("metrics", {})
        if isinstance(metrics, Mapping) and name in metrics:
            value = metrics[name]
        elif name in item:
            value = item[name]
        else:
            return False
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _cluster_diffs(
        paired: Sequence[Mapping[str, Mapping[str, Any]]],
        diffs: Sequence[float],
    ) -> list[float]:
        grouped: dict[str, list[float]] = {}
        for index, (pair, diff) in enumerate(zip(paired, diffs)):
            candidate = pair["candidate"]
            baseline = pair["baseline"]
            query_id = str(candidate.get("query_id") or baseline.get("query_id") or "")
            # Imported results may predate query_id.  In that case each explicit
            # pair remains one independent unit instead of collapsing everything
            # into an empty-key cluster.
            key = query_id or str(candidate.get("pair_id") or baseline.get("pair_id") or index)
            grouped.setdefault(key, []).append(float(diff))
        return [statistics.fmean(values) for _, values in sorted(grouped.items())]

    @staticmethod
    def _paired_effect_size(diffs: Sequence[float]) -> float:
        """Cohen's dz over independent, query-level paired differences."""

        if len(diffs) < 2:
            return 0.0
        spread = statistics.stdev(diffs)
        return statistics.fmean(diffs) / spread if spread > 1e-12 else 0.0

    def _mean_delta(self, candidate: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]], name: str) -> float:
        return statistics.fmean(self._metric(c, name) - self._metric(b, name) for b, c in zip(baseline, candidate))

    def _relative_change(self, candidate: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]], name: str) -> float:
        base = statistics.fmean(self._metric(item, name) for item in baseline)
        current = statistics.fmean(self._metric(item, name) for item in candidate)
        return (current - base) / base if base else (0.0 if current == 0 else math.inf)

    def _relative_p95_change(self, candidate: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]], name: str) -> float:
        def p95(values: list[float]) -> float:
            values.sort()
            return values[min(len(values) - 1, math.ceil(len(values) * 0.95) - 1)]
        base = p95([self._metric(item, name) for item in baseline])
        current = p95([self._metric(item, name) for item in candidate])
        return (current - base) / base if base else (0.0 if current == 0 else math.inf)

    def _improved_types(self, paired: Sequence[Mapping[str, Mapping[str, Any]]]) -> list[str]:
        buckets: dict[str, list[float]] = {}
        for pair in paired:
            name = str(pair["candidate"].get("query_type", "unknown"))
            buckets.setdefault(name, []).append(
                self._metric(pair["candidate"], "composite_score") - self._metric(pair["baseline"], "composite_score")
            )
        return sorted(name for name, values in buckets.items() if statistics.fmean(values) > 0)

    @staticmethod
    def _gate(name: str, passed: bool, observed: Any) -> dict[str, Any]:
        return {"gate": name, "passed": bool(passed), "observed": observed}

    @staticmethod
    def _decision(eligible: bool, track: str, metrics: Mapping[str, Any], gates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        payload = {"track": track, "eligible": bool(eligible), "metrics": dict(metrics), "gates": list(gates)}
        payload["decision_sha256"] = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        return payload


def save_promotion_decision(path: str | Path, decision: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(decision), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
