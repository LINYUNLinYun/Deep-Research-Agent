import json

from src.harness_evolution.evaluation import DatasetManifest, PromotionGate, redact_held_out_evaluation


def row(pair, group, score, kind, **metrics):
    return {
        "pair_id": pair, "group": group, "query_type": kind,
        "metrics": {
            "composite_score": score,
            "unsupported_claim_rate": 0.1,
            "critical_factual_errors": 0,
            "false_supported_rate": 0,
            "total_tokens": 100,
            "latency_seconds": 1,
            "evidence_yield": 1,
            "claim_macro_f1": 0.8,
            **metrics,
        },
        "mechanism_triggered": group == "candidate",
    }


def test_dataset_manifest_rejects_split_leakage():
    try:
        DatasetManifest({"splits": {"miner": ["a"], "development": ["b"], "held_out": ["a"]}})
    except ValueError as exc:
        assert "leakage" in str(exc)
    else:
        raise AssertionError("split leakage accepted")


def test_promotion_gate_accepts_clear_two_type_gain():
    details = []
    for index in range(20):
        kind = "tech" if index % 2 else "finance"
        details += [row(str(index), "baseline", 0.5, kind), row(str(index), "candidate", 0.7, kind)]
    decision = PromotionGate().decide({"details": details}, track="policy")
    assert decision["eligible"] is True
    assert decision["metrics"]["win_tie_loss"]["win"] == 20


def test_promotion_gate_rejects_cost_or_factual_regression():
    details = []
    for index in range(20):
        kind = "tech" if index % 2 else "finance"
        details += [
            row(str(index), "baseline", 0.5, kind),
            row(str(index), "candidate", 0.7, kind, total_tokens=130, false_supported_rate=0.2),
        ]
    decision = PromotionGate().decide({"details": details}, track="skill")
    assert decision["eligible"] is False
    failed = {item["gate"] for item in decision["gates"] if not item["passed"]}
    assert {"token_budget", "false_supported"} <= failed


def test_promotion_gate_fails_closed_when_false_supported_metric_is_missing():
    details = []
    for index in range(20):
        kind = "tech" if index % 2 else "finance"
        baseline = row(str(index), "baseline", 0.5, kind)
        candidate = row(str(index), "candidate", 0.7, kind)
        baseline["metrics"].pop("false_supported_rate")
        candidate["metrics"].pop("false_supported_rate")
        details += [baseline, candidate]

    decision = PromotionGate().decide({"details": details}, track="policy")

    gate = next(item for item in decision["gates"] if item["gate"] == "false_supported")
    assert decision["eligible"] is False
    assert gate["observed"] == {"available": False, "delta": None}


def test_promotion_statistics_cluster_repeats_by_query():
    details = []
    for qid, kind in (("q1", "tech"), ("q2", "finance")):
        for repeat in range(3):
            pair_id = f"{qid}:r{repeat}"
            baseline = row(pair_id, "baseline", 0.5, kind)
            candidate = row(pair_id, "candidate", 0.7, kind)
            baseline["query_id"] = candidate["query_id"] = qid
            details += [baseline, candidate]

    decision = PromotionGate().decide({"details": details}, track="policy")

    assert decision["metrics"]["quality"]["n"] == 2
    assert decision["metrics"]["win_tie_loss"] == {"win": 2, "tie": 0, "loss": 0}


def test_held_out_public_evaluation_redacts_row_labels_and_keeps_aggregates():
    raw = {
        "experiment_id": "secret-exp",
        "split": "held_out",
        "dataset_sha256": "dataset-hash",
        "evaluator_sha256": "evaluator-hash",
        "details": [
            {
                "pair_id": "secret-pair",
                "query_id": "secret-query",
                "group": "baseline",
                "expected": "supported",
                "predicted": "contradicted",
                "reason": "secret failure reason",
                "telemetry": {"source_span": "private source"},
                "metrics": {"composite_score": 0.0, "total_tokens": 10, "expected": 1},
                "mechanism_triggered": False,
            },
            {
                "pair_id": "secret-pair",
                "query_id": "secret-query",
                "group": "candidate",
                "expected": "supported",
                "predicted": "supported",
                "reason": "secret success reason",
                "metrics": {"composite_score": 1.0, "total_tokens": 11},
                "mechanism_triggered": True,
            },
        ],
    }
    public = redact_held_out_evaluation(raw)
    encoded = json.dumps(public, ensure_ascii=False)

    assert public["details_redacted"] is True
    assert "details" not in public
    assert public["traceability"] == {
        "dataset_sha256": "dataset-hash",
        "evaluator_sha256": "evaluator-hash",
        "split": "held_out",
    }
    assert public["aggregate"]["pair_count"] == 1
    assert public["aggregate"]["groups"]["candidate"]["metrics"]["composite_score"] == 1.0
    for secret in ("secret-query", "supported", "contradicted", "secret failure reason", "private source"):
        assert secret not in encoded


def test_development_public_evaluation_remains_unchanged():
    raw = {"split": "development", "details": [{"expected": "supported"}]}
    assert redact_held_out_evaluation(raw) == raw
