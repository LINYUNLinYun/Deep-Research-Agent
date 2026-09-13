import asyncio

from src.harness_evolution.candidate import CandidateGenerator, apply_candidate_patch
from src.harness_evolution.experience import FailureMiner


def test_failure_miner_labels_without_llm():
    labels = FailureMiner().label({"harness": {
        "search": {"stats": {"calls": 2, "backend_calls": 2, "duplicate_results": 5, "new_results": 1, "rewritten_queries": 0}},
        "evidence_verification": {"total_claims": 2, "unknown": 1, "contradicted": 1},
    }})
    assert "duplicate_search" in labels
    assert "unsupported_claim" in labels
    assert "contradicted_claim" in labels


def test_failure_miner_labels_numeric_evaluation_failures():
    labels = FailureMiner().label({
        "expected": "contradicted",
        "predicted": "supported",
        "normalization_correct": False,
    })
    assert labels == [
        "false_supported_numeric_claim",
        "numeric_normalization_failure",
        "numeric_verdict_mismatch",
    ]


def test_failure_miner_maps_challenge_failure_to_safe_primitive_gap():
    labels = FailureMiner().label({
        "query_type": "multi_entity_alignment",
        "expected": "contradicted",
        "predicted": "supported",
    })
    assert "missing_entity_value_alignment" in labels


def test_failure_miner_reads_nested_policy_decision_signals():
    labels = FailureMiner().label({"harness": {"search": {
        "stats": {"calls": 2, "backend_calls": 1, "new_results": 2},
        "policy_decisions": [
            {
                "action": "accept_results",
                "signals": {
                    "rule_id": "fallback",
                    "policy_signals": {
                        "result_count": 2,
                        "duplicate_ratio": 0.5,
                        "evidence_novelty": 0.0,
                        "unresolved_claims": 0,
                        "remaining_search_budget": 1,
                    },
                },
            },
            {
                "action": "stop_search",
                "signals": {
                    "rule_id": "hard_limit:max_search_attempts",
                    "policy_signals": {
                        "result_count": 0,
                        "duplicate_ratio": 0.0,
                        "evidence_novelty": 0.0,
                        "unresolved_claims": 0,
                        "remaining_search_budget": 1,
                    },
                },
            },
        ],
    }}})

    assert {
        "duplicate_search",
        "low_evidence_novelty",
        "missed_query_rewrite",
        "rewrite_blocked_by_hard_cap",
        "search_hard_cap_reached",
        "unresolved_signal_unavailable",
    } <= set(labels)


def test_candidate_generator_only_changes_one_allowlisted_field():
    base = {
        "parameters": {"novelty_threshold": 0.25, "query_similarity_threshold": 0.88, "max_rewrites": 2},
        "rules": [{
            "id": "rewrite_low_novelty",
            "priority": 100,
            "all": [
                {"signal": "evidence_novelty", "operator": "<", "value": 0.25},
                {"signal": "unresolved_claims", "operator": ">", "value": 0},
            ],
            "action": {"type": "rewrite_uncovered_facets"},
            "limits": {"max_activations_per_run": 2},
        }],
        "hard_limits": {"max_search_attempts": 10, "max_total_rewrites": 2},
    }
    experiences = [{"failure_labels": [
        "duplicate_search", "unresolved_signal_unavailable", "search_hard_cap_reached",
    ]}]
    values = asyncio.run(CandidateGenerator(max_candidates=5).generate("policy", base, experiences))
    assert len(values) == 2
    candidate = apply_candidate_patch(base, values[0])
    assert values[0].field == "rules"
    assert values[0].patch_kind == "rule_condition"
    assert candidate["rules"][0]["all"][1] == {
        "signal": "duplicate_ratio", "operator": ">=", "value": 0.4,
    }
    assert base["rules"][0]["all"][1]["signal"] == "unresolved_claims"
    assert values[1].field == "hard_limits.max_search_attempts"


def test_policy_generator_refuses_unlabelled_noop_candidates():
    base = {"parameters": {"novelty_threshold": 0.25, "query_similarity_threshold": 0.88, "max_rewrites": 2}}
    assert asyncio.run(CandidateGenerator().generate("policy", base, [])) == []


def test_policy_screening_drops_candidate_that_never_changes_miner_behavior():
    base = {
        "artifact_id": "search_control_policy",
        "version": "v0001",
        "parent": None,
        "description": "test",
        "decision_point": "after_search",
        "parameters": {"novelty_threshold": 0.25, "query_similarity_threshold": 0.88, "max_rewrites": 2},
        "rules": [{
            "id": "rewrite_low_novelty", "priority": 100,
            "all": [
                {"signal": "evidence_novelty", "operator": "<", "value": 0.25},
                {"signal": "unresolved_claims", "operator": ">", "value": 0},
                {"signal": "remaining_search_budget", "operator": ">=", "value": 1},
            ],
            "action": {"type": "rewrite_uncovered_facets"},
            "limits": {"max_activations_per_run": 2},
        }],
        "fallback": {"action": {"type": "accept_results"}},
        "hard_limits": {"max_search_attempts": 10, "max_total_rewrites": 2},
    }
    experience = {
        "failure_labels": ["duplicate_search", "unresolved_signal_unavailable", "search_hard_cap_reached"],
        "harness": {"search": {"policy_decisions": [{"signals": {
            "evidence_novelty": 0.5,
            "duplicate_ratio": 0.5,
            "unresolved_claims": 0,
            "remaining_search_budget": 1,
            "search_attempts": 10,
        }}]}},
    }
    generator = CandidateGenerator()
    values = asyncio.run(generator.generate("policy", base, [experience]))

    assert [item.field for item in values] == ["hard_limits.max_search_attempts"]
    screening = generator.proposal_telemetry["candidate_screening"]
    assert screening[0]["field"] == "rules" and screening[0]["retained"] is False
    assert screening[1]["retained"] is True


def test_skill_candidates_cover_scalar_conflict_and_source_order():
    base = {
        "parameters": {"relative_tolerance": 0.001},
        "conflict_policy": "mark_unknown",
        "source_priority": ["primary", "official", "other"],
        "primitives": [
            "detect_numeric_claim",
            "normalize_number",
            "normalize_percentage",
            "classify_supported_contradicted_unknown",
        ],
    }
    values = asyncio.run(CandidateGenerator(max_candidates=5).generate("skill", base, []))
    assert {value.field for value in values} >= {
        "parameters.relative_tolerance",
        "conflict_policy",
        "source_priority",
    }
    assert not any(
        value.field == "primitives" and set(value.new) == set(value.old)
        for value in values
    )


def test_skill_candidates_add_only_allowlisted_primitive_from_failure_labels():
    base = {
        "parameters": {"relative_tolerance": 0.001},
        "conflict_policy": "mark_unknown",
        "source_priority": ["primary", "official", "other"],
        "primitives": [
            "detect_numeric_claim",
            "compare_value_unit_date",
            "classify_supported_contradicted_unknown",
        ],
    }
    experiences = [{"failure_labels": ["missing_entity_value_alignment"]}]
    values = asyncio.run(CandidateGenerator(max_candidates=5).generate("skill", base, experiences))
    patch = values[0]
    assert patch.field == "primitives"
    assert set(patch.new) - set(patch.old) == {"align_entity_value"}
    assert patch.new[-1] == "classify_supported_contradicted_unknown"


def test_llm_can_only_rank_implemented_safe_primitive_candidates():
    def proposer(_messages):
        return {"content": '{"hypothesis":"prefer units","primitive_suggestions":'
                           '["run_shell","convert_measurement_unit","align_entity_value"]}'}

    base = {
        "parameters": {"relative_tolerance": 0.001},
        "conflict_policy": "mark_unknown",
        "source_priority": ["primary", "official", "other"],
        "primitives": [
            "detect_numeric_claim",
            "compare_value_unit_date",
            "classify_supported_contradicted_unknown",
        ],
    }
    experiences = [{
        "failure_labels": [
            "missing_entity_value_alignment",
            "missing_measurement_unit_conversion",
        ]
    }]
    generator = CandidateGenerator(proposer)
    values = asyncio.run(generator.generate("skill", base, experiences))

    assert set(values[0].new) - set(values[0].old) == {"convert_measurement_unit"}
    assert all("run_shell" not in value.new for value in values if value.field == "primitives")
    assert generator.proposal_telemetry == {
        "requested": True,
        "succeeded": True,
        "structured": True,
        "fallback_reason": "",
        "primitive_suggestions": ["convert_measurement_unit", "align_entity_value"],
        "rejected_suggestions": ["run_shell"],
        "candidate_screening": [],
    }


def test_empty_llm_proposal_is_recorded_as_deterministic_fallback():
    generator = CandidateGenerator(lambda _messages: {"content": ""})
    values = asyncio.run(generator.generate(
        "skill",
        {
            "parameters": {"relative_tolerance": 0.001},
            "conflict_policy": "mark_unknown",
            "source_priority": ["primary", "official", "other"],
            "primitives": [
                "detect_numeric_claim",
                "compare_value_unit_date",
                "classify_supported_contradicted_unknown",
            ],
        },
        [{"failure_labels": ["missing_entity_value_alignment"]}],
    ))

    assert set(values[0].new) - set(values[0].old) == {"align_entity_value"}
    assert generator.proposal_telemetry["succeeded"] is False
    assert generator.proposal_telemetry["fallback_reason"] == "empty_response"


def test_three_development_gaps_enable_only_predeclared_atomic_bundle():
    base = {
        "parameters": {"relative_tolerance": 0.001},
        "conflict_policy": "mark_unknown",
        "source_priority": ["primary", "official", "other"],
        "primitives": [
            "detect_numeric_claim",
            "compare_value_unit_date",
            "classify_supported_contradicted_unknown",
        ],
    }
    experiences = [{
        "failure_labels": [
            "missing_entity_value_alignment",
            "missing_measurement_unit_conversion",
            "missing_irrelevant_number_filter",
        ]
    }]
    values = asyncio.run(CandidateGenerator().generate("skill", base, experiences))

    assert set(values[0].new) - set(values[0].old) == {
        "align_entity_value",
        "convert_measurement_unit",
        "filter_semantically_irrelevant_numbers",
    }
    assert values[0].patch_kind == "atomic_bundle"
    assert values[0].bundle_id == "core_numeric_grounding_v1"
    assert values[0].new[-1] == "classify_supported_contradicted_unknown"


def test_candidate_generator_hides_heldout_records_from_proposer():
    seen = {}
    def proposer(messages):
        seen["prompt"] = messages[0]["content"]
        return {"content": '{"hypothesis":"safe"}'}
    records = [{"query_id": "secret", "dataset_split": "held_out", "failure_labels": ["x"]}]
    asyncio.run(CandidateGenerator(proposer).generate("skill", {"settings": {}}, records))
    assert "secret" not in seen["prompt"]
