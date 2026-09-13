import json

from src.evidence.schemas import Claim, Evidence, VerificationStatus
from src.evidence.verifier import EvidenceVerifier
from src.harness_evolution.experiment import _matches_expected_normalized, run_numeric_skill_experiment
from src.harness_evolution.numeric_skill import NumericFact, NumericVerificationSkill

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = PROJECT_ROOT / "configs" / "harness_evolution" / "registries"


SPEC = {
    "primitives": [
        "detect_numeric_claim",
        "normalize_number",
        "normalize_percentage",
        "normalize_currency_unit",
        "normalize_date_or_fiscal_year",
        "compare_value_unit_date",
        "resolve_source_conflict",
        "classify_supported_contradicted_unknown",
    ],
    "settings": {"relative_tolerance": 0.001, "conflict_policy": "unknown", "fallback": "unknown"},
}


def test_numeric_skill_matches_percent_currency_and_year():
    skill = NumericVerificationSkill(SPEC)
    claim = Claim("c1", "公司在 FY2025 的营收为 US$2.5 billion，同比增长 20%。")
    evidence = [Evidence(source_span="FY2025 revenue was US$2.5 billion, an increase of 20%.")]
    result = skill(claim, evidence)
    assert result["status"] == "supported"


def test_numeric_skill_does_not_treat_four_digit_counts_as_year():
    skill = NumericVerificationSkill(SPEC)
    claim = Claim("c-count", "系统处理了 2048 个样本。")
    evidence = [Evidence(source_span="本次实验总计处理样本 2,048 个。")]

    claim_facts = skill.extract_facts(claim.text)
    assert [(fact.kind, fact.value) for fact in claim_facts] == [("number", 2048.0)]
    assert skill(claim, evidence)["status"] == "supported"


def test_numeric_skill_does_not_treat_weight_units_as_scales():
    skill = NumericVerificationSkill(SPEC)
    chinese = skill.extract_facts("设备重量为 2.5 千克。")
    english = skill.extract_facts("The device weighs 2.5 kg.")

    assert [(fact.kind, fact.value, fact.unit) for fact in chinese] == [
        ("measurement", 2.5, "kg")
    ]
    assert [(fact.kind, fact.value, fact.unit) for fact in english] == [
        ("measurement", 2.5, "kg")
    ]


def test_numeric_skill_ignores_markdown_citation_numbers():
    skill = NumericVerificationSkill(SPEC)
    facts = skill.extract_facts("2024 年增长率为 20%。[1]")

    assert [(fact.kind, fact.value) for fact in facts] == [
        ("date", 2024.0),
        ("percentage", 20.0),
    ]


def test_numeric_skill_detects_mismatch_and_conflict():
    skill = NumericVerificationSkill(SPEC)
    claim = Claim("c2", "2024 年增长率为 20%。")
    mismatch = skill(claim, [Evidence(source_span="2024 年增长率为 12%。")])
    assert mismatch["status"] == "contradicted"
    conflict = skill(claim, [
        Evidence(source_span="2024 年增长率为 20%。"),
        Evidence(source_span="2024 年增长率为 12%。"),
    ])
    assert conflict["status"] == "unknown"
    assert "conflicting" in conflict["reason"]


def test_evidence_verifier_passes_all_attributable_sources_to_numeric_skill():
    skill = NumericVerificationSkill(SPEC)
    verifier = EvidenceVerifier(policy=skill)
    claim = Claim("c-conflict", "2024 年增长率为 20%。")
    evidence = [
        Evidence(source_span="2024 年增长率为 20%。"),
        Evidence(source_span="2024 年增长率为 12%。"),
    ]

    results = __import__("asyncio").run(
        verifier.verify("", evidence, claims=[claim])
    )

    assert results[0].status == VerificationStatus.UNKNOWN
    assert "conflicting" in results[0].reason


def test_evidence_verifier_keeps_trailing_chinese_citation_bound_to_claim():
    skill = NumericVerificationSkill(SPEC)
    verifier = EvidenceVerifier(policy=skill)
    sources = [
        Evidence(source_span="2024 年增长率为 20%。", metadata={"index": 1}),
    ]

    claims = verifier.extract_claims("2024 年增长率为 20%。[2]")
    assert claims[0].source_refs == ["2"]

    results = __import__("asyncio").run(verifier.verify("", sources, claims=claims))
    assert results[0].status == VerificationStatus.UNKNOWN
    assert results[0].reason == "numeric_skill:no_source_span"


def test_numeric_skill_never_supports_without_span():
    skill = NumericVerificationSkill(SPEC)
    verifier = EvidenceVerifier(policy=skill)
    claim = Claim("c3", "2025 年收入为 100 万美元。")
    results = __import__("asyncio").run(verifier.verify("", [Evidence(source_url="https://x.test")], claims=[claim]))
    assert results[0].status == VerificationStatus.UNKNOWN


def test_numeric_skill_rejects_arbitrary_primitive():
    bad = {"primitives": ["run_shell"], "settings": {}}
    try:
        NumericVerificationSkill(bad)
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("unsafe primitive accepted")


def test_numeric_experiment_checks_normalized_kind_value_and_unit():
    assert _matches_expected_normalized(
        [NumericFact(kind="number", value=2048.0)],
        {"value": 2048, "unit": "count"},
    )
    assert _matches_expected_normalized(
        [NumericFact(kind="measurement", value=2.5, unit="kg")],
        {"value": 2.5, "unit": "kg"},
    )
    assert not _matches_expected_normalized(
        [NumericFact(kind="measurement", value=2500.0, unit="kg")],
        {"value": 2.5, "unit": "kg"},
    )


def test_same_skill_version_does_not_count_as_triggered_mechanism(tmp_path):
    dataset = tmp_path / "claims.jsonl"
    dataset.write_text(
        json.dumps({
            "id": "same-version-1",
            "claim_type": "number",
            "label": "supported",
            "split": "held_out",
            "claim": "共有 10 个样本。",
            "source_span": "实验共计 10 个样本。",
            "expected_normalized": {"value": 10, "unit": "count"},
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    result = run_numeric_skill_experiment(
        registry_dir=REGISTRY_DIR,
        baseline_registry="production",
        candidate_registry="production",
        dataset_path=dataset,
        output_dir=tmp_path / "outputs",
        split="held_out",
        experiment_id="same-version-mechanism",
    )
    candidate_rows = [item for item in result["details"] if item["group"] == "candidate"]
    assert candidate_rows and not candidate_rows[0]["mechanism_triggered"]


def test_numeric_experiment_accepts_multi_source_and_structured_expected_facts(tmp_path):
    dataset = tmp_path / "claims.jsonl"
    dataset.write_text(
        json.dumps({
            "id": "multi-source-1",
            "claim_type": "number",
            "label": "supported",
            "split": "development",
            "claim": "共有 10 个样本。",
            "sources": [
                {
                    "source_id": "primary-1",
                    "source_tier": "primary",
                    "source_span": "实验共计 10 个样本。",
                },
            ],
            "expected_normalized": {"facts": [{"value": 10, "unit": "count"}]},
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    result = run_numeric_skill_experiment(
        registry_dir=REGISTRY_DIR,
        baseline_registry="production",
        candidate_registry="production",
        dataset_path=dataset,
        output_dir=tmp_path / "outputs",
        split="development",
        experiment_id="multi-source-expected-facts",
    )
    assert all(item["normalization_correct"] for item in result["details"])


def test_normalization_metric_rejects_unsupported_iso_date_without_crashing():
    assert not _matches_expected_normalized(
        [NumericFact(kind="date", value=2024, unit="year")],
        {"value": "2024-12-31", "unit": "date"},
    )
