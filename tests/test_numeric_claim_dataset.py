import json
import hashlib
from collections import Counter
from pathlib import Path

import yaml


DATASET = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "harness_evolution"
    / "datasets"
    / "numeric_claims_v1.jsonl"
)


def _records():
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_numeric_claim_dataset_is_valid_unique_and_balanced():
    records = _records()
    required = {"id", "claim_type", "label", "split", "claim", "source_span", "expected_normalized"}
    assert len(records) == 60
    assert len({record["id"] for record in records}) == 60
    assert all(set(record) == required for record in records)
    assert all(record["claim"] and record["source_span"] for record in records)
    assert all(isinstance(record["expected_normalized"], dict) for record in records)

    combinations = Counter((record["claim_type"], record["label"]) for record in records)
    assert set(combinations) == {
        (claim_type, label)
        for claim_type in ("number", "percentage", "currency_unit", "date_fiscal_year")
        for label in ("supported", "contradicted", "unknown")
    }
    assert set(combinations.values()) == {5}


def test_numeric_claim_dataset_uses_three_development_and_two_held_out_per_cell():
    records = _records()
    split_counts = Counter(
        (record["claim_type"], record["label"], record["split"]) for record in records
    )
    for claim_type in ("number", "percentage", "currency_unit", "date_fiscal_year"):
        for label in ("supported", "contradicted", "unknown"):
            assert split_counts[(claim_type, label, "development")] == 3
            assert split_counts[(claim_type, label, "held_out")] == 2


CHALLENGE_MANIFEST = DATASET.with_name("numeric_challenge_v1.yaml")
CHALLENGE_DEVELOPMENT = DATASET.with_name("numeric_challenge_v1_development.jsonl")
CHALLENGE_HELD_OUT = DATASET.parent / "private" / "numeric_challenge_v1_held_out.jsonl"


def _challenge_records(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_numeric_challenge_manifest_is_versioned_and_hash_pinned():
    manifest = yaml.safe_load(CHALLENGE_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == "numeric_challenge_v1"
    assert manifest["version"] == "v1"
    assert manifest["schema_version"] == 2
    assert manifest["record_schema"] == "numeric_claim_v2"
    assert manifest["files"]["development"]["count"] == 48
    assert manifest["files"]["held_out"]["count"] == 24
    for split, path in (("development", CHALLENGE_DEVELOPMENT), ("held_out", CHALLENGE_HELD_OUT)):
        expected = manifest["files"][split]["sha256"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert expected == actual
    assert manifest["isolation"] == {
        "development_public": True,
        "held_out_private": True,
        "held_out_exposes_row_labels": False,
        "evaluator_output": "aggregate_only",
    }


def test_numeric_challenge_has_disjoint_balanced_failure_splits_and_structured_sources():
    manifest = yaml.safe_load(CHALLENGE_MANIFEST.read_text(encoding="utf-8"))
    development = _challenge_records(CHALLENGE_DEVELOPMENT)
    held_out = _challenge_records(CHALLENGE_HELD_OUT)
    assert len(development) == 48
    assert len(held_out) == 24
    records = development + held_out
    assert len({row["id"] for row in records}) == len(records)
    assert {row["split"] for row in development} == {"development"}
    assert {row["split"] for row in held_out} == {"held_out"}
    assert {row["id"] for row in development}.isdisjoint({row["id"] for row in held_out})
    assert set(manifest["splits"]["development"]) == {row["id"] for row in development}
    assert set(manifest["splits"]["held_out"]) == {row["id"] for row in held_out}
    # Held-out IDs are intentionally opaque: labels/failure classes must not
    # leak through a public manifest even when the evaluator exposes IDs.
    assert all(row["id"].startswith("hc_") for row in held_out)
    assert all(not any(label in row["id"] for label in ("supported", "contradicted", "unknown")) for row in held_out)

    failure_types = set(manifest["failure_types"])
    assert len(failure_types) == 8
    assert set(row["failure_type"] for row in records) == failure_types
    for split_rows, expected_per_type in ((development, 6), (held_out, 3)):
        assert Counter(row["failure_type"] for row in split_rows) == {
            failure_type: expected_per_type for failure_type in failure_types
        }
        assert Counter(row["label"] for row in split_rows) == {
            "supported": len(split_rows) // 3,
            "contradicted": len(split_rows) // 3,
            "unknown": len(split_rows) // 3,
        }
    for row in records:
        assert isinstance(row["sources"], list) and row["sources"]
        assert all(source.get("source_id") and source.get("source_span") for source in row["sources"])
        facts = row["expected_normalized"]["facts"]
        assert facts and all({"entity", "value", "unit", "kind"} <= set(fact) for fact in facts)


def test_numeric_challenge_contains_tolerance_and_primitive_gap_signals():
    rows = _challenge_records(CHALLENGE_DEVELOPMENT) + _challenge_records(CHALLENGE_HELD_OUT)
    by_type = Counter(row["failure_type"] for row in rows)
    assert by_type["multi_entity_alignment"] == 9
    assert by_type["unit_missing_or_conversion"] == 9
    assert by_type["derived_percentage"] == 9
    # Close-but-not-equal observations exercise v0001's 0.1% tolerance while
    # remaining valid contradictions for a zero-tolerance candidate.
    close_rows = [
        row
        for row in rows
        if row["label"] == "contradicted"
        and any(token in row["failure_type"] for token in ("currency_scale", "equivalent_numeric"))
    ]
    assert len(close_rows) >= 3
    assert any(row["failure_type"] == "multi_entity_alignment" for row in rows)
    assert any(row["failure_type"] == "unit_missing_or_conversion" for row in rows)
    assert any(row["failure_type"] == "derived_percentage" for row in rows)
