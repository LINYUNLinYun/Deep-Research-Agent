import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from evaluation.benchmarks.research_bench import ResearchBench
from src.harness_evolution.manifest import RunManifest, stable_hash
from src.harness_evolution.registry import RegistryError, VersionRegistry, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs" / "harness_evolution"


def _rehash(decision: dict) -> dict:
    decision = dict(decision)
    decision.pop("decision_sha256", None)
    decision["decision_sha256"] = hashlib.sha256(
        json.dumps(decision, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return decision


def _write_promotion_decision(
    tmp_path: Path,
    name: str,
    *,
    baseline,
    candidate,
    eligible: bool = True,
    baseline_sha256: str | None = None,
) -> Path:
    development_path = tmp_path / f"{name}-development.json"
    development = _rehash({
        "eligible": True,
        "traceability": {"split": "development"},
    })
    development_path.write_text(json.dumps(development), encoding="utf-8")
    evaluation_path = tmp_path / f"{name}-evaluation.json"
    evaluation_path.write_text(json.dumps({"details_redacted": True}), encoding="utf-8")
    track = "policy" if candidate.artifact_id == "search_control_policy" else "skill"
    decision = _rehash({
        "track": track,
        "eligible": eligible,
        "gates": [{"gate": "test", "passed": True}],
        "evaluation_path": str(evaluation_path),
        "baseline": {
            "artifact_id": baseline.artifact_id,
            "version": baseline.version,
            "sha256": baseline.sha256 if baseline_sha256 is None else baseline_sha256,
        },
        "candidate": {
            "artifact_id": candidate.artifact_id,
            "version": candidate.version,
            "sha256": candidate.sha256,
        },
        "traceability": {
            "split": "held_out",
            "dataset_sha256": "1" * 64,
            "evaluator_sha256": "2" * 64,
            "evaluation_sha256": sha256_file(evaluation_path),
            "development_prerequisite": {
                "path": str(development_path),
                "sha256": development["decision_sha256"],
            },
        },
    })
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(decision), encoding="utf-8")
    return path


def _isolated_registry(tmp_path: Path) -> VersionRegistry:
    root = tmp_path / "harness_evolution"
    shutil.copytree(CONFIG_ROOT, root)
    return VersionRegistry(root / "registries")


def test_shipped_registries_resolve_relative_paths_and_hashes():
    registry = VersionRegistry(CONFIG_ROOT / "registries")
    for registry_name in ("production", "candidate", "canary"):
        policy = registry.resolve(registry_name, "search_control_policy")
        skill = registry.resolve(registry_name, "verify_numeric_claim_skill")
        assert policy.version == ("v0003" if registry_name == "candidate" else "v0001")
        assert skill.version == "v0001"
        assert Path(policy.path).is_absolute()
        assert policy.sha256 == sha256_file(policy.path)
        assert skill.sha256 == sha256_file(skill.path)


def test_hash_mismatch_is_rejected(tmp_path):
    registry = _isolated_registry(tmp_path)
    reference = registry.resolve("production", "search_control_policy")
    Path(reference.path).write_text(Path(reference.path).read_text() + "\n# tampered\n")
    with pytest.raises(RegistryError, match="sha256 mismatch"):
        registry.resolve("production", "search_control_policy")


def test_mirroring_production_clears_cross_track_candidate_contamination(tmp_path):
    registry = _isolated_registry(tmp_path)
    production_skill = registry.resolve("production", "verify_numeric_claim_skill")
    stale_skill = registry.create_candidate(
        production_skill, {"parameters": {"relative_tolerance": 0.002}}
    )
    assert registry.resolve("candidate", "verify_numeric_claim_skill").version == stale_skill.version

    registry.mirror_registry("production", "candidate")

    for artifact_id in ("search_control_policy", "verify_numeric_claim_skill"):
        production = registry.resolve("production", artifact_id)
        candidate = registry.resolve("candidate", artifact_id)
        assert (candidate.version, candidate.sha256) == (production.version, production.sha256)


def test_candidate_is_immutable_isolated_and_promoted_atomically(tmp_path):
    registry = _isolated_registry(tmp_path)
    production = registry.resolve("production", "search_control_policy")
    candidate = registry.create_candidate(production, {"parameters": {"novelty_threshold": 0.3}})

    assert candidate.version != production.version
    assert candidate.parent == "v0001"
    assert registry.resolve("production", candidate.artifact_id).version == "v0001"
    assert registry.resolve("candidate", candidate.artifact_id) == candidate
    with pytest.raises(FileExistsError):
        Path(candidate.path).open("x")

    decision_path = _write_promotion_decision(
        tmp_path, "decision", baseline=production, candidate=candidate
    )
    registry.promote(candidate, decision_path)
    assert registry.resolve("production", candidate.artifact_id).version == candidate.version

    registry.rollback(candidate.artifact_id, production)
    registry.rollback(candidate.artifact_id, production)
    assert registry.resolve("production", candidate.artifact_id).version == "v0001"
    assert Path(candidate.path).exists()


def test_invalid_or_stale_candidate_cannot_be_promoted(tmp_path):
    registry = _isolated_registry(tmp_path)
    production = registry.resolve("production", "search_control_policy")
    candidate = registry.create_candidate(production, {"parameters": {"novelty_threshold": 0.3}})
    decision_path = _write_promotion_decision(
        tmp_path, "decision-ineligible", baseline=production, candidate=candidate, eligible=False
    )
    with pytest.raises(RegistryError, match="not eligible"):
        registry.promote(candidate, decision_path)

    decision_path = _write_promotion_decision(
        tmp_path, "decision-stale", baseline=production, candidate=candidate
    )
    newer = registry.create_candidate(production, {"parameters": {"novelty_threshold": 0.35}})
    # Simulate a different candidate becoming production first.
    newer_decision = _write_promotion_decision(
        tmp_path, "newer", baseline=production, candidate=newer
    )
    registry.promote(newer, newer_decision)
    with pytest.raises(RegistryError, match="stale candidate"):
        registry.promote(candidate, decision_path)


def test_candidate_cannot_be_promoted_against_wrong_parent_hash(tmp_path):
    registry = _isolated_registry(tmp_path)
    production = registry.resolve("production", "search_control_policy")
    candidate = registry.create_candidate(
        production, {"parameters": {"novelty_threshold": 0.3}}
    )
    decision_path = _write_promotion_decision(
        tmp_path,
        "wrong-parent-hash",
        baseline=production,
        candidate=candidate,
        baseline_sha256="0" * 64,
    )

    with pytest.raises(RegistryError, match="baseline hash changed"):
        registry.promote(candidate, decision_path)


def test_promotion_rejects_missing_hash_and_development_only_decisions(tmp_path):
    registry = _isolated_registry(tmp_path)
    production = registry.resolve("production", "search_control_policy")
    candidate = registry.create_candidate(
        production, {"parameters": {"novelty_threshold": 0.3}}
    )
    missing_hash = tmp_path / "missing-hash.json"
    missing_hash.write_text(json.dumps({
        "eligible": True,
        "candidate": {
            "artifact_id": candidate.artifact_id,
            "version": candidate.version,
            "sha256": candidate.sha256,
        },
    }), encoding="utf-8")
    with pytest.raises(RegistryError, match="missing decision_sha256"):
        registry.promote(candidate, missing_hash)

    decision_path = _write_promotion_decision(
        tmp_path, "wrong-split", baseline=production, candidate=candidate
    )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["traceability"]["split"] = "development"
    decision_path.write_text(json.dumps(_rehash(decision)), encoding="utf-8")
    with pytest.raises(RegistryError, match="held-out"):
        registry.promote(candidate, decision_path)


def test_schema_rejects_unknown_action_and_out_of_range_threshold(tmp_path):
    registry = _isolated_registry(tmp_path)
    production = registry.resolve("production", "search_control_policy")
    with pytest.raises(RegistryError, match="invalid"):
        registry.create_candidate(production, {"parameters": {"novelty_threshold": 1.5}})
    with pytest.raises(RegistryError, match="invalid"):
        registry.create_candidate(
            production,
            {"rules": [{"id": "bad", "priority": 1, "all": [{"signal": "result_count", "operator": "==", "value": 0}], "action": {"type": "run_shell"}}]},
        )


def test_researchbench_manifest_is_complete_and_disjoint():
    data = yaml.safe_load((CONFIG_ROOT / "datasets" / "researchbench_v1.yaml").read_text())
    splits = data["splits"]
    all_ids = [item for split in splits.values() for item in split]
    expected_ids = {item["id"] for item in ResearchBench.DEFAULT_QUESTIONS}
    assert len(all_ids) == data["expected_count"] == 35
    assert len(set(all_ids)) == len(all_ids)
    assert set(all_ids) == expected_ids
    assert {key: len(value) for key, value in splits.items()} == {
        "miner": 12,
        "development": 9,
        "held_out": 14,
    }


def test_run_manifest_records_reproducible_versions_and_is_write_once(tmp_path):
    registry = VersionRegistry(CONFIG_ROOT / "registries")
    policy = registry.resolve("production", "search_control_policy")
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text('{"result": 1}\n', encoding="utf-8")
    config = {"model": {"temperature": 0, "top_p": 1}}
    manifest = RunManifest.create(
        run_id="run-1",
        experiment_id="exp-1",
        pair_id="pair-1",
        query_id="tech_001",
        group="baseline",
        registry=CONFIG_ROOT / "registries" / "production.yaml",
        artifacts={policy.artifact_id: policy},
        config=config,
        executor={"backend": "fake", "model": "frozen", "temperature": 0},
        budgets={"timeout": 60, "tool_calls": 5, "max_tokens": 2048},
        memory_session="isolated-run-1",
        dataset={"id": "researchbench_v1", "split": "development", "version": "v1"},
        tool={"mode": "replay", "fixture_path": str(fixture)},
        evaluator={"version": "v1", "weights": {"quality": 1.0}},
        created_at="2026-09-08T00:00:00Z",
        repo_root=PROJECT_ROOT,
    )
    assert manifest.config_sha256 == stable_hash(config)
    assert manifest.artifacts[policy.artifact_id]["sha256"] == policy.sha256
    assert manifest.tool["fixture_sha256"] == sha256_file(fixture)
    destination = tmp_path / "manifest.json"
    written_hash = manifest.write(destination)
    assert written_hash == sha256_file(destination)
    assert RunManifest.read(destination).to_dict() == manifest.to_dict()
    with pytest.raises(FileExistsError, match="immutable"):
        manifest.write(destination)
