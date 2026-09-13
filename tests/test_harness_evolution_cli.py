import json
import shutil
import subprocess
import sys
from pathlib import Path

from src.harness_evolution.registry import VersionRegistry
from scripts.run_harness_evolution import _evaluator_sha256, _validate_imported_results


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_harness_evolution.py"
CONFIG_ROOT = PROJECT_ROOT / "configs" / "harness_evolution"


def _run(registry_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--registry-dir", str(registry_dir), *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": "."},
    )


def test_mine_rejects_heldout_split_from_parent_evaluation(tmp_path):
    registry_dir = tmp_path / "registries"
    registry_dir.mkdir()
    heldout = tmp_path / "heldout.json"
    heldout.write_text(json.dumps({
        "split": "heldout",
        "details": [{"query_id": "secret", "metrics": {}}],
    }), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--registry-dir",
            str(registry_dir),
            "mine",
            "--input",
            str(heldout),
            "--output",
            str(tmp_path / "experiences.jsonl"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": "."},
    )
    assert result.returncode != 0
    assert "held-out records cannot enter candidate mining" in result.stderr


def test_imported_results_must_match_current_artifacts(tmp_path):
    isolated_root = tmp_path / "harness_evolution"
    shutil.copytree(CONFIG_ROOT, isolated_root)
    manager = VersionRegistry(isolated_root / "registries")
    baseline = manager.resolve("production", "search_control_policy")
    candidate = manager.resolve("candidate", "search_control_policy")
    evaluation = {
        "evaluator_sha256": _evaluator_sha256(),
        "artifacts": {
            "baseline": {
                "artifact_id": baseline.artifact_id,
                "version": baseline.version,
                "sha256": baseline.sha256,
            },
            "candidate": {
                "artifact_id": candidate.artifact_id,
                "version": candidate.version,
                "sha256": "0" * 64,
            },
        },
    }

    try:
        _validate_imported_results(evaluation, baseline=baseline, candidate=candidate)
    except ValueError as exc:
        assert "candidate artifact mismatch" in str(exc)
    else:
        raise AssertionError("stale imported result was accepted")


def test_offline_cli_mine_propose_evaluate_report_promote_rollback(tmp_path):
    isolated_root = tmp_path / "harness_evolution"
    shutil.copytree(CONFIG_ROOT, isolated_root)
    registry_dir = isolated_root / "registries"

    # Start the copied candidate pointer from production even though the
    # repository intentionally ships a rejected v0002 example.
    shutil.copyfile(registry_dir / "production.yaml", registry_dir / "candidate.yaml")

    miner_input = tmp_path / "development.json"
    miner_input.write_text(json.dumps({
        "run_id": "dev-1",
        "query_id": "dev-1",
        "dataset_split": "development",
        "harness": {
            "search": {"stats": {"calls": 1, "backend_calls": 1, "new_results": 0, "rewritten_queries": 0}},
            "evidence_verification": {"total_claims": 1, "unknown": 1},
        },
    }), encoding="utf-8")
    experiences = tmp_path / "experiences.jsonl"
    _run(registry_dir, "mine", "--input", str(miner_input), "--output", str(experiences))

    proposal = tmp_path / "proposal.json"
    _run(
        registry_dir,
        "propose", "--track", "skill", "--experiences", str(experiences), "--output", str(proposal),
    )
    manager = VersionRegistry(registry_dir)
    candidate = manager.resolve("candidate", "verify_numeric_claim_skill")
    assert candidate.parent == "v0001"

    # v0001's 0.1% tolerance treats these close-but-different values as
    # supported.  The enumerated exact-tolerance candidate correctly returns
    # contradicted in two claim categories, satisfying the transfer gate.
    claims = tmp_path / "claims.jsonl"
    rows = []
    for claim_type, unit in (("number", ""), ("currency", "美元")):
        for index in range(2):
            rows.append({
                "id": f"{claim_type}-{index}",
                "claim_type": claim_type,
                "claim": f"指标为100{unit}",
                "source_span": f"官方记录显示指标为100.05{unit}",
                "label": "contradicted",
                "split": "held_out",
            })
    claims.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    output_root = tmp_path / "outputs"
    missing_prerequisite = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--registry-dir", str(registry_dir),
            "evaluate", "--track", "skill", "--claim-dataset", str(claims),
            "--split", "held_out", "--experiment-id", "blocked-heldout",
            "--output-dir", str(output_root),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": "."},
    )
    assert missing_prerequisite.returncode != 0
    assert "requires --development-decision" in missing_prerequisite.stderr
    assert not (output_root / "blocked-heldout").exists()

    development_claims = tmp_path / "development-claims.jsonl"
    development_claims.write_text(
        "".join(json.dumps({**row, "split": "development"}, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    _run(
        registry_dir,
        "evaluate", "--track", "skill", "--claim-dataset", str(development_claims),
        "--split", "development", "--experiment-id", "cli-development", "--output-dir", str(output_root),
    )
    development_decision = output_root / "cli-development" / "promotion_decision.json"
    _run(
        registry_dir,
        "evaluate", "--track", "skill", "--claim-dataset", str(claims),
        "--split", "held_out", "--experiment-id", "cli-e2e", "--output-dir", str(output_root),
        "--development-decision", str(development_decision),
    )
    experiment = output_root / "cli-e2e"
    public_evaluation = json.loads((experiment / "evaluation.json").read_text(encoding="utf-8"))
    assert public_evaluation["details_redacted"] is True
    assert "details" not in public_evaluation
    assert public_evaluation["traceability"]["dataset_sha256"]
    assert public_evaluation["traceability"]["evaluator_sha256"]
    public_text = json.dumps(public_evaluation, ensure_ascii=False)
    assert all(secret not in public_text for secret in ("expected", "predicted", "reason"))
    decision = json.loads((experiment / "promotion_decision.json").read_text(encoding="utf-8"))
    assert decision["eligible"] is True
    assert decision["traceability"]["details_persisted"] is False
    assert decision["traceability"]["development_prerequisite"]["path"] == str(development_decision)
    assert "details" not in decision
    _run(registry_dir, "report", "--experiment", str(experiment))
    report = (experiment / "promotion_report.md").read_text(encoding="utf-8")
    assert "expected" not in report and "predicted" not in report and "reason" not in report

    _run(registry_dir, "promote", "--decision", str(experiment / "promotion_decision.json"))
    assert manager.resolve("production", "verify_numeric_claim_skill").version == candidate.version
    _run(
        registry_dir,
        "rollback", "--artifact", "verify_numeric_claim_skill", "--version", "v0001",
    )
    assert manager.resolve("production", "verify_numeric_claim_skill").version == "v0001"
