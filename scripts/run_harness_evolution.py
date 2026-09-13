#!/usr/bin/env python3
"""Offline, manually promoted Policy/Skill Harness evolution V1."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.runner import load_config
from src.harness_evolution.candidate import CandidateGenerator, apply_candidate_patch
from src.harness_evolution.evaluation import (
    PromotionGate,
    canonical_split,
    is_held_out_split,
    redact_held_out_evaluation,
    save_promotion_decision,
)
from src.harness_evolution.experience import ExperienceStore, FailureMiner
from src.harness_evolution.experiment import run_numeric_skill_experiment, run_policy_replay_experiment
from src.harness_evolution.registry import VersionRef, VersionRegistry, sha256_file
from src.models.model_router import ModelRouter


DEFAULT_ROOT = PROJECT_ROOT / "configs" / "harness_evolution"


def _manager(args: argparse.Namespace) -> VersionRegistry:
    return VersionRegistry(Path(args.registry_dir).resolve())


def _artifact_id(track: str) -> str:
    return "search_control_policy" if track == "policy" else "verify_numeric_claim_skill"


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _rehash_decision(decision: dict[str, Any]) -> dict[str, Any]:
    decision.pop("decision_sha256", None)
    decision["decision_sha256"] = hashlib.sha256(
        json.dumps(decision, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return decision


def _evaluator_sha256() -> str:
    """Hash the code that turns raw paired results into a gate decision."""

    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "harness_evolution" / "evaluation.py",
        PROJECT_ROOT / "src" / "harness_evolution" / "experiment.py",
        PROJECT_ROOT / "src" / "harness_evolution" / "numeric_skill.py",
        PROJECT_ROOT / "src" / "harness_evolution" / "numeric_primitives.py",
        PROJECT_ROOT / "src" / "harness_evolution" / "policy.py",
        PROJECT_ROOT / "src" / "harness_evolution" / "replay.py",
        PROJECT_ROOT / "src" / "harness_evolution" / "schemas.py",
        PROJECT_ROOT / "src" / "tools" / "search_controller.py",
        PROJECT_ROOT / "evaluation" / "metrics" / "rule_based.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_development_decision(
    args: argparse.Namespace,
    *,
    manager: VersionRegistry,
    baseline: VersionRef,
    candidate: VersionRef,
) -> dict[str, Any] | None:
    """Require an eligible, current Development decision before Held-out.

    The check happens before the protected dataset is opened.  Besides the
    gate result, it binds the prerequisite to the exact track, artifacts and
    evaluator implementation used by the requested Held-out run.
    """

    if canonical_split(args.split) != "held_out":
        return None
    if not args.development_decision:
        raise ValueError("held-out evaluation requires --development-decision")
    path = Path(args.development_decision).resolve()
    decision = _read_json(path)
    supplied_hash = str(decision.get("decision_sha256", ""))
    canonical = dict(decision)
    canonical.pop("decision_sha256", None)
    actual_hash = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if not supplied_hash or supplied_hash != actual_hash:
        raise ValueError("development decision hash mismatch")
    if decision.get("eligible") is not True:
        raise ValueError("development decision is not eligible")
    if canonical_split(decision.get("traceability", {}).get("split")) != "development":
        raise ValueError("prerequisite decision must be from the development split")
    if str(decision.get("track", "")) != args.track:
        raise ValueError("development decision track mismatch")
    for group, expected in (("baseline", baseline), ("candidate", candidate)):
        observed = decision.get(group, {})
        identity = (observed.get("artifact_id"), observed.get("version"), observed.get("sha256"))
        expected_identity = (expected.artifact_id, expected.version, expected.sha256)
        if identity != expected_identity:
            raise ValueError(f"development decision {group} artifact mismatch")
    if str(decision.get("traceability", {}).get("evaluator_sha256", "")) != _evaluator_sha256():
        raise ValueError("development decision evaluator mismatch; rerun Development with current code")
    return {"path": str(path), "sha256": supplied_hash}


def _validate_imported_results(
    evaluation: Any,
    *,
    baseline: VersionRef,
    candidate: VersionRef,
) -> None:
    """Bind an imported result file to the artifacts and evaluator that produced it."""

    if not isinstance(evaluation, dict):
        raise ValueError("imported results must be a JSON object")
    artifacts = evaluation.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("imported results must contain baseline/candidate artifact identities")
    for group, expected in (("baseline", baseline), ("candidate", candidate)):
        observed = artifacts.get(group)
        if not isinstance(observed, dict):
            raise ValueError(f"imported results have no {group} artifact identity")
        identity = (observed.get("artifact_id"), observed.get("version"), observed.get("sha256"))
        expected_identity = (expected.artifact_id, expected.version, expected.sha256)
        if identity != expected_identity:
            raise ValueError(f"imported results {group} artifact mismatch")
    evaluator = evaluation.get("evaluator")
    evaluator_hash = evaluation.get("evaluator_sha256")
    if not evaluator_hash and isinstance(evaluator, dict):
        evaluator_hash = evaluator.get("sha256")
    if str(evaluator_hash or "") != _evaluator_sha256():
        raise ValueError("imported results evaluator mismatch; rerun with current code")


def command_mine(args: argparse.Namespace) -> None:
    raw = _read_json(args.input)
    if isinstance(raw, dict) and "details" in raw:
        records = raw["details"]
        input_split = raw.get("dataset_split", raw.get("split", "development"))
    elif isinstance(raw, dict):
        records = [raw]
        input_split = raw.get("dataset_split", raw.get("split", "development"))
    else:
        records = raw
        input_split = "development"
    if not isinstance(records, list):
        raise ValueError("mine input must be a list or evaluation object with details")
    safe = []
    for record in records:
        item = dict(record)
        split = item.get("dataset_split", item.get("split", input_split))
        if is_held_out_split(split):
            raise ValueError("held-out records cannot enter candidate mining")
        item["dataset_split"] = canonical_split(split) or "development"
        safe.append(item)
    mined = FailureMiner().mine(safe)
    store = ExperienceStore(args.output)
    for record in mined:
        store.append(record)
    print(json.dumps({"records": len(mined), "output": str(Path(args.output).resolve())}, ensure_ascii=False))


async def command_propose(args: argparse.Namespace) -> None:
    manager = _manager(args)
    artifact_id = _artifact_id(args.track)
    base_ref = manager.resolve(args.base_registry, artifact_id)
    current = yaml.safe_load(Path(base_ref.path).read_text(encoding="utf-8"))
    experiences = ExperienceStore(args.experiences).read()
    if any(is_held_out_split(item.get("dataset_split", item.get("split", ""))) for item in experiences):
        raise ValueError("candidate input contains held-out records")
    proposer = None
    if args.use_llm:
        config = load_config(args.config)
        model_cfg = config.get("model", {})
        backend = model_cfg.get("backend")
        proposer = ModelRouter.create_backend(
            backend,
            use_cache=False,
            temperature=0,
            top_p=1.0,
            max_tokens=2048,
            thinking=False,
        )
    generator = CandidateGenerator(proposer=proposer, max_candidates=5)
    candidates = await generator.generate(args.track, current, experiences)
    if not candidates:
        raise RuntimeError("no valid bounded candidates were generated")
    if args.select < 0 or args.select >= len(candidates):
        raise ValueError(f"--select must be between 0 and {len(candidates) - 1}")
    selected = candidates[args.select]
    candidate_doc = apply_candidate_patch(current, selected)
    # Start every track candidate from the complete base registry so a stale
    # rejected artifact from another track cannot contaminate attribution.
    manager.mirror_registry(args.base_registry, "candidate")
    candidate_ref = manager.create_candidate(base_ref, candidate_doc)
    payload = {
        "track": args.track,
        "base": asdict(base_ref),
        "candidate": asdict(candidate_ref),
        "selected": vars(selected),
        "alternatives": [vars(item) for item in candidates],
        "llm_proposer_used": bool(args.use_llm),
        "llm_proposer_succeeded": bool(generator.proposal_telemetry.get("succeeded")),
        "proposer_telemetry": dict(generator.proposal_telemetry),
        "experiences_sha256": sha256_file(args.experiences),
        "generator_sha256": sha256_file(PROJECT_ROOT / "src" / "harness_evolution" / "candidate.py"),
    }
    _write_json(args.output, payload)
    print(json.dumps(payload["candidate"], ensure_ascii=False))


async def command_evaluate(args: argparse.Namespace) -> None:
    output_root = Path(args.output_dir).resolve()
    manager = _manager(args)
    artifact_id = _artifact_id(args.track)
    baseline = manager.resolve(args.baseline_registry, artifact_id)
    candidate = manager.resolve(args.candidate_registry, artifact_id)
    development_prerequisite = _validate_development_decision(
        args,
        manager=manager,
        baseline=baseline,
        candidate=candidate,
    )
    if args.results:
        evaluation = _read_json(args.results)
        _validate_imported_results(evaluation, baseline=baseline, candidate=candidate)
    elif args.track == "skill":
        evaluation = run_numeric_skill_experiment(
            registry_dir=args.registry_dir,
            baseline_registry=args.baseline_registry,
            candidate_registry=args.candidate_registry,
            dataset_path=args.claim_dataset,
            output_dir=output_root,
            split=args.split,
            experiment_id=args.experiment_id,
        )
    else:
        if not args.fixture_dir:
            raise ValueError("policy evaluation requires --fixture-dir or --results")
        evaluation = await run_policy_replay_experiment(
            config_path=args.config,
            registry_dir=args.registry_dir,
            baseline_registry=args.baseline_registry,
            candidate_registry=args.candidate_registry,
            dataset_manifest=args.dataset_manifest,
            fixture_dir=args.fixture_dir,
            output_dir=output_root,
            split=args.split,
            repeats=args.repeats,
            experiment_id=args.experiment_id,
        )
    # The raw result is intentionally kept in this local variable until the
    # PromotionGate has consumed it.  It is never written for a held-out run.
    evaluation = dict(evaluation)
    requested_split = canonical_split(args.split)
    result_split = canonical_split(evaluation.get("split", requested_split)) or requested_split
    if result_split != requested_split:
        raise ValueError(
            f"evaluation split mismatch: requested {requested_split!r}, result contains {result_split!r}"
        )
    evaluation_split = requested_split
    evaluation["split"] = evaluation_split
    evaluator_meta = evaluation.get("evaluator")
    if not isinstance(evaluator_meta, dict):
        evaluator_meta = {"id": "harness_evolution_cli_v1", "sha256": _evaluator_sha256()}
        evaluation["evaluator"] = evaluator_meta
    else:
        evaluator_meta = dict(evaluator_meta)
        evaluator_meta.setdefault("id", "harness_evolution_cli_v1")
        evaluator_meta.setdefault("sha256", _evaluator_sha256())
        evaluation["evaluator"] = evaluator_meta
    evaluation.setdefault("evaluator_sha256", str(evaluator_meta.get("sha256", "")))
    if args.track == "skill":
        evaluation.setdefault(
            "dataset",
            {
                "id": Path(args.claim_dataset).stem,
                "split": evaluation_split,
                "sha256": evaluation.get("dataset_sha256", ""),
            },
        )
    exp_id = str(evaluation.get("experiment_id", args.experiment_id or "evaluation"))
    exp_dir = output_root / exp_id
    decision = PromotionGate().decide(evaluation, track=args.track)
    public_evaluation = redact_held_out_evaluation(evaluation)
    evaluation_path = _write_json(exp_dir / "evaluation.json", public_evaluation)
    decision.update({
        "experiment_id": exp_id,
        "evaluation_path": str(evaluation_path),
        "baseline": {"artifact_id": baseline.artifact_id, "version": baseline.version, "sha256": baseline.sha256},
        "candidate": {"artifact_id": candidate.artifact_id, "version": candidate.version, "sha256": candidate.sha256},
        "manual_promotion_required": True,
        "rollback_to": baseline.version,
        "traceability": {
            "split": evaluation_split,
            "dataset_sha256": str(
                evaluation.get("dataset_sha256")
                or (evaluation.get("dataset", {}) or {}).get("sha256", "")
            ),
            "evaluator_sha256": str(evaluation.get("evaluator_sha256", "")),
            "evaluation_sha256": sha256_file(evaluation_path),
            "details_persisted": evaluation_split != "held_out",
            "development_prerequisite": development_prerequisite,
        },
    })
    _rehash_decision(decision)
    decision_path = exp_dir / "promotion_decision.json"
    save_promotion_decision(decision_path, decision)
    print(json.dumps({"eligible": decision["eligible"], "decision": str(decision_path)}, ensure_ascii=False))


def command_report(args: argparse.Namespace) -> None:
    exp_dir = Path(args.experiment).resolve()
    decision = _read_json(exp_dir / "promotion_decision.json")
    metrics = decision.get("metrics", {})
    failed = [item["gate"] for item in decision.get("gates", []) if not item.get("passed")]
    lines = [
        f"# Harness Evolution Report: {decision.get('experiment_id', exp_dir.name)}",
        "",
        f"- Track: `{decision.get('track')}`",
        f"- Eligible: `{decision.get('eligible')}`",
        f"- Baseline: `{decision.get('baseline', {}).get('version')}`",
        f"- Candidate: `{decision.get('candidate', {}).get('version')}`",
        f"- Failed gates: `{', '.join(failed) if failed else 'none'}`",
        f"- Dataset SHA-256: `{decision.get('traceability', {}).get('dataset_sha256', '')}`",
        f"- Evaluator SHA-256: `{decision.get('traceability', {}).get('evaluator_sha256', '')}`",
        f"- Held-out row details persisted: `{decision.get('traceability', {}).get('details_persisted', True)}`",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "> Promotion is never automatic. Run the explicit promote command only after reviewing this report.",
    ]
    target = exp_dir / "promotion_report.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(target))


def command_promote(args: argparse.Namespace) -> None:
    manager = _manager(args)
    decision = _read_json(args.decision)
    candidate_data = decision.get("candidate", {})
    artifact_id = str(candidate_data.get("artifact_id", ""))
    candidate = manager.resolve(args.candidate_registry, artifact_id)
    manager.promote(candidate, args.decision)
    print(json.dumps({"promoted": artifact_id, "version": candidate.version}, ensure_ascii=False))


def command_rollback(args: argparse.Namespace) -> None:
    manager = _manager(args)
    current = manager.resolve("production", args.artifact)
    target_path = Path(current.path).parent / f"{args.version}.yaml"
    if not target_path.is_file():
        raise FileNotFoundError(target_path)
    data = yaml.safe_load(target_path.read_text(encoding="utf-8"))
    target = VersionRef(
        artifact_id=args.artifact,
        version=str(data.get("version")),
        sha256=sha256_file(target_path),
        parent=data.get("parent"),
        path=str(target_path.resolve()),
    )
    manager.rollback(args.artifact, target)
    print(json.dumps({"rolled_back": args.artifact, "version": target.version}, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline Policy/Skill Harness evolution V1")
    parser.add_argument("--registry-dir", default=str(DEFAULT_ROOT / "registries"))
    sub = parser.add_subparsers(dest="command", required=True)

    mine = sub.add_parser("mine")
    mine.add_argument("--input", required=True)
    mine.add_argument("--output", default="outputs/harness_evolution/experiences.jsonl")

    propose = sub.add_parser("propose")
    propose.add_argument("--track", choices=["policy", "skill"], required=True)
    propose.add_argument("--base-registry", default="production")
    propose.add_argument("--experiences", default="outputs/harness_evolution/experiences.jsonl")
    propose.add_argument("--select", type=int, default=0)
    propose.add_argument("--use-llm", action="store_true")
    propose.add_argument("--config", default=None)
    propose.add_argument("--output", default="outputs/harness_evolution/candidate_proposal.json")

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--track", choices=["policy", "skill"], required=True)
    evaluate.add_argument("--baseline-registry", default="production")
    evaluate.add_argument("--candidate-registry", default="candidate")
    evaluate.add_argument("--config", default=None)
    evaluate.add_argument("--dataset-manifest", default=str(DEFAULT_ROOT / "datasets" / "researchbench_v1.yaml"))
    evaluate.add_argument("--claim-dataset", default=str(DEFAULT_ROOT / "datasets" / "numeric_claims_v1.jsonl"))
    evaluate.add_argument("--fixture-dir")
    evaluate.add_argument("--results")
    evaluate.add_argument("--split", choices=["development", "held_out"], default="held_out")
    evaluate.add_argument(
        "--development-decision",
        help="eligible Development promotion_decision.json required for held_out evaluation",
    )
    evaluate.add_argument("--repeats", type=int, default=2)
    evaluate.add_argument("--experiment-id")
    evaluate.add_argument("--output-dir", default="outputs/harness_evolution")

    report = sub.add_parser("report")
    report.add_argument("--experiment", required=True)

    promote = sub.add_parser("promote")
    promote.add_argument("--decision", required=True)
    promote.add_argument("--candidate-registry", default="candidate")

    rollback = sub.add_parser("rollback")
    rollback.add_argument("--artifact", choices=["search_control_policy", "verify_numeric_claim_skill"], required=True)
    rollback.add_argument("--version", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "mine":
        command_mine(args)
    elif args.command == "propose":
        asyncio.run(command_propose(args))
    elif args.command == "evaluate":
        asyncio.run(command_evaluate(args))
    elif args.command == "report":
        command_report(args)
    elif args.command == "promote":
        command_promote(args)
    elif args.command == "rollback":
        command_rollback(args)


if __name__ == "__main__":
    main()
