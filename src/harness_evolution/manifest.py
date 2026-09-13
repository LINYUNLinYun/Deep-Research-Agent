"""Reproducible run manifests for paired Harness evaluation."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .registry import VersionRef


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: str
    experiment_id: str
    pair_id: str
    query_id: str
    group: str
    registry: str
    artifacts: Mapping[str, Mapping[str, Any]]
    config_sha256: str
    git_commit: str
    git_dirty: bool
    executor: Mapping[str, Any]
    budgets: Mapping[str, Any]
    memory_session: str
    dataset: Mapping[str, Any]
    tool: Mapping[str, Any]
    evaluator: Mapping[str, Any]
    created_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        experiment_id: str,
        pair_id: str,
        query_id: str,
        group: str,
        registry: str | Path,
        artifacts: Mapping[str, VersionRef],
        config: Mapping[str, Any],
        executor: Mapping[str, Any],
        budgets: Mapping[str, Any],
        memory_session: str,
        dataset: Mapping[str, Any],
        tool: Mapping[str, Any],
        evaluator: Mapping[str, Any],
        created_at: str,
        repo_root: str | Path = ".",
        metadata: Mapping[str, Any] | None = None,
    ) -> "RunManifest":
        root = Path(repo_root).resolve()
        artifact_records = {
            artifact_id: {
                "version": ref.version,
                "sha256": ref.sha256,
                "parent": ref.parent,
                "path": str(Path(ref.path).resolve()) if ref.path else "",
            }
            for artifact_id, ref in sorted(artifacts.items())
        }
        commit, dirty = cls._git_state(root)
        dataset_record = dict(dataset)
        dataset_record.setdefault("sha256", stable_hash(dataset_record))
        evaluator_record = dict(evaluator)
        evaluator_record.setdefault("sha256", stable_hash(evaluator_record))
        tool_record = dict(tool)
        if tool_record.get("fixture_path") and not tool_record.get("fixture_sha256"):
            fixture = Path(str(tool_record["fixture_path"])).resolve()
            if fixture.is_file():
                tool_record["fixture_path"] = str(fixture)
                tool_record["fixture_sha256"] = hashlib.sha256(fixture.read_bytes()).hexdigest()
        return cls(
            run_id=run_id,
            experiment_id=experiment_id,
            pair_id=pair_id,
            query_id=query_id,
            group=group,
            registry=str(Path(registry).resolve()),
            artifacts=artifact_records,
            config_sha256=stable_hash(config),
            git_commit=commit,
            git_dirty=dirty,
            executor=dict(executor),
            budgets=dict(budgets),
            memory_session=memory_session,
            dataset=dataset_record,
            tool=tool_record,
            evaluator=evaluator_record,
            created_at=created_at,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def _git_state(root: Path) -> tuple[str, bool]:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
            ).stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
            ).stdout
            return commit, bool(status.strip())
        except (OSError, subprocess.CalledProcessError):
            return "unknown", True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> str:
        """Write once and return the SHA-256 of the exact manifest bytes."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        try:
            with destination.open("x", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
        except FileExistsError as exc:
            raise FileExistsError(f"run manifest is immutable: {destination}") from exc
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def read(cls, path: str | Path) -> "RunManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
