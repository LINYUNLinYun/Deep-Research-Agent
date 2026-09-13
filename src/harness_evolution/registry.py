"""Immutable YAML artifact versions and atomic registry pointers."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .schemas import SchemaValidationError, validate_artifact


class RegistryError(RuntimeError):
    """A registry is invalid, stale or fails an integrity check."""


@dataclass(frozen=True, slots=True)
class VersionRef:
    artifact_id: str
    version: str
    sha256: str
    parent: str | None
    path: str = ""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class VersionRegistry:
    """Manage immutable artifacts referenced by small YAML registries.

    Paths stored in a registry are always interpreted relative to that
    registry file, not the process working directory.
    """

    def __init__(self, registry_dir: str | Path):
        self.registry_dir = Path(registry_dir).expanduser().resolve()
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def _registry_path(self, registry: str | Path) -> Path:
        candidate = Path(registry)
        if candidate.suffix not in {".yaml", ".yml"}:
            candidate = candidate.with_suffix(".yaml")
        if not candidate.is_absolute():
            candidate = self.registry_dir / candidate
        return candidate.resolve()

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise RegistryError(f"cannot read registry {path}: {exc}") from exc
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise RegistryError(f"unsupported registry schema: {path}")
        if not isinstance(data.get("artifacts"), dict):
            raise RegistryError(f"registry has no artifacts mapping: {path}")
        return data

    @staticmethod
    def _atomic_yaml(path: Path, data: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                yaml.safe_dump(dict(data), handle, allow_unicode=True, sort_keys=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def resolve(self, registry: str | Path, artifact_id: str) -> VersionRef:
        registry_path = self._registry_path(registry)
        data = self._load_yaml(registry_path)
        raw = data["artifacts"].get(artifact_id)
        if not isinstance(raw, Mapping):
            raise RegistryError(f"artifact {artifact_id!r} is not present in {registry_path}")
        required = {"version", "sha256", "path"}
        if not required <= set(raw):
            raise RegistryError(f"artifact {artifact_id!r} has an incomplete registry entry")
        artifact_path = (registry_path.parent / str(raw["path"])).resolve()
        if not artifact_path.is_file():
            raise RegistryError(f"artifact file does not exist: {artifact_path}")
        actual_hash = sha256_file(artifact_path)
        if actual_hash != raw["sha256"]:
            raise RegistryError(f"sha256 mismatch for {artifact_id}@{raw['version']}")
        try:
            with artifact_path.open("r", encoding="utf-8") as handle:
                artifact = yaml.safe_load(handle)
            validate_artifact(artifact)
        except (OSError, yaml.YAMLError, SchemaValidationError) as exc:
            raise RegistryError(f"invalid artifact {artifact_path}: {exc}") from exc
        if artifact.get("artifact_id") != artifact_id or artifact.get("version") != raw["version"]:
            raise RegistryError("registry identity does not match artifact contents")
        return VersionRef(
            artifact_id=artifact_id,
            version=str(raw["version"]),
            sha256=actual_hash,
            parent=raw.get("parent"),
            path=str(artifact_path),
        )

    def create_candidate(self, base_ref: VersionRef, yaml_patch: Mapping[str, Any]) -> VersionRef:
        """Create the next immutable version and point ``candidate`` at it."""

        if not base_ref.path:
            raise RegistryError("base_ref must have been resolved from a registry")
        base_path = Path(base_ref.path).resolve()
        if sha256_file(base_path) != base_ref.sha256:
            raise RegistryError("base artifact changed after it was resolved")
        try:
            with base_path.open("r", encoding="utf-8") as handle:
                base = yaml.safe_load(handle)
        except (OSError, yaml.YAMLError) as exc:
            raise RegistryError(f"cannot read base artifact: {exc}") from exc
        if not isinstance(yaml_patch, Mapping):
            raise RegistryError("yaml_patch must be a mapping")

        merged = self._merge(base, yaml_patch)
        versions = []
        for item in base_path.parent.glob("v*.yaml"):
            match = re.fullmatch(r"v(\d+)\.yaml", item.name)
            if match:
                versions.append(int(match.group(1)))
        width = max(4, len(base_ref.version.removeprefix("v")))
        version = f"v{max(versions, default=0) + 1:0{width}d}"
        merged["artifact_id"] = base_ref.artifact_id
        merged["version"] = version
        merged["parent"] = base_ref.version
        try:
            validate_artifact(merged)
        except SchemaValidationError as exc:
            raise RegistryError(f"candidate schema is invalid: {exc}") from exc

        target = base_path.parent / f"{version}.yaml"
        try:
            with target.open("x", encoding="utf-8") as handle:
                yaml.safe_dump(merged, handle, allow_unicode=True, sort_keys=False)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise RegistryError(f"immutable version already exists: {target}") from exc
        candidate = VersionRef(base_ref.artifact_id, version, sha256_file(target), base_ref.version, str(target))
        self._set_pointer(self._registry_path("candidate"), candidate)
        return candidate

    def mirror_registry(self, source: str | Path, target: str | Path = "candidate") -> None:
        """Atomically seed a candidate registry from a validated base registry.

        This prevents a rejected candidate from another evolution track from
        leaking into a new paired experiment. Immutable artifact files are not
        changed or removed; only the mutable target pointers are replaced.
        """

        source_path = self._registry_path(source)
        source_data = self._load_yaml(source_path)
        target_path = self._registry_path(target)
        artifacts: dict[str, Any] = {}
        for artifact_id in sorted(source_data["artifacts"]):
            ref = self.resolve(source_path, artifact_id)
            artifacts[artifact_id] = {
                "version": ref.version,
                "sha256": ref.sha256,
                "parent": ref.parent,
                "path": os.path.relpath(Path(ref.path).resolve(), target_path.parent),
            }
        self._atomic_yaml(target_path, {"schema_version": 1, "artifacts": artifacts})

    def promote(self, candidate_ref: VersionRef, decision_file: str | Path) -> None:
        decision_path = Path(decision_file).resolve()
        try:
            decision_bytes = decision_path.read_bytes()
            decision = json.loads(decision_bytes)
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError(f"cannot read promotion decision: {exc}") from exc
        expected_decision_hash = decision.get("decision_sha256")
        if not isinstance(expected_decision_hash, str) or not expected_decision_hash:
            raise RegistryError("promotion decision is missing decision_sha256")
        canonical = dict(decision)
        canonical.pop("decision_sha256", None)
        actual = hashlib.sha256(
            json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if actual != expected_decision_hash:
            raise RegistryError("promotion decision hash mismatch")
        if decision.get("eligible") is not True:
            raise RegistryError("promotion decision is not eligible")
        gates = decision.get("gates")
        if not isinstance(gates, list) or not gates or any(
            not isinstance(gate, Mapping) or gate.get("passed") is not True for gate in gates
        ):
            raise RegistryError("promotion decision does not contain an all-passed gate set")
        traceability = decision.get("traceability")
        if not isinstance(traceability, Mapping):
            raise RegistryError("promotion decision has no traceability record")
        split = str(traceability.get("split", "")).strip().lower().replace("-", "_")
        if split not in {"heldout", "held_out"}:
            raise RegistryError("only a held-out decision can be promoted")
        for key in ("dataset_sha256", "evaluator_sha256", "evaluation_sha256"):
            value = traceability.get(key)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise RegistryError(f"promotion decision has invalid {key}")
        evaluation_path_value = decision.get("evaluation_path")
        if not isinstance(evaluation_path_value, str) or not evaluation_path_value:
            raise RegistryError("promotion decision has no evaluation_path")
        evaluation_path = Path(evaluation_path_value).expanduser().resolve()
        if not evaluation_path.is_file() or sha256_file(evaluation_path) != traceability["evaluation_sha256"]:
            raise RegistryError("promotion evaluation artifact hash mismatch")
        prerequisite = traceability.get("development_prerequisite")
        if not isinstance(prerequisite, Mapping):
            raise RegistryError("held-out promotion has no Development prerequisite")
        self._verify_development_prerequisite(prerequisite)

        track = decision.get("track")
        expected_artifact_for_track = {
            "policy": "search_control_policy",
            "skill": "verify_numeric_claim_skill",
        }.get(track)
        if expected_artifact_for_track != candidate_ref.artifact_id:
            raise RegistryError("promotion decision track does not match candidate artifact")
        expected = decision.get("candidate", {})
        identity = (candidate_ref.artifact_id, candidate_ref.version, candidate_ref.sha256)
        if identity != (expected.get("artifact_id"), expected.get("version"), expected.get("sha256")):
            raise RegistryError("promotion decision refers to a different candidate")
        self._verify_ref(candidate_ref)
        production_path = self._registry_path("production")
        current = self.resolve(production_path, candidate_ref.artifact_id)
        if current.version != candidate_ref.parent:
            raise RegistryError(
                f"stale candidate: production is {current.version}, candidate parent is {candidate_ref.parent}"
            )
        expected_baseline = decision.get("baseline", {})
        if not isinstance(expected_baseline, Mapping) or not expected_baseline:
            raise RegistryError("promotion decision has no baseline identity")
        expected_identity = (
            expected_baseline.get("artifact_id"),
            expected_baseline.get("version"),
            expected_baseline.get("sha256"),
        )
        current_identity = (current.artifact_id, current.version, current.sha256)
        if expected_identity != current_identity:
            raise RegistryError("stale candidate: production baseline hash changed")
        self._set_pointer(production_path, candidate_ref)

    @staticmethod
    def _verify_development_prerequisite(prerequisite: Mapping[str, Any]) -> None:
        path_value = prerequisite.get("path")
        expected_hash = prerequisite.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected_hash, str):
            raise RegistryError("invalid Development prerequisite reference")
        path = Path(path_value).expanduser().resolve()
        try:
            decision = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError(f"cannot read Development prerequisite: {exc}") from exc
        supplied_hash = decision.get("decision_sha256")
        canonical = dict(decision)
        canonical.pop("decision_sha256", None)
        actual_hash = hashlib.sha256(
            json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        traceability = decision.get("traceability")
        split = (
            str(traceability.get("split", "")).strip().lower().replace("-", "_")
            if isinstance(traceability, Mapping)
            else ""
        )
        if (
            supplied_hash != expected_hash
            or actual_hash != expected_hash
            or decision.get("eligible") is not True
            or split != "development"
        ):
            raise RegistryError("Development prerequisite is invalid or changed")

    def rollback(self, artifact_id: str, target_ref: VersionRef) -> None:
        if target_ref.artifact_id != artifact_id:
            raise RegistryError("rollback target belongs to a different artifact")
        self._verify_ref(target_ref)
        production_path = self._registry_path("production")
        current = self.resolve(production_path, artifact_id)
        if current.version == target_ref.version and current.sha256 == target_ref.sha256:
            return
        self._set_pointer(production_path, target_ref)

    @staticmethod
    def _merge(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in patch.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = VersionRegistry._merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _verify_ref(reference: VersionRef) -> None:
        if not reference.path or not Path(reference.path).is_file():
            raise RegistryError("version reference has no valid artifact path")
        if sha256_file(reference.path) != reference.sha256:
            raise RegistryError(f"sha256 mismatch for {reference.artifact_id}@{reference.version}")

    def _set_pointer(self, registry_path: Path, reference: VersionRef) -> None:
        if registry_path.exists():
            data = self._load_yaml(registry_path)
        else:
            data = {"schema_version": 1, "artifacts": {}}
        relative = os.path.relpath(Path(reference.path).resolve(), registry_path.parent)
        data["artifacts"][reference.artifact_id] = {
            "version": reference.version,
            "sha256": reference.sha256,
            "parent": reference.parent,
            "path": relative,
        }
        self._atomic_yaml(registry_path, data)
