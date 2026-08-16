"""Fail-closed loading and serialization for case inputs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, TypeVar

from pydantic import BaseModel, ValidationError

from hypothesis_reasoning.errors import InputReferenceError, InputValidationError
from hypothesis_reasoning.models import (
    EvidenceItem,
    Hypothesis,
    ObservationRecord,
    ResearchContext,
    VerificationStatus,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class CaseBundle:
    """Validated context and source records available to a reasoning run."""

    context: ResearchContext
    evidence: list[EvidenceItem]
    observations: list[ObservationRecord]


def load_case(case_dir: Path, *, allow_unverified: bool = False) -> CaseBundle:
    """Load the complete, validated case at *case_dir* without dropping any input."""

    context = _validate_model(_read_json(case_dir / "research_context.json"), ResearchContext)
    evidence = _validate_list(
        _read_json(case_dir / "evidence_items.json"), EvidenceItem, "evidence_items.json"
    )
    observations = _validate_list(
        _read_json(case_dir / "observation_records.json"),
        ObservationRecord,
        "observation_records.json",
    )
    _require_nonempty(evidence, "evidence_items.json")
    _require_nonempty(observations, "observation_records.json")
    _require_unique_ids(evidence, "evidence_id", "evidence_items.json")
    _require_unique_ids(observations, "observation_id", "observation_records.json")
    if not allow_unverified:
        unverified_ids = [
            item.evidence_id
            for item in evidence
            if item.verification_status is VerificationStatus.UNVERIFIED
        ]
        if unverified_ids:
            raise InputReferenceError(
                "Unverified evidence is not allowed: " + ", ".join(sorted(unverified_ids))
            )
    return CaseBundle(context=context, evidence=evidence, observations=observations)


def validate_hypothesis_references(hypothesis: Hypothesis, bundle: CaseBundle) -> None:
    """Reject a hypothesis that refers to evidence or observations outside *bundle*."""

    _require_known_ids(
        hypothesis.evidence_ids,
        {item.evidence_id for item in bundle.evidence},
        "evidence",
    )
    _require_known_ids(
        hypothesis.observation_ids,
        {record.observation_id for record in bundle.observations},
        "observation",
    )


def write_json_atomic(path: Path, value: BaseModel | dict[object, object]) -> None:
    """Serialize *value* and atomically replace *path* only after serialization succeeds."""

    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary_path.write_text(serialized + "\n", encoding="utf-8")
        temporary_path.replace(path)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_non_finite_json)
    except (OSError, ValueError) as error:
        raise InputValidationError(f"Invalid JSON input at {path}: {error}") from error


def _reject_non_finite_json(constant: str) -> NoReturn:
    raise ValueError(f"Non-finite JSON constant: {constant}")


def _validate_model(value: object, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate(value)
    except ValidationError as error:
        raise InputValidationError(str(error)) from error


def _validate_list(value: object, model: type[ModelT], filename: str) -> list[ModelT]:
    if not isinstance(value, list):
        raise InputValidationError(f"Invalid {filename}: expected a JSON array")
    return [_validate_model(item, model) for item in value]


def _require_nonempty(items: Sequence[object], filename: str) -> None:
    if not items:
        raise InputValidationError(f"{filename} must not be empty")


def _require_unique_ids(items: Sequence[object], attribute: str, filename: str) -> None:
    identifiers = [getattr(item, attribute) for item in items]
    if len(identifiers) != len(set(identifiers)):
        raise InputValidationError(f"{filename} contains duplicate {attribute} values")


def _require_known_ids(references: list[str], known_ids: set[str], kind: str) -> None:
    unknown_ids = sorted(set(references) - known_ids)
    if unknown_ids:
        raise InputReferenceError(f"Unknown {kind} references: {', '.join(unknown_ids)}")
