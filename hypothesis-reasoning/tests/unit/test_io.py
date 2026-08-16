from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hypothesis_reasoning.errors import InputReferenceError, InputValidationError
from hypothesis_reasoning.io import (
    CaseBundle,
    load_case,
    validate_hypothesis_references,
    write_json_atomic,
)
from hypothesis_reasoning.models import Hypothesis
from tests.factories import (
    evidence_payload,
    hypothesis_payload,
    observation_payload,
    write_case_fixture,
)


def test_load_case_parses_the_valid_fixture() -> None:
    case_dir = Path(__file__).parents[1] / "fixtures" / "valid_case"

    bundle = load_case(case_dir)

    assert bundle.context.context_id == "case-001"
    assert [item.evidence_id for item in bundle.evidence] == ["evidence-001"]
    assert [record.observation_id for record in bundle.observations] == ["observation-001"]


def test_load_case_rejects_duplicate_evidence_ids(tmp_path: Path) -> None:
    case_dir = write_case_fixture(
        tmp_path,
        evidence=[evidence_payload(evidence_id="e1"), evidence_payload(evidence_id="e1")],
    )

    with pytest.raises(InputValidationError):
        load_case(case_dir)


def test_load_case_rejects_empty_observation_lists(tmp_path: Path) -> None:
    case_dir = write_case_fixture(tmp_path, observations=[])

    with pytest.raises(InputValidationError):
        load_case(case_dir)


def test_load_case_rejects_empty_evidence_lists(tmp_path: Path) -> None:
    case_dir = write_case_fixture(tmp_path, evidence=[])

    with pytest.raises(InputValidationError):
        load_case(case_dir)


def test_load_case_rejects_duplicate_observation_ids(tmp_path: Path) -> None:
    case_dir = write_case_fixture(
        tmp_path,
        observations=[
            observation_payload(observation_id="o1"),
            observation_payload(observation_id="o1"),
        ],
    )

    with pytest.raises(InputValidationError):
        load_case(case_dir)


def test_load_case_rejects_missing_required_input_file(tmp_path: Path) -> None:
    case_dir = write_case_fixture(tmp_path)
    (case_dir / "evidence_items.json").unlink()

    with pytest.raises(InputValidationError):
        load_case(case_dir)


def test_load_case_rejects_invalid_json(tmp_path: Path) -> None:
    case_dir = write_case_fixture(tmp_path)
    (case_dir / "observation_records.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(InputValidationError):
        load_case(case_dir)


@pytest.mark.parametrize("non_finite_constant", ["NaN", "Infinity", "-Infinity"])
def test_load_case_rejects_non_finite_json_constants(
    tmp_path: Path,
    non_finite_constant: str,
) -> None:
    case_dir = write_case_fixture(tmp_path)
    serialized = json.dumps([evidence_payload(location={"page": 1})])
    (case_dir / "evidence_items.json").write_text(
        serialized.replace('"page": 1', f'"page": {non_finite_constant}'),
        encoding="utf-8",
    )

    with pytest.raises(InputValidationError):
        load_case(case_dir)


def test_load_case_wraps_invalid_pydantic_record_fields(tmp_path: Path) -> None:
    case_dir = write_case_fixture(tmp_path, evidence=[evidence_payload(paper_title="")])

    with pytest.raises(InputValidationError):
        load_case(case_dir)


def test_load_case_rejects_unverified_evidence_by_default(tmp_path: Path) -> None:
    case_dir = write_case_fixture(
        tmp_path,
        evidence=[evidence_payload(verification_status="unverified")],
    )

    with pytest.raises(InputReferenceError):
        load_case(case_dir)


def test_load_case_allows_unverified_evidence_only_when_explicit(tmp_path: Path) -> None:
    case_dir = write_case_fixture(
        tmp_path,
        evidence=[evidence_payload(verification_status="unverified")],
    )

    bundle = load_case(case_dir, allow_unverified=True)

    assert bundle.evidence[0].verification_status == "unverified"


def test_hypothesis_reference_check_rejects_unknown_evidence_id(valid_bundle: CaseBundle) -> None:
    hypothesis = Hypothesis.model_validate(hypothesis_payload(evidence_ids=["missing"]))

    with pytest.raises(InputReferenceError):
        validate_hypothesis_references(hypothesis, valid_bundle)


def test_hypothesis_reference_check_rejects_unknown_observation_id(
    valid_bundle: CaseBundle,
) -> None:
    hypothesis = Hypothesis.model_validate(hypothesis_payload(observation_ids=["missing"]))

    with pytest.raises(InputReferenceError):
        validate_hypothesis_references(hypothesis, valid_bundle)


def test_hypothesis_reference_check_fails_closed_for_unknown_observation_among_known_ids(
    valid_bundle: CaseBundle,
) -> None:
    hypothesis = Hypothesis.model_validate(
        hypothesis_payload(observation_ids=["observation-001", "missing"])
    )

    with pytest.raises(InputReferenceError):
        validate_hypothesis_references(hypothesis, valid_bundle)


def test_write_json_atomic_preserves_existing_file_when_value_cannot_be_serialized(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output.json"
    output.write_text('{"old": true}', encoding="utf-8")

    with pytest.raises(TypeError):
        write_json_atomic(output, {"not_json": object()})

    assert json.loads(output.read_text(encoding="utf-8")) == {"old": True}
    assert not output.with_suffix(".json.tmp").exists()


def test_write_json_atomic_replaces_target_with_json(tmp_path: Path) -> None:
    output = tmp_path / "output.json"
    output.write_text('{"old": true}', encoding="utf-8")

    write_json_atomic(output, {"new": "value"})

    assert json.loads(output.read_text(encoding="utf-8")) == {"new": "value"}


@pytest.mark.parametrize("non_finite_value", [float("nan"), float("inf"), float("-inf")])
def test_write_json_atomic_rejects_non_finite_values_without_replacing_target(
    tmp_path: Path,
    non_finite_value: float,
) -> None:
    output = tmp_path / "output.json"
    output.write_text('{"old": true}', encoding="utf-8")

    with pytest.raises(ValueError):
        write_json_atomic(output, {"value": non_finite_value})

    assert json.loads(output.read_text(encoding="utf-8")) == {"old": True}
    assert not output.with_suffix(".json.tmp").exists()


def test_export_schemas_writes_all_contract_schemas_deterministically(tmp_path: Path) -> None:
    project_dir = Path(__file__).parents[2]
    committed_schema_dir = project_dir / "schemas"
    expected_names = {
        "research-context.schema.json",
        "evidence-item.schema.json",
        "observation-record.schema.json",
        "topic-candidate.schema.json",
        "topic-decision.schema.json",
        "hypothesis.schema.json",
        "critique.schema.json",
        "score-card.schema.json",
        "run-trace.schema.json",
    }

    command = [sys.executable, "scripts/export_schemas.py"]
    first_schema_dir = tmp_path / "first"
    second_schema_dir = tmp_path / "second"
    first_run = subprocess.run(
        [*command, str(first_schema_dir)], cwd=project_dir, env=os.environ.copy(), check=False
    )
    assert first_run.returncode == 0
    first_contents = {
        path.name: path.read_text(encoding="utf-8")
        for path in first_schema_dir.glob("*.schema.json")
    }

    second_run = subprocess.run(
        [*command, str(second_schema_dir)], cwd=project_dir, env=os.environ.copy(), check=False
    )
    assert second_run.returncode == 0
    second_contents = {
        path.name: path.read_text(encoding="utf-8")
        for path in second_schema_dir.glob("*.schema.json")
    }
    committed_contents = {
        path.name: path.read_text(encoding="utf-8")
        for path in committed_schema_dir.glob("*.schema.json")
    }

    assert set(first_contents) == expected_names
    assert first_contents == committed_contents
    assert second_contents == committed_contents
    assert "context_id" in json.loads(first_contents["research-context.schema.json"])["properties"]
