from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from hypothesis_reasoning.errors import (
    InputReferenceError,
    RefinementLimitError,
    RefinementValidationError,
)
from hypothesis_reasoning.models import Critique, Hypothesis
from hypothesis_reasoning.skill_adapters.base import RunContext
from tests.factories import hypothesis_payload


class FakeCompletionClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.requests: list[Any] = []

    def complete_json(self, request: Any) -> SimpleNamespace:
        self.requests.append(request)
        return SimpleNamespace(parsed_json=self.response)


def original(**overrides: Any) -> Hypothesis:
    return Hypothesis.model_validate(hypothesis_payload(**overrides))


def critiques(hypothesis_id: str = "hypothesis-001") -> list[Critique]:
    return [
        Critique(
            critique_id="critique-001",
            hypothesis_id=hypothesis_id,
            issue_type="logical_gap",
            severity="high",
            target_field="mechanism",
            evidence_ids=["evidence-001"],
            description="The mechanism skips a causal step.",
            actionable_revision="Add the missing causal step.",
            decision="revise",
        )
    ]


def hypothesis_json(**overrides: Any) -> dict[str, Any]:
    payload = hypothesis_payload(
        hypothesis_id="model-supplied-id",
        parent_id="hypothesis-001",
    )
    payload.update(overrides)
    return Hypothesis.model_validate(payload).model_dump(mode="json")


def run_context() -> RunContext:
    return RunContext(
        run_id="run-task7-refiner",
        case_id="case-001",
        method_id="B3",
        requested_model="qwen-max",
        random_seed=37,
        temperature=0.2,
        prompt_version="refinement-v1",
        budget_partition="development",
        estimated_cost_cny=Decimal("0.10"),
        experiment_batch="task7-tests",
    )


def test_refiner_cannot_add_new_sources(valid_bundle: Any) -> None:
    from hypothesis_reasoning.refinement import Refiner

    fake_llm = FakeCompletionClient(
        hypothesis_json(evidence_ids=["new-paper"])
    )

    with pytest.raises(InputReferenceError, match="new-paper"):
        Refiner(fake_llm).revise(
            original(),
            critiques(),
            valid_bundle,
            run_context(),
        )


def test_refiner_requires_parent_to_equal_original_id(valid_bundle: Any) -> None:
    from hypothesis_reasoning.refinement import Refiner

    fake_llm = FakeCompletionClient(hypothesis_json(parent_id="different-parent"))

    with pytest.raises(RefinementValidationError, match="parent_id"):
        Refiner(fake_llm).revise(
            original(),
            critiques(),
            valid_bundle,
            run_context(),
        )


def test_refiner_generates_new_id_and_allows_only_one_lineage_call(
    valid_bundle: Any,
) -> None:
    from hypothesis_reasoning.refinement import Refiner

    fake_llm = FakeCompletionClient(hypothesis_json())
    refiner = Refiner(fake_llm)

    revised = refiner.revise(
        original(),
        critiques(),
        valid_bundle,
        run_context(),
    )

    assert revised.parent_id == "hypothesis-001"
    assert revised.hypothesis_id.startswith("hypothesis-001-r1-")
    assert revised.hypothesis_id != "model-supplied-id"
    with pytest.raises(RefinementLimitError):
        refiner.revise(
            revised,
            critiques(revised.hypothesis_id),
            valid_bundle,
            run_context(),
        )
    assert len(fake_llm.requests) == 1


def test_failed_refinement_consumes_the_only_lineage_call(valid_bundle: Any) -> None:
    from hypothesis_reasoning.refinement import Refiner

    fake_llm = FakeCompletionClient(hypothesis_json(evidence_ids=["new-paper"]))
    refiner = Refiner(fake_llm)

    with pytest.raises(InputReferenceError):
        refiner.revise(
            original(),
            critiques(),
            valid_bundle,
            run_context(),
        )
    with pytest.raises(RefinementLimitError):
        refiner.revise(
            original(),
            critiques(),
            valid_bundle,
            run_context(),
        )
    assert len(fake_llm.requests) == 1


def test_same_lineage_can_be_revised_once_in_each_independent_run(
    valid_bundle: Any,
) -> None:
    from hypothesis_reasoning.refinement import Refiner

    fake_llm = FakeCompletionClient(hypothesis_json())
    refiner = Refiner(fake_llm)
    first_context = run_context()
    second_context = replace(first_context, run_id="run-task7-refiner-repeat")

    first = refiner.revise(
        original(),
        critiques(),
        valid_bundle,
        first_context,
    )
    second = refiner.revise(
        original(),
        critiques(),
        valid_bundle,
        second_context,
    )

    assert first.hypothesis_id == second.hypothesis_id
    assert len(fake_llm.requests) == 2


def test_refiner_reruns_blocking_topic_gate(valid_bundle: Any) -> None:
    from hypothesis_reasoning.refinement import Refiner

    fake_llm = FakeCompletionClient(hypothesis_json(topic_id="topic-other"))

    with pytest.raises(RefinementValidationError, match="blocking evidence gate"):
        Refiner(fake_llm).revise(
            original(),
            critiques(),
            valid_bundle,
            run_context(),
        )


def test_refiner_rejects_more_than_one_configured_round() -> None:
    from hypothesis_reasoning.refinement import Refiner

    with pytest.raises(ValueError, match="exactly one"):
        Refiner(FakeCompletionClient(hypothesis_json()), max_refinement_rounds=2)
