from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from hypothesis_reasoning.errors import ContractValidationError, InputReferenceError
from hypothesis_reasoning.models import ReasoningType, TopicCandidate
from hypothesis_reasoning.skill_adapters.base import RunContext
from tests.factories import hypothesis_payload


class FakeCompletionClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response
        self.requests: list[Any] = []

    @property
    def last_request(self) -> Any:
        return self.requests[-1]

    def complete_json(self, request: Any) -> SimpleNamespace:
        self.requests.append(request)
        return SimpleNamespace(parsed_json=self._response)


class RecordingPromptRegistry:
    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        self.requests: list[ReasoningType] = []

    def get(self, reasoning_type: ReasoningType) -> str:
        self.requests.append(reasoning_type)
        return self.prompt

    def __bool__(self) -> bool:
        return False


def run_context() -> RunContext:
    return RunContext(
        run_id="run-task6",
        case_id="case-001",
        method_id="B2",
        requested_model="qwen-max",
        random_seed=17,
        temperature=0.2,
        prompt_version="generation-v1",
        budget_partition="development",
        estimated_cost_cny=Decimal("0.10"),
        experiment_batch="task6-tests",
    )


def test_inductive_generator_receives_only_case_bundle_and_selected_topic(
    valid_bundle: Any,
    selected_topic: dict[str, object],
) -> None:
    from hypothesis_reasoning.generation import InductiveGenerator

    fake_llm = FakeCompletionClient(
        {
            "hypotheses": [
                hypothesis_payload(reasoning_type="inductive")
            ]
        }
    )
    bundle_with_hidden_channel_output = SimpleNamespace(
        context=valid_bundle.context,
        evidence=valid_bundle.evidence,
        observations=valid_bundle.observations,
        deductive_output="deductive output must remain invisible",
    )

    InductiveGenerator(fake_llm).generate(
        bundle_with_hidden_channel_output,
        TopicCandidate.model_validate(selected_topic),
        run_context(),
    )

    request_text = "\n".join(
        message.content for message in fake_llm.last_request.messages
    )
    payload = json.loads(fake_llm.last_request.messages[1].content)
    assert "deductive output" not in request_text
    assert payload["topic"] == selected_topic
    assert "topics" not in payload


def test_deductive_generator_receives_only_case_bundle_and_selected_topic(
    valid_bundle: Any,
    selected_topic: dict[str, object],
) -> None:
    from hypothesis_reasoning.generation import DeductiveGenerator

    fake_llm = FakeCompletionClient(
        {
            "hypotheses": [
                hypothesis_payload(reasoning_type="deductive")
            ]
        }
    )
    bundle_with_hidden_channel_output = SimpleNamespace(
        context=valid_bundle.context,
        evidence=valid_bundle.evidence,
        observations=valid_bundle.observations,
        inductive_output="inductive output must remain invisible",
    )

    DeductiveGenerator(fake_llm).generate(
        bundle_with_hidden_channel_output,
        TopicCandidate.model_validate(selected_topic),
        run_context(),
    )

    request_text = "\n".join(
        message.content for message in fake_llm.last_request.messages
    )
    payload = json.loads(fake_llm.last_request.messages[1].content)
    assert "inductive output" not in request_text
    assert payload["topic"] == selected_topic
    assert "topics" not in payload


def test_inductive_prompt_maps_multiple_observations_without_external_facts(
    valid_bundle: Any,
    selected_topic: dict[str, object],
) -> None:
    from hypothesis_reasoning.generation import InductiveGenerator

    fake_llm = FakeCompletionClient(
        {"hypotheses": [hypothesis_payload(reasoning_type="inductive")]}
    )

    InductiveGenerator(fake_llm).generate(
        valid_bundle,
        TopicCandidate.model_validate(selected_topic),
        run_context(),
    )

    prompt = fake_llm.last_request.messages[0].content.casefold()
    assert "multiple" in prompt and "observations" in prompt
    assert "external facts" in prompt
    for required_field in (
        "topic_id",
        "evidence_ids",
        "observation_ids",
        "unsupported_claims",
        "alternative_explanations",
        "falsification_criteria",
    ):
        assert required_field in prompt


def test_deductive_prompt_enumerates_premises_and_derives_predictions(
    valid_bundle: Any,
    selected_topic: dict[str, object],
) -> None:
    from hypothesis_reasoning.generation import DeductiveGenerator

    fake_llm = FakeCompletionClient(
        {"hypotheses": [hypothesis_payload(reasoning_type="deductive")]}
    )

    DeductiveGenerator(fake_llm).generate(
        valid_bundle,
        TopicCandidate.model_validate(selected_topic),
        run_context(),
    )

    prompt = fake_llm.last_request.messages[0].content.casefold()
    assert "premises" in prompt and "assumptions" in prompt
    assert "derive" in prompt and "predictions" in prompt
    assert "external facts" in prompt
    for required_field in (
        "topic_id",
        "evidence_ids",
        "observation_ids",
        "unsupported_claims",
        "alternative_explanations",
        "falsification_criteria",
    ):
        assert required_field in prompt


def test_generator_rejects_unknown_case_references(
    valid_bundle: Any,
    selected_topic: dict[str, object],
) -> None:
    from hypothesis_reasoning.generation import InductiveGenerator

    fake_llm = FakeCompletionClient(
        {
            "hypotheses": [
                hypothesis_payload(
                    reasoning_type="inductive",
                    evidence_ids=["invented-evidence"],
                )
            ]
        }
    )

    with pytest.raises(InputReferenceError, match="invented-evidence"):
        InductiveGenerator(fake_llm).generate(
            valid_bundle,
            TopicCandidate.model_validate(selected_topic),
            run_context(),
        )


def test_a1_generator_relaxes_claim_evidence_requirement_in_request(
    valid_bundle: Any,
    selected_topic: dict[str, object],
) -> None:
    from hypothesis_reasoning.generation import InductiveGenerator

    fake_llm = FakeCompletionClient(
        {
            "hypotheses": [
                hypothesis_payload(
                    reasoning_type="inductive",
                    evidence_ids=[],
                    unsupported_claims=["Intentionally unsupported for the A1 ablation."],
                )
            ]
        }
    )

    result = InductiveGenerator(fake_llm, require_grounding=False).generate(
        valid_bundle,
        TopicCandidate.model_validate(selected_topic),
        run_context(),
    )

    payload = json.loads(fake_llm.last_request.messages[1].content)
    assert payload["grounding_policy"] == {
        "claim_evidence_required": False,
        "known_identifiers_only": True,
    }
    assert result[0].evidence_ids == []


def test_a1_generator_still_rejects_invented_identifiers(
    valid_bundle: Any,
    selected_topic: dict[str, object],
) -> None:
    from hypothesis_reasoning.generation import DeductiveGenerator

    fake_llm = FakeCompletionClient(
        {
            "hypotheses": [
                hypothesis_payload(
                    reasoning_type="deductive",
                    evidence_ids=["invented-evidence"],
                )
            ]
        }
    )

    with pytest.raises(InputReferenceError, match="invented-evidence"):
        DeductiveGenerator(fake_llm, require_grounding=False).generate(
            valid_bundle,
            TopicCandidate.model_validate(selected_topic),
            run_context(),
        )


def test_generator_rejects_hypothesis_for_another_topic(
    valid_bundle: Any,
    selected_topic: dict[str, object],
) -> None:
    from hypothesis_reasoning.generation import DeductiveGenerator

    fake_llm = FakeCompletionClient(
        {
            "hypotheses": [
                hypothesis_payload(
                    reasoning_type="deductive",
                    topic_id="topic-outside-selection",
                )
            ]
        }
    )

    with pytest.raises(InputReferenceError, match="topic-outside-selection"):
        DeductiveGenerator(fake_llm).generate(
            valid_bundle,
            TopicCandidate.model_validate(selected_topic),
            run_context(),
        )


def test_generator_rejects_output_from_the_other_reasoning_channel(
    valid_bundle: Any,
    selected_topic: dict[str, object],
) -> None:
    from hypothesis_reasoning.generation import InductiveGenerator

    fake_llm = FakeCompletionClient(
        {"hypotheses": [hypothesis_payload(reasoning_type="deductive")]}
    )

    with pytest.raises(ContractValidationError, match="reasoning_type=deductive"):
        InductiveGenerator(fake_llm).generate(
            valid_bundle,
            TopicCandidate.model_validate(selected_topic),
            run_context(),
        )


def test_generator_uses_injected_prompt_registry(
    valid_bundle: Any,
    selected_topic: dict[str, object],
) -> None:
    from hypothesis_reasoning.generation import InductiveGenerator

    fake_llm = FakeCompletionClient(
        {"hypotheses": [hypothesis_payload(reasoning_type="inductive")]}
    )
    registry = RecordingPromptRegistry("INJECTED INDUCTIVE PROMPT")

    InductiveGenerator(fake_llm, registry).generate(
        valid_bundle,
        TopicCandidate.model_validate(selected_topic),
        run_context(),
    )

    assert registry.requests == [ReasoningType.INDUCTIVE]
    assert fake_llm.last_request.messages[0].content == "INJECTED INDUCTIVE PROMPT"
