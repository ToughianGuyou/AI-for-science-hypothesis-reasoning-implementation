from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from hypothesis_reasoning.errors import ContractValidationError, InputReferenceError
from hypothesis_reasoning.models import Hypothesis, Severity
from hypothesis_reasoning.skill_adapters.base import RunContext
from tests.factories import hypothesis_payload


def hypothesis(**overrides: Any) -> Hypothesis:
    return Hypothesis.model_validate(hypothesis_payload(**overrides))


def run_context() -> RunContext:
    return RunContext(
        run_id="run-task7-critic",
        case_id="case-001",
        method_id="B3",
        requested_model="qwen-max",
        random_seed=31,
        temperature=0.2,
        prompt_version="critique-v1",
        budget_partition="development",
        estimated_cost_cny=Decimal("0.10"),
        experiment_batch="task7-tests",
    )


class FakeCompletionClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.requests: list[Any] = []

    def complete_json(self, request: Any) -> SimpleNamespace:
        self.requests.append(request)
        return SimpleNamespace(parsed_json=self.response)


def critique_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "critique_id": "critic-001",
        "hypothesis_id": "candidate",
        "issue_type": "logical_gap",
        "severity": "high",
        "target_field": "mechanism",
        "evidence_ids": ["evidence-001"],
        "description": "The mechanism skips an intermediate causal step.",
        "actionable_revision": "State the missing causal step explicitly.",
        "decision": "revise",
    }
    payload.update(overrides)
    return payload


def test_unknown_evidence_is_blocking(valid_bundle: Any) -> None:
    from hypothesis_reasoning.critique import EvidenceGate

    critiques = EvidenceGate().check(
        hypothesis(evidence_ids=["missing"]),
        valid_bundle,
    )

    assert any(
        critique.severity is Severity.BLOCKING
        and critique.target_field == "evidence_ids"
        and critique.evidence_ids == ["missing"]
        for critique in critiques
    )


def test_unknown_observation_and_wrong_topic_are_blocking(valid_bundle: Any) -> None:
    from hypothesis_reasoning.critique import EvidenceGate

    critiques = EvidenceGate(expected_topic_id="topic-001").check(
        hypothesis(topic_id="topic-other", observation_ids=["missing-observation"]),
        valid_bundle,
    )

    blocking_fields = {
        critique.target_field
        for critique in critiques
        if critique.severity is Severity.BLOCKING
    }
    assert blocking_fields == {"topic_id", "observation_ids"}


def test_missing_repairable_content_is_high_severity(valid_bundle: Any) -> None:
    from hypothesis_reasoning.critique import EvidenceGate

    critiques = EvidenceGate().check(
        hypothesis(
            evidence_ids=[],
            observation_ids=[],
            predictions=[],
            falsification_criteria=[],
            alternative_explanations=[],
            validation_plan={
                "method": "Inspect the observation.",
                "baselines": [],
                "metrics": [],
                "required_data": [],
            },
        ),
        valid_bundle,
    )

    high_fields = {
        critique.target_field
        for critique in critiques
        if critique.severity is Severity.HIGH
    }
    assert high_fields == {
        "evidence_ids",
        "observation_ids",
        "predictions",
        "falsification_criteria",
        "alternative_explanations",
        "validation_plan.baselines",
        "validation_plan.metrics",
        "validation_plan.required_data",
    }


def test_unsupported_claim_requires_low_confidence_and_revision_status(
    valid_bundle: Any,
) -> None:
    from hypothesis_reasoning.critique import EvidenceGate

    critiques = EvidenceGate().check(
        hypothesis(
            unsupported_claims=["A second body has a specific unobserved mass."],
            confidence=0.9,
            status="accepted",
        ),
        valid_bundle,
    )

    assert {
        critique.issue_type
        for critique in critiques
        if critique.severity is Severity.HIGH
    } == {
        "unsupported_claims_declared",
        "unsupported_claim_confidence",
        "unsupported_claim_status",
    }


def test_relaxed_grounding_retains_unknown_id_safety_only(valid_bundle: Any) -> None:
    from hypothesis_reasoning.critique import EvidenceGate

    relaxed = EvidenceGate(require_grounding=False)

    assert relaxed.check(
        hypothesis(
            evidence_ids=[],
            observation_ids=[],
            predictions=[],
            falsification_criteria=[],
            unsupported_claims=["An intentionally unsupported claim."],
            confidence=0.95,
        ),
        valid_bundle,
    ) == []
    assert any(
        critique.severity is Severity.BLOCKING
        for critique in relaxed.check(
            hypothesis(evidence_ids=["invented-evidence"]),
            valid_bundle,
        )
    )


def test_adversarial_critic_receives_one_anonymized_hypothesis(
    valid_bundle: Any,
) -> None:
    from hypothesis_reasoning.critique import AdversarialCritic

    fake_llm = FakeCompletionClient({"critiques": [critique_payload()]})
    original = hypothesis(hypothesis_id="secret-hypothesis-id")

    critiques = AdversarialCritic(fake_llm, prompt="Review one candidate.").review(
        original,
        valid_bundle,
        run_context(),
    )

    request = fake_llm.requests[0]
    user_payload = json.loads(request.messages[1].content)
    assert "secret-hypothesis-id" not in request.messages[1].content
    assert "B3" not in request.messages[1].content
    assert user_payload["hypothesis"]["hypothesis_id"] == "candidate"
    assert critiques[0].hypothesis_id == "secret-hypothesis-id"
    assert critiques[0].target_field == "mechanism"


def test_adversarial_critic_rejects_unknown_evidence_reference(
    valid_bundle: Any,
) -> None:
    from hypothesis_reasoning.critique import AdversarialCritic

    fake_llm = FakeCompletionClient(
        {"critiques": [critique_payload(evidence_ids=["invented-evidence"])]}
    )

    with pytest.raises(InputReferenceError, match="invented-evidence"):
        AdversarialCritic(fake_llm, prompt="Review one candidate.").review(
            hypothesis(),
            valid_bundle,
            run_context(),
        )


def test_adversarial_critic_rejects_nonexistent_target_field(valid_bundle: Any) -> None:
    from hypothesis_reasoning.critique import AdversarialCritic

    fake_llm = FakeCompletionClient(
        {"critiques": [critique_payload(target_field="imaginary_field")]}
    )

    with pytest.raises(ContractValidationError, match="imaginary_field"):
        AdversarialCritic(fake_llm, prompt="Review one candidate.").review(
            hypothesis(),
            valid_bundle,
            run_context(),
        )


def test_adversarial_critic_rejects_nonexistent_nested_target_field(
    valid_bundle: Any,
) -> None:
    from hypothesis_reasoning.critique import AdversarialCritic

    fake_llm = FakeCompletionClient(
        {
            "critiques": [
                critique_payload(target_field="validation_plan.imaginary_field")
            ]
        }
    )

    with pytest.raises(ContractValidationError, match="imaginary_field"):
        AdversarialCritic(fake_llm, prompt="Review one candidate.").review(
            hypothesis(),
            valid_bundle,
            run_context(),
        )


def test_adversarial_critic_can_disable_only_evidence_review(valid_bundle: Any) -> None:
    from hypothesis_reasoning.critique import AdversarialCritic

    fake_llm = FakeCompletionClient({"critiques": [critique_payload()]})

    AdversarialCritic(fake_llm, prompt="Review one candidate.").review(
        hypothesis(),
        valid_bundle,
        run_context(),
        include_evidence_review=False,
    )

    payload = json.loads(fake_llm.requests[0].messages[1].content)
    assert "evidence_support" not in payload["review_dimensions"]
    assert "logical_coherence" in payload["review_dimensions"]
