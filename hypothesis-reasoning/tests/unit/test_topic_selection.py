from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from hypothesis_reasoning.errors import InputReferenceError, TopicSelectionError
from hypothesis_reasoning.models import TopicCandidate, TopicDecisionStatus
from hypothesis_reasoning.skill_adapters.base import RunContext
from hypothesis_reasoning.topic_selection import TopicSelector
from tests.factories import topic_candidate_payload


class FakeCompletionClient:
    def __init__(self, *responses: dict[str, Any]) -> None:
        self._responses = list(responses)
        self.requests: list[Any] = []

    @property
    def last_request(self) -> Any:
        return self.requests[-1]

    def complete_json(self, request: Any) -> SimpleNamespace:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("The scorer must not be called")
        index = min(len(self.requests) - 1, len(self._responses) - 1)
        return SimpleNamespace(parsed_json=self._responses[index])


def run_context() -> RunContext:
    return RunContext(
        run_id="run-topic-selection",
        case_id="case-001",
        method_id="T1",
        requested_model="qwen-plus",
        random_seed=11,
        temperature=0.0,
        prompt_version="topic-score-v1",
        budget_partition="development",
        estimated_cost_cny=Decimal("0.10"),
        experiment_batch="topic-tests",
    )


def three_topic_candidates() -> list[TopicCandidate]:
    return [
        TopicCandidate.model_validate(
            topic_candidate_payload(
                topic_id=f"topic-{index:03d}",
                research_question=f"Could mechanism {index} explain the timing deviation?",
            )
        )
        for index in range(1, 4)
    ]


def score(
    topic_id: str,
    *,
    evidence_sufficiency: int = 3,
    scientific_value_proxy: int = 3,
    testability: int = 3,
    data_availability: int = 3,
    validation_cost: int = 3,
    novelty_proxy: int = 3,
    uncertainty: int = 3,
) -> dict[str, Any]:
    return {
        "topic_id": topic_id,
        "evidence_sufficiency": evidence_sufficiency,
        "scientific_value_proxy": scientific_value_proxy,
        "testability": testability,
        "data_availability": data_availability,
        "validation_cost": validation_cost,
        "novelty_proxy": novelty_proxy,
        "uncertainty": uncertainty,
        "reason": f"Evidence-based scoring reason for {topic_id}.",
    }


def low_score_topic_cards_json() -> dict[str, list[dict[str, Any]]]:
    return {
        "scores": [
            score("topic-001", evidence_sufficiency=1),
            score("topic-002", testability=1),
            score("topic-003", data_availability=1),
        ]
    }


def test_selector_rejects_unknown_evidence(valid_bundle: Any) -> None:
    candidate = TopicCandidate.model_validate(
        topic_candidate_payload(evidence_ids=["missing"])
    )

    with pytest.raises(InputReferenceError, match="Unknown evidence references"):
        TopicSelector(fake_llm=None).validate_candidate(candidate, valid_bundle)


@pytest.mark.parametrize(
    "claim",
    [
        "The observed timing deviation is caused by an additional planet.",
        "The observations are inconsistent because the instrument is unstable.",
        "The supplied evidence does establish a unique mechanism.",
    ],
)
def test_selector_rejects_declarative_result_claims(
    valid_bundle: Any,
    claim: str,
) -> None:
    candidate = TopicCandidate.model_validate(
        topic_candidate_payload(research_question=claim)
    )

    with pytest.raises(TopicSelectionError, match="without explicit uncertainty"):
        TopicSelector(fake_llm=None).validate_candidate(candidate, valid_bundle)


def test_selector_abstains_when_every_candidate_fails_hard_thresholds(
    valid_bundle: Any,
) -> None:
    fake_llm = FakeCompletionClient(low_score_topic_cards_json())

    decision = TopicSelector(fake_llm).select(
        three_topic_candidates(),
        valid_bundle,
        run_context(),
    )

    assert decision.status is TopicDecisionStatus.ABSTAINED
    assert decision.selected_topic_id is None
    assert decision.ranked_topic_ids == ["topic-003", "topic-001", "topic-002"]
    assert {item["topic_id"] for item in decision.rejection_reasons} == {
        "topic-001",
        "topic-002",
        "topic-003",
    }


def test_selector_applies_deterministic_gates_before_llm_scoring(valid_bundle: Any) -> None:
    candidates = [
        TopicCandidate.model_validate(
            topic_candidate_payload(topic_id="topic-unknown", evidence_ids=["missing"])
        ),
        TopicCandidate.model_validate(
            topic_candidate_payload(topic_id="topic-empty", research_question=" ")
        ),
        TopicCandidate.model_validate(
            topic_candidate_payload(
                topic_id="topic-claim",
                research_question="An additional planet causes the timing deviation.",
            )
        ),
        TopicCandidate.model_validate(
            topic_candidate_payload(topic_id="topic-no-validation", expected_validation=[])
        ),
    ]
    fake_llm = FakeCompletionClient()

    decision = TopicSelector(fake_llm).select(candidates, valid_bundle, run_context())

    assert decision.status is TopicDecisionStatus.ABSTAINED
    assert decision.ranked_topic_ids == []
    assert decision.scores == []
    assert fake_llm.requests == []
    assert {item["topic_id"] for item in decision.rejection_reasons} == {
        "topic-unknown",
        "topic-empty",
        "topic-claim",
        "topic-no-validation",
    }


def test_selector_uses_weighted_score_and_preserves_dimension_scores(valid_bundle: Any) -> None:
    candidates = three_topic_candidates()[:2]
    fake_llm = FakeCompletionClient(
        {
            "scores": [
                score(
                    "topic-001",
                    evidence_sufficiency=4,
                    scientific_value_proxy=0,
                    testability=4,
                    data_availability=2,
                    validation_cost=0,
                    novelty_proxy=0,
                    uncertainty=0,
                ),
                score(
                    "topic-002",
                    evidence_sufficiency=2,
                    scientific_value_proxy=4,
                    testability=2,
                    data_availability=4,
                    validation_cost=4,
                    novelty_proxy=4,
                    uncertainty=4,
                ),
            ]
        }
    )

    decision = TopicSelector(fake_llm).select(candidates, valid_bundle, run_context())

    assert decision.status is TopicDecisionStatus.SELECTED
    assert decision.selected_topic_id == "topic-002"
    assert decision.ranked_topic_ids == ["topic-002", "topic-001"]
    assert decision.scores[0].scientific_value_proxy == 4
    assert decision.scores[0].validation_cost == 4
    assert decision.scores[0].uncertainty == 4


def test_selector_breaks_weighted_score_ties_by_topic_id(valid_bundle: Any) -> None:
    candidates = list(reversed(three_topic_candidates()))
    fake_llm = FakeCompletionClient(
        {"scores": [score(candidate.topic_id) for candidate in candidates]}
    )

    decision = TopicSelector(fake_llm).select(candidates, valid_bundle, run_context())

    assert decision.ranked_topic_ids == ["topic-001", "topic-002", "topic-003"]
    assert decision.selected_topic_id == "topic-001"


def test_selector_rejects_duplicate_candidate_ids_before_scoring(valid_bundle: Any) -> None:
    candidates = three_topic_candidates()
    candidates[1] = candidates[1].model_copy(update={"topic_id": "topic-001"})
    fake_llm = FakeCompletionClient()

    with pytest.raises(TopicSelectionError, match="Duplicate topic candidate IDs"):
        TopicSelector(fake_llm).select(candidates, valid_bundle, run_context())

    assert fake_llm.requests == []
