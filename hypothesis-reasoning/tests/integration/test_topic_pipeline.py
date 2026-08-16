from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hypothesis_reasoning.errors import TopicSelectionError
from hypothesis_reasoning.models import TopicDecisionStatus
from hypothesis_reasoning.skill_adapters.base import RunContext
from hypothesis_reasoning.topic_discovery import TopicDiscoverer
from hypothesis_reasoning.topic_selection import (
    OneShotTopicSelector,
    StructuredTopicPipeline,
    TopicSelector,
)


class FakeCompletionClient:
    def __init__(self, *responses: dict[str, Any]) -> None:
        self._responses = list(responses)
        self.requests: list[Any] = []

    def complete_json(self, request: Any) -> SimpleNamespace:
        self.requests.append(request)
        return SimpleNamespace(parsed_json=self._responses[len(self.requests) - 1])


class SpyGenerator:
    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self.topic_ids: list[str] = []

    def generate(self, bundle: Any, topic: Any, context: Any) -> list[Any]:
        assert (self._output_dir / "topic_candidates.json").is_file()
        assert (self._output_dir / "topic_decision.json").is_file()
        self.topic_ids.append(topic.topic_id)
        return []


def run_context(method_id: str) -> RunContext:
    return RunContext(
        run_id=f"run-{method_id.lower()}",
        case_id="case-001",
        method_id=method_id,
        requested_model="qwen-plus",
        random_seed=13,
        temperature=0.0,
        prompt_version=f"topic-{method_id.lower()}-v1",
        budget_partition="development",
        estimated_cost_cny=Decimal("0.10"),
        experiment_batch="topic-pipeline-tests",
    )


def candidates_payload() -> list[dict[str, Any]]:
    gap_types = ["anomaly", "contradiction", "knowledge_gap"]
    return [
        {
            "topic_id": f"topic-{index:03d}",
            "research_question": f"Could mechanism {index} explain the timing deviation?",
            "gap_type": gap_types[index - 1],
            "motivation": "The visible evidence leaves the deviation unresolved.",
            "evidence_ids": ["evidence-001"],
            "observation_ids": ["observation-001"],
            "expected_validation": ["Compare fitted models against the visible observation."],
            "risks": ["The signal may contain an unmodeled systematic."],
        }
        for index in range(1, 4)
    ]


def score_payload(topic_id: str, *, evidence: int = 3) -> dict[str, Any]:
    return {
        "topic_id": topic_id,
        "evidence_sufficiency": evidence,
        "scientific_value_proxy": 3,
        "testability": 3,
        "data_availability": 3,
        "validation_cost": 3,
        "novelty_proxy": 3,
        "uncertainty": 3,
        "reason": f"Grounded scoring reason for {topic_id}.",
    }


def selected_decision_payload() -> dict[str, Any]:
    return {
        "selected_topic_id": "topic-001",
        "ranked_topic_ids": ["topic-001", "topic-002", "topic-003"],
        "scores": [score_payload(f"topic-{index:03d}") for index in range(1, 4)],
        "selection_reason": "topic-001 is the stable top-ranked grounded topic.",
        "rejection_reasons": [
            {"topic_id": "topic-002", "reason": "Stable tie-break placed it second."},
            {"topic_id": "topic-003", "reason": "Stable tie-break placed it third."},
        ],
        "status": "selected",
    }


def decision_payload(
    scores: list[dict[str, Any]],
    *,
    ranked_topic_ids: list[str],
    selected_topic_id: str | None,
    status: str,
) -> dict[str, Any]:
    candidate_ids = {f"topic-{index:03d}" for index in range(1, 4)}
    return {
        "selected_topic_id": selected_topic_id,
        "ranked_topic_ids": ranked_topic_ids,
        "scores": scores,
        "selection_reason": "One-shot decision under invariant testing.",
        "rejection_reasons": [
            {"topic_id": topic_id, "reason": "Not selected by the one-shot decision."}
            for topic_id in sorted(candidate_ids - {selected_topic_id})
        ],
        "status": status,
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_t0_uses_one_call_and_writes_standard_topic_artifacts(
    valid_bundle: Any,
    tmp_path: Path,
) -> None:
    fake_llm = FakeCompletionClient(
        {"candidates": candidates_payload(), "decision": selected_decision_payload()}
    )
    pipeline = OneShotTopicSelector(fake_llm, output_dir=tmp_path)

    candidates, decision = pipeline.run(valid_bundle, run_context("T0"))

    assert len(candidates) == 3
    assert decision.selected_topic_id == "topic-001"
    assert [request.stage for request in fake_llm.requests] == ["topic_one_shot"]
    written_candidates = read_json(tmp_path / "topic_candidates.json")["candidates"]
    assert [item["topic_id"] for item in written_candidates] == [
        "topic-001",
        "topic-002",
        "topic-003",
    ]
    assert read_json(tmp_path / "topic_decision.json")["status"] == "selected"


def test_t0_rejects_selected_topic_that_fails_a_hard_threshold(
    valid_bundle: Any,
    tmp_path: Path,
) -> None:
    weak_selected = score_payload("topic-001", evidence=1)
    weak_selected.update(
        scientific_value_proxy=4,
        testability=4,
        data_availability=4,
        validation_cost=4,
        novelty_proxy=4,
        uncertainty=4,
    )
    scores = [
        weak_selected,
        score_payload("topic-002", evidence=0),
        score_payload("topic-003", evidence=0),
    ]
    fake_llm = FakeCompletionClient(
        {
            "candidates": candidates_payload(),
            "decision": decision_payload(
                scores,
                ranked_topic_ids=["topic-001", "topic-002", "topic-003"],
                selected_topic_id="topic-001",
                status="selected",
            ),
        }
    )

    with pytest.raises(TopicSelectionError, match="hard threshold"):
        OneShotTopicSelector(fake_llm, output_dir=tmp_path).run(
            valid_bundle,
            run_context("T0"),
        )


def test_t0_rejects_ranking_inconsistent_with_weighted_scores(
    valid_bundle: Any,
    tmp_path: Path,
) -> None:
    strongest = score_payload("topic-002")
    strongest.update(
        evidence_sufficiency=4,
        scientific_value_proxy=4,
        testability=4,
        data_availability=4,
        validation_cost=4,
        novelty_proxy=4,
        uncertainty=4,
    )
    scores = [score_payload("topic-001"), strongest, score_payload("topic-003")]
    fake_llm = FakeCompletionClient(
        {
            "candidates": candidates_payload(),
            "decision": decision_payload(
                scores,
                ranked_topic_ids=["topic-001", "topic-002", "topic-003"],
                selected_topic_id="topic-001",
                status="selected",
            ),
        }
    )

    with pytest.raises(TopicSelectionError, match="weighted ranking"):
        OneShotTopicSelector(fake_llm, output_dir=tmp_path).run(
            valid_bundle,
            run_context("T0"),
        )


def test_t0_rejects_abstention_when_a_candidate_passes_all_thresholds(
    valid_bundle: Any,
    tmp_path: Path,
) -> None:
    scores = [score_payload(f"topic-{index:03d}") for index in range(1, 4)]
    fake_llm = FakeCompletionClient(
        {
            "candidates": candidates_payload(),
            "decision": decision_payload(
                scores,
                ranked_topic_ids=["topic-001", "topic-002", "topic-003"],
                selected_topic_id=None,
                status="abstained",
            ),
        }
    )

    with pytest.raises(TopicSelectionError, match="cannot abstain"):
        OneShotTopicSelector(fake_llm, output_dir=tmp_path).run(
            valid_bundle,
            run_context("T0"),
        )


@pytest.mark.parametrize(
    ("rejection_reasons", "message"),
    [
        (
            [
                {"topic_id": "topic-002", "reason": ""},
                {"topic_id": "topic-003", "reason": "Ranked third."},
            ],
            "nonblank reason",
        ),
        (
            [
                {"topic_id": "topic-002", "reason": "Ranked second."},
                {"topic_id": "topic-003", "reason": "Ranked third."},
                {"topic_id": "topic-999", "reason": "Unknown topic."},
            ],
            "exactly the unselected",
        ),
        (
            [
                {"topic_id": "topic-001", "reason": "Selected topic."},
                {"topic_id": "topic-002", "reason": "Ranked second."},
                {"topic_id": "topic-003", "reason": "Ranked third."},
            ],
            "exactly the unselected",
        ),
        (
            [
                {"topic_id": "topic-002", "reason": "Ranked second."},
                {"topic_id": "topic-002", "reason": "Duplicate reason."},
                {"topic_id": "topic-003", "reason": "Ranked third."},
            ],
            "exactly the unselected",
        ),
    ],
)
def test_t0_rejects_invalid_rejection_reason_sets(
    valid_bundle: Any,
    tmp_path: Path,
    rejection_reasons: list[dict[str, str]],
    message: str,
) -> None:
    decision = selected_decision_payload()
    decision["rejection_reasons"] = rejection_reasons
    fake_llm = FakeCompletionClient(
        {"candidates": candidates_payload(), "decision": decision}
    )

    with pytest.raises(TopicSelectionError, match=message):
        OneShotTopicSelector(fake_llm, output_dir=tmp_path).run(
            valid_bundle,
            run_context("T0"),
        )


def test_t1_traces_discovery_and_selection_as_separate_stages(
    valid_bundle: Any,
    tmp_path: Path,
) -> None:
    fake_llm = FakeCompletionClient(
        {"candidates": candidates_payload()},
        {"scores": [score_payload(f"topic-{index:03d}") for index in range(1, 4)]},
    )
    generator = SpyGenerator(tmp_path)
    pipeline = StructuredTopicPipeline(
        TopicDiscoverer(fake_llm),
        TopicSelector(fake_llm),
        output_dir=tmp_path,
        downstream=generator,
    )

    candidates, decision = pipeline.run(valid_bundle, run_context("T1"))

    assert len(candidates) == 3
    assert decision.status is TopicDecisionStatus.SELECTED
    assert [request.stage for request in fake_llm.requests] == [
        "topic_discovery",
        "topic_selection",
    ]
    assert (tmp_path / "topic_candidates.json").is_file()
    assert (tmp_path / "topic_decision.json").is_file()
    assert generator.topic_ids == ["topic-001"]


def test_t1_abstention_writes_successful_result_and_bypasses_hypothesis_generation(
    valid_bundle: Any,
    tmp_path: Path,
) -> None:
    fake_llm = FakeCompletionClient(
        {"candidates": candidates_payload()},
        {
            "scores": [
                score_payload("topic-001", evidence=1),
                score_payload("topic-002", evidence=1),
                score_payload("topic-003", evidence=1),
            ]
        },
    )
    generator = SpyGenerator(tmp_path)
    pipeline = StructuredTopicPipeline(
        TopicDiscoverer(fake_llm),
        TopicSelector(fake_llm),
        output_dir=tmp_path,
        downstream=generator,
    )

    _, decision = pipeline.run(valid_bundle, run_context("T1"))

    assert decision.status is TopicDecisionStatus.ABSTAINED
    assert read_json(tmp_path / "topic_decision.json")["status"] == "abstained"
    assert [request.stage for request in fake_llm.requests] == [
        "topic_discovery",
        "topic_selection",
    ]
    assert generator.topic_ids == []
