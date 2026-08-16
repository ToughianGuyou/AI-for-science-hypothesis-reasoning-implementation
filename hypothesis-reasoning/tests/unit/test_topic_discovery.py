from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from hypothesis_reasoning.errors import TopicDiscoveryError
from hypothesis_reasoning.skill_adapters.base import RunContext
from hypothesis_reasoning.topic_discovery import TopicDiscoverer


class FakeCompletionClient:
    def __init__(self, *responses: dict[str, Any]) -> None:
        self._responses = list(responses)
        self.requests: list[Any] = []

    @property
    def last_request(self) -> Any:
        return self.requests[-1]

    def complete_json(self, request: Any) -> SimpleNamespace:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self._responses) - 1)
        return SimpleNamespace(parsed_json=self._responses[index])


def run_context() -> RunContext:
    return RunContext(
        run_id="run-topic-discovery",
        case_id="case-001",
        method_id="T1",
        requested_model="qwen-plus",
        random_seed=7,
        temperature=0.0,
        prompt_version="topic-discovery-v1",
        budget_partition="development",
        estimated_cost_cny=Decimal("0.10"),
        experiment_batch="topic-tests",
    )


def topic_candidates_json(*, count: int = 3) -> dict[str, list[dict[str, Any]]]:
    gap_types = ["anomaly", "contradiction", "knowledge_gap"]
    return {
        "candidates": [
            {
                "topic_id": f"topic-{index + 1:03d}",
                "research_question": (
                    f"Could mechanism {index + 1} explain the observed timing deviation?"
                ),
                "gap_type": gap_types[index % len(gap_types)],
                "motivation": "The supplied evidence leaves the deviation unresolved.",
                "evidence_ids": ["evidence-001"],
                "observation_ids": ["observation-001"],
                "expected_validation": ["Compare fitted timing models against the observation."],
                "risks": ["The deviation may reflect an unmodeled systematic."],
            }
            for index in range(count)
        ]
    }


def request_text(request: Any) -> str:
    return "\n".join(message.content for message in request.messages)


def test_discoverer_requires_three_grounded_candidates(valid_bundle: Any) -> None:
    fake_llm = FakeCompletionClient(topic_candidates_json(count=2))

    with pytest.raises(TopicDiscoveryError, match="exactly 3"):
        TopicDiscoverer(fake_llm).discover(valid_bundle, run_context())

    assert len(fake_llm.requests) == 2


def test_discoverer_receives_no_hidden_outcome(valid_bundle: Any) -> None:
    fake_llm = FakeCompletionClient(topic_candidates_json())
    bundle_with_hidden_data = SimpleNamespace(
        context=valid_bundle.context,
        evidence=valid_bundle.evidence,
        observations=valid_bundle.observations,
        published_outcome="The published paper claims a planet.",
    )

    candidates = TopicDiscoverer(fake_llm).discover(
        bundle_with_hidden_data,
        run_context(),
    )

    assert len(candidates) == 3
    assert "published_outcome" not in request_text(fake_llm.last_request)
    assert "published paper claims a planet" not in request_text(fake_llm.last_request)


def test_discoverer_repairs_invalid_count_exactly_once(valid_bundle: Any) -> None:
    fake_llm = FakeCompletionClient(
        topic_candidates_json(count=2),
        topic_candidates_json(count=3),
    )

    candidates = TopicDiscoverer(fake_llm).discover(valid_bundle, run_context())

    assert [candidate.topic_id for candidate in candidates] == [
        "topic-001",
        "topic-002",
        "topic-003",
    ]
    assert len(fake_llm.requests) == 2
    assert fake_llm.requests[0].stage == "topic_discovery"
    assert fake_llm.requests[1].stage == "topic_discovery_repair"


def test_discoverer_rejects_invalid_references_after_one_repair(valid_bundle: Any) -> None:
    invalid = topic_candidates_json()
    invalid["candidates"][0]["evidence_ids"] = ["missing"]
    fake_llm = FakeCompletionClient(invalid)

    with pytest.raises(TopicDiscoveryError, match="Unknown evidence references"):
        TopicDiscoverer(fake_llm).discover(valid_bundle, run_context())

    assert len(fake_llm.requests) == 2
