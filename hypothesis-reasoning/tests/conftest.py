"""Shared fixtures reserved for the implementation plan's later tasks."""

from __future__ import annotations

import pytest

from hypothesis_reasoning.io import CaseBundle
from hypothesis_reasoning.models import EvidenceItem, ObservationRecord, ResearchContext
from tests.factories import (
    context_payload,
    evidence_payload,
    hypothesis_payload,
    observation_payload,
    topic_candidate_payload,
    topic_decision_payload,
)


@pytest.fixture
def valid_bundle() -> CaseBundle:
    return CaseBundle(
        context=ResearchContext.model_validate(context_payload()),
        evidence=[EvidenceItem.model_validate(evidence_payload())],
        observations=[ObservationRecord.model_validate(observation_payload())],
    )


@pytest.fixture
def selected_topic() -> dict[str, object]:
    return topic_candidate_payload()


@pytest.fixture
def run_context() -> dict[str, object]:
    return {"run_id": "run-001", "case_id": "case-001", "method_id": "B0"}


@pytest.fixture
def fake_llm() -> object:
    return object()


@pytest.fixture
def qwen_client() -> object:
    return object()


@pytest.fixture
def sample_hypothesis() -> dict[str, object]:
    return hypothesis_payload()


@pytest.fixture
def sample_topic_decision() -> dict[str, object]:
    return topic_decision_payload()
