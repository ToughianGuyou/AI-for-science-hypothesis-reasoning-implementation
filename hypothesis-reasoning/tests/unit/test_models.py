from __future__ import annotations

import pytest
from pydantic import ValidationError

from hypothesis_reasoning.models import (
    Decision,
    EvidenceItem,
    Hypothesis,
    ResearchContext,
    VerificationStatus,
)
from tests.factories import context_payload, evidence_payload, hypothesis_payload


def test_hypothesis_rejects_out_of_range_confidence() -> None:
    payload = hypothesis_payload(confidence=1.1)
    with pytest.raises(ValidationError):
        Hypothesis.model_validate(payload)


def test_unverified_evidence_is_representable_but_not_silently_promoted() -> None:
    item = EvidenceItem.model_validate(evidence_payload(verification_status="unverified"))
    assert item.verification_status is VerificationStatus.UNVERIFIED


def test_research_context_does_not_accept_a_preselected_question() -> None:
    payload = context_payload(research_question="What mechanism caused the known signal?")
    with pytest.raises(ValidationError):
        ResearchContext.model_validate(payload)


def test_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        EvidenceItem.model_validate(evidence_payload(invented_fact="not allowed"))


def test_models_are_frozen_after_validation() -> None:
    item = EvidenceItem.model_validate(evidence_payload())
    with pytest.raises(ValidationError):
        item.paper_title = "changed"


def test_enum_values_are_stable_strings() -> None:
    assert VerificationStatus.SOURCE_LOCATED.value == "source_located"
    assert Decision.REVISE.value == "revise"
