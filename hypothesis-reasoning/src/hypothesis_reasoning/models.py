"""Versioned domain contracts for the hypothesis reasoning prototype."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ContractModel(BaseModel):
    """Base model with strict, immutable contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class VerificationStatus(StrEnum):
    SOURCE_LOCATED = "source_located"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"


class ObservationModality(StrEnum):
    TIME_SERIES = "time_series"
    SPECTRUM = "spectrum"
    IMAGE = "image"
    CATALOG = "catalog"
    OTHER = "other"


class GapType(StrEnum):
    ANOMALY = "anomaly"
    CONTRADICTION = "contradiction"
    KNOWLEDGE_GAP = "knowledge_gap"


class TopicDecisionStatus(StrEnum):
    SELECTED = "selected"
    ABSTAINED = "abstained"


class ReasoningType(StrEnum):
    INDUCTIVE = "inductive"
    DEDUCTIVE = "deductive"
    HYBRID = "hybrid"


class HypothesisStatus(StrEnum):
    PROPOSED = "proposed"
    NEEDS_REVISION = "needs_revision"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKING = "blocking"


class Decision(StrEnum):
    PASS = "pass"
    REVISE = "revise"
    REJECT = "reject"


class ResearchContext(ContractModel):
    context_id: str = Field(min_length=1)
    domain_scope: str = Field(min_length=1)
    topic_constraints: list[str]
    available_resources: dict[str, Any]
    max_topic_candidates: int = Field(ge=1)
    selection_policy_id: str = Field(min_length=1)


class EvidenceItem(ContractModel):
    evidence_id: str = Field(min_length=1)
    paper_title: str = Field(min_length=1)
    identifier: str = Field(min_length=1)
    location: dict[str, Any]
    source_text: str = Field(min_length=1)
    extracted_fact: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    verification_status: VerificationStatus


class ObservationRecord(ContractModel):
    observation_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    data_source: str = Field(min_length=1)
    modality: ObservationModality
    summary: str = Field(min_length=1)
    features: list[dict[str, Any]]
    anomaly_tags: list[str]
    method: str = Field(min_length=1)
    provenance: dict[str, Any]


class TopicCandidate(ContractModel):
    topic_id: str = Field(min_length=1)
    research_question: str = Field(min_length=1)
    gap_type: GapType
    motivation: str = Field(min_length=1)
    evidence_ids: list[str]
    observation_ids: list[str]
    expected_validation: list[str]
    risks: list[str]


class TopicDimensionScore(ContractModel):
    topic_id: str = Field(min_length=1)
    evidence_sufficiency: int = Field(ge=0, le=4)
    scientific_value_proxy: int = Field(ge=0, le=4)
    testability: int = Field(ge=0, le=4)
    data_availability: int = Field(ge=0, le=4)
    validation_cost: int = Field(ge=0, le=4)
    novelty_proxy: int = Field(ge=0, le=4)
    uncertainty: int = Field(ge=0, le=4)
    reason: str = Field(min_length=1)


class TopicDecision(ContractModel):
    selected_topic_id: str | None = None
    ranked_topic_ids: list[str]
    scores: list[TopicDimensionScore]
    selection_reason: str = Field(min_length=1)
    rejection_reasons: list[dict[str, str]]
    status: TopicDecisionStatus


class Prediction(ContractModel):
    description: str = Field(min_length=1)
    variable: str = Field(min_length=1)
    expected_direction: str = Field(min_length=1)
    falsifiable: bool


class ValidationPlan(ContractModel):
    method: str = Field(min_length=1)
    baselines: list[str]
    metrics: list[str]
    required_data: list[str]


class Hypothesis(ContractModel):
    hypothesis_id: str = Field(min_length=1)
    parent_id: str | None = None
    topic_id: str = Field(min_length=1)
    reasoning_type: ReasoningType
    statement: str = Field(min_length=1)
    mechanism: str = Field(min_length=1)
    evidence_ids: list[str]
    observation_ids: list[str]
    assumptions: list[str]
    scope_conditions: list[str]
    predictions: list[Prediction]
    falsification_criteria: list[str]
    alternative_explanations: list[str]
    validation_plan: ValidationPlan
    unsupported_claims: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    status: HypothesisStatus


class Critique(ContractModel):
    critique_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    issue_type: str = Field(min_length=1)
    severity: Severity
    target_field: str = Field(min_length=1)
    evidence_ids: list[str]
    description: str = Field(min_length=1)
    actionable_revision: str = Field(min_length=1)
    decision: Decision


class ScoreCard(ContractModel):
    evidence_fidelity: int = Field(ge=0, le=4)
    question_alignment: int = Field(ge=0, le=4)
    logical_coherence: int = Field(ge=0, le=4)
    mechanism_clarity: int = Field(ge=0, le=4)
    falsifiability: int = Field(ge=0, le=4)
    validation_feasibility: int = Field(ge=0, le=4)
    alternative_handling: int = Field(ge=0, le=4)
    hidden_outcome_alignment: int = Field(ge=0, le=4)
    reasons: dict[str, str]


class Usage(ContractModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    estimated_cost_cny: float = Field(ge=0.0)
    latency_ms: int = Field(ge=0)
    cache_hit: bool = False


class RunTrace(ContractModel):
    run_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    method_id: str = Field(min_length=1)
    random_seed: int
    requested_model: str = Field(min_length=1)
    returned_model: str = Field(min_length=1)
    parameters: dict[str, Any]
    started_at: datetime
    finished_at: datetime | None = None
    prompt_version: str = Field(min_length=1)
    code_version: str = Field(min_length=1)
    input_hash: str = Field(min_length=1)
    output_hash: str | None = None
    usage: Usage
    retries: int = Field(ge=0)
    stage: str = Field(min_length=1)
    status: str = Field(min_length=1)
    error_type: str | None = None
    adaptations: list[dict[str, str]] = Field(default_factory=list)
