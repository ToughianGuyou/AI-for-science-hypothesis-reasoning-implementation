"""Deterministic and model-backed review of generated hypotheses."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from importlib.resources import files
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from hypothesis_reasoning.errors import ContractValidationError, InputReferenceError
from hypothesis_reasoning.io import CaseBundle
from hypothesis_reasoning.llm.types import ChatMessage, LLMRequest, LLMResponse
from hypothesis_reasoning.models import (
    Critique,
    Decision,
    Hypothesis,
    HypothesisStatus,
    Prediction,
    Severity,
    ValidationPlan,
)
from hypothesis_reasoning.skill_adapters.base import RunContext

_REVIEW_DIMENSIONS = (
    "evidence_support",
    "logical_coherence",
    "mechanism_completeness",
    "falsifiability",
    "validation_feasibility",
    "scope_conditions",
)


class CompletionClient(Protocol):
    def complete_json(self, request: LLMRequest) -> LLMResponse: ...


class CritiqueBatch(BaseModel):
    """Strict response envelope for adversarial review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    critiques: list[Critique]


class EvidenceGate:
    """Return structured, deterministic findings without making model calls."""

    def __init__(
        self,
        *,
        expected_topic_id: str | None = None,
        require_grounding: bool = True,
    ) -> None:
        self._expected_topic_id = expected_topic_id
        self._require_grounding = require_grounding

    def check(self, hypothesis: Hypothesis, bundle: CaseBundle) -> list[Critique]:
        critiques: list[Critique] = []
        if (
            self._expected_topic_id is not None
            and hypothesis.topic_id != self._expected_topic_id
        ):
            critiques.append(
                _critique(
                    hypothesis,
                    issue_type="wrong_topic_reference",
                    severity=Severity.BLOCKING,
                    target_field="topic_id",
                    description=(
                        f"Hypothesis topic {hypothesis.topic_id} does not match the "
                        f"selected topic {self._expected_topic_id}."
                    ),
                    actionable_revision="Use the selected topic ID exactly.",
                )
            )

        known_evidence = {item.evidence_id for item in bundle.evidence}
        unknown_evidence = sorted(set(hypothesis.evidence_ids) - known_evidence)
        if unknown_evidence:
            critiques.append(
                _critique(
                    hypothesis,
                    issue_type="unknown_evidence_reference",
                    severity=Severity.BLOCKING,
                    target_field="evidence_ids",
                    evidence_ids=unknown_evidence,
                    description="Hypothesis references evidence IDs outside the case bundle.",
                    actionable_revision="Remove every unknown evidence ID.",
                )
            )

        known_observations = {item.observation_id for item in bundle.observations}
        unknown_observations = sorted(
            set(hypothesis.observation_ids) - known_observations
        )
        if unknown_observations:
            critiques.append(
                _critique(
                    hypothesis,
                    issue_type="unknown_observation_reference",
                    severity=Severity.BLOCKING,
                    target_field="observation_ids",
                    description=(
                        "Hypothesis references observation IDs outside the case bundle: "
                        + ", ".join(unknown_observations)
                    ),
                    actionable_revision="Remove every unknown observation ID.",
                )
            )

        if not self._require_grounding:
            return critiques

        missing_requirements: tuple[tuple[bool, str, str, str], ...] = (
            (
                not hypothesis.evidence_ids,
                "missing_evidence_linkage",
                "evidence_ids",
                "Link the claim to at least one supplied evidence item.",
            ),
            (
                not hypothesis.observation_ids,
                "missing_observation_linkage",
                "observation_ids",
                "Link the claim to at least one supplied observation.",
            ),
            (
                not hypothesis.predictions,
                "missing_predictions",
                "predictions",
                "Add at least one structured, falsifiable prediction.",
            ),
            (
                not hypothesis.falsification_criteria,
                "missing_falsification_criteria",
                "falsification_criteria",
                "State a concrete result that would count against the hypothesis.",
            ),
            (
                not hypothesis.alternative_explanations,
                "missing_alternative_explanations",
                "alternative_explanations",
                "Add at least one competing mechanism.",
            ),
            (
                not hypothesis.validation_plan.baselines,
                "missing_validation_baselines",
                "validation_plan.baselines",
                "Add at least one comparison baseline.",
            ),
            (
                not hypothesis.validation_plan.metrics,
                "missing_validation_metrics",
                "validation_plan.metrics",
                "Add at least one measurable validation metric.",
            ),
            (
                not hypothesis.validation_plan.required_data,
                "missing_validation_data",
                "validation_plan.required_data",
                "State the data required to run the validation.",
            ),
        )
        for missing, issue_type, target_field, action in missing_requirements:
            if missing:
                critiques.append(
                    _critique(
                        hypothesis,
                        issue_type=issue_type,
                        severity=Severity.HIGH,
                        target_field=target_field,
                        description=f"Required content is missing from {target_field}.",
                        actionable_revision=action,
                    )
                )

        if hypothesis.predictions and any(
            not prediction.falsifiable for prediction in hypothesis.predictions
        ):
            critiques.append(
                _critique(
                    hypothesis,
                    issue_type="non_falsifiable_prediction",
                    severity=Severity.HIGH,
                    target_field="predictions",
                    description="At least one prediction is not declared falsifiable.",
                    actionable_revision="Replace or make every prediction falsifiable.",
                )
            )

        if hypothesis.unsupported_claims:
            critiques.append(
                _critique(
                    hypothesis,
                    issue_type="unsupported_claims_declared",
                    severity=Severity.HIGH,
                    target_field="unsupported_claims",
                    description="The hypothesis declares claims not supported by the case.",
                    actionable_revision=(
                        "Remove, qualify, or explicitly isolate every unsupported claim."
                    ),
                )
            )
            if hypothesis.confidence > 0.5:
                critiques.append(
                    _critique(
                        hypothesis,
                        issue_type="unsupported_claim_confidence",
                        severity=Severity.HIGH,
                        target_field="confidence",
                        description=(
                            "Confidence exceeds 0.5 while unsupported claims remain."
                        ),
                        actionable_revision="Lower confidence to 0.5 or below.",
                    )
                )
            if hypothesis.status is not HypothesisStatus.NEEDS_REVISION:
                critiques.append(
                    _critique(
                        hypothesis,
                        issue_type="unsupported_claim_status",
                        severity=Severity.HIGH,
                        target_field="status",
                        description=(
                            "A hypothesis with unsupported claims must need revision."
                        ),
                        actionable_revision="Set status to needs_revision.",
                    )
                )
        return critiques


class AdversarialCritic:
    """Ask one blind reviewer for field-targeted findings, never a rewrite."""

    def __init__(self, llm: CompletionClient, *, prompt: str | None = None) -> None:
        self._llm = llm
        self._prompt = prompt if prompt is not None else _load_prompt()

    def review(
        self,
        hypothesis: Hypothesis,
        bundle: CaseBundle,
        context: RunContext,
        *,
        include_evidence_review: bool = True,
    ) -> list[Critique]:
        dimensions = tuple(
            dimension
            for dimension in _REVIEW_DIMENSIONS
            if include_evidence_review or dimension != "evidence_support"
        )
        anonymous = hypothesis.model_copy(
            update={"hypothesis_id": "candidate", "parent_id": None}
        )
        payload = {
            "context": bundle.context.model_dump(mode="json"),
            "evidence": [item.model_dump(mode="json") for item in bundle.evidence],
            "observations": [
                item.model_dump(mode="json") for item in bundle.observations
            ],
            "hypothesis": anonymous.model_dump(mode="json"),
            "review_dimensions": list(dimensions),
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        response = self._llm.complete_json(
            LLMRequest(
                requested_model=context.requested_model,
                messages=(
                    ChatMessage(role="system", content=self._prompt),
                    ChatMessage(role="user", content=serialized),
                ),
                seed=context.random_seed,
                temperature=context.temperature,
                prompt_version=context.prompt_version,
                input_hashes={
                    "critique_input": _sha256(serialized),
                    "prompt": _sha256(self._prompt),
                },
                response_model=CritiqueBatch,
                budget_partition=context.budget_partition,
                estimated_cost_cny=context.estimated_cost_cny,
                run_id=context.run_id,
                case_id=context.case_id,
                stage="adversarial_critique",
                experiment_batch=context.experiment_batch,
                adaptations=tuple(context.adaptations),
            )
        )
        batch = CritiqueBatch.model_validate(response.parsed_json)
        _validate_critic_findings(batch.critiques, bundle)
        return [
            critique.model_copy(update={"hypothesis_id": hypothesis.hypothesis_id})
            for critique in batch.critiques
        ]


def _validate_critic_findings(
    critiques: Sequence[Critique], bundle: CaseBundle
) -> None:
    critique_ids = [critique.critique_id for critique in critiques]
    if len(critique_ids) != len(set(critique_ids)):
        raise ContractValidationError("Adversarial critique returned duplicate critique IDs")
    known_evidence = {item.evidence_id for item in bundle.evidence}
    valid_fields = set(Hypothesis.model_fields)
    valid_fields.update(
        f"predictions.{field}" for field in Prediction.model_fields
    )
    valid_fields.update(
        f"validation_plan.{field}" for field in ValidationPlan.model_fields
    )
    for critique in critiques:
        if critique.hypothesis_id != "candidate":
            raise ContractValidationError(
                "Adversarial critique must target the anonymous candidate"
            )
        if critique.target_field not in valid_fields:
            raise ContractValidationError(
                f"Unknown critique target field: {critique.target_field}"
            )
        unknown = sorted(set(critique.evidence_ids) - known_evidence)
        if unknown:
            raise InputReferenceError(
                "Unknown evidence references in critique: " + ", ".join(unknown)
            )
        if critique.severity is Severity.BLOCKING and critique.decision is not Decision.REJECT:
            raise ContractValidationError(
                "Blocking adversarial critiques must use decision=reject"
            )


def _critique(
    hypothesis: Hypothesis,
    *,
    issue_type: str,
    severity: Severity,
    target_field: str,
    description: str,
    actionable_revision: str,
    evidence_ids: list[str] | None = None,
) -> Critique:
    identity = "|".join((hypothesis.hypothesis_id, issue_type, target_field, description))
    suffix = _sha256(identity)[:12]
    return Critique(
        critique_id=f"gate-{suffix}",
        hypothesis_id=hypothesis.hypothesis_id,
        issue_type=issue_type,
        severity=severity,
        target_field=target_field,
        evidence_ids=evidence_ids if evidence_ids is not None else [],
        description=description,
        actionable_revision=actionable_revision,
        decision=(Decision.REJECT if severity is Severity.BLOCKING else Decision.REVISE),
    )


def _load_prompt() -> str:
    resource = files("hypothesis_reasoning").joinpath(
        "prompts", "critique", "adversarial.md"
    )
    return resource.read_text(encoding="utf-8")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
