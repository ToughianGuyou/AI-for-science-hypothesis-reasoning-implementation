"""Shared interfaces and run context for hypothesis generators."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Protocol

from hypothesis_reasoning.io import CaseBundle
from hypothesis_reasoning.llm.types import TraceAdaptation
from hypothesis_reasoning.models import Hypothesis, TopicCandidate


class AdaptationKind(StrEnum):
    HOST_DIRECTION_REMOVED = "host_direction_removed"
    REFERENCE_CONSTRAINTS_ADDED = "reference_constraints_added"
    JSON_CONTRACT_ADDED = "json_contract_added"


class AdaptationRecord(TraceAdaptation):
    kind: AdaptationKind


@dataclass(slots=True)
class RunContext:
    run_id: str
    case_id: str
    method_id: str
    requested_model: str
    random_seed: int
    temperature: float
    prompt_version: str
    budget_partition: str
    estimated_cost_cny: Decimal
    experiment_batch: str
    adaptations: list[AdaptationRecord] = field(default_factory=list)

    def trace_adaptations(self) -> list[dict[str, str]]:
        return [
            {"kind": str(record.kind), "detail": record.detail}
            for record in self.adaptations
        ]


class HypothesisGenerator(Protocol):
    """Uniform B0/B1 generator interface consumed by later pipeline tasks."""

    def generate(
        self, bundle: CaseBundle, topic: TopicCandidate, context: RunContext
    ) -> list[Hypothesis]: ...
