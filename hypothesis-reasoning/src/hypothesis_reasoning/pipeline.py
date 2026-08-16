"""Explicit B0-B3 and A1-A3 hypothesis pipeline orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from hypothesis_reasoning.critique import EvidenceGate
from hypothesis_reasoning.errors import InputReferenceError, PipelineConfigurationError
from hypothesis_reasoning.io import CaseBundle
from hypothesis_reasoning.models import (
    Critique,
    Hypothesis,
    Severity,
    TopicCandidate,
    TopicDecision,
    TopicDecisionStatus,
)
from hypothesis_reasoning.pool import HypothesisPool
from hypothesis_reasoning.skill_adapters.base import HypothesisGenerator, RunContext


class ExperimentMode(StrEnum):
    B0 = "B0"
    B1 = "B1"
    B2 = "B2"
    B3 = "B3"
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"


class PipelineStatus(StrEnum):
    COMPLETED = "completed"
    ABSTAINED = "abstained"
    FAILED = "failed"


class PipelineResult(BaseModel):
    """Auditable final state for one explicit experiment mode."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: ExperimentMode
    status: PipelineStatus
    topic_id: str | None
    hypotheses: list[Hypothesis]
    critiques: list[Critique]
    degraded: bool = False
    failed_stages: list[str]
    revision_count: int = Field(ge=0)
    reason: str = Field(min_length=1)


class HypothesisCritic(Protocol):
    def review(
        self,
        hypothesis: Hypothesis,
        bundle: CaseBundle,
        context: RunContext,
        *,
        include_evidence_review: bool = True,
    ) -> list[Critique]: ...


class HypothesisRefiner(Protocol):
    def revise(
        self,
        hypothesis: Hypothesis,
        critiques: list[Critique],
        bundle: CaseBundle,
        context: RunContext,
    ) -> Hypothesis: ...


@dataclass(frozen=True, slots=True)
class _ModeSpec:
    generator_routes: tuple[str, ...]
    require_grounding: bool
    adversarial_critique: bool
    evidence_review: bool
    refinement: bool
    a1_refiner: bool = False


_MODE_SPECS = {
    ExperimentMode.B0: _ModeSpec(("baseline",), False, False, False, False),
    ExperimentMode.B1: _ModeSpec(("skill",), False, False, False, False),
    ExperimentMode.B2: _ModeSpec(
        ("inductive", "deductive"), False, False, False, False
    ),
    ExperimentMode.B3: _ModeSpec(
        ("inductive", "deductive"), True, True, True, True
    ),
    ExperimentMode.A1: _ModeSpec(
        ("a1_inductive", "a1_deductive"), False, True, False, True, True
    ),
    ExperimentMode.A2: _ModeSpec(
        ("inductive", "deductive"), True, False, False, True
    ),
    ExperimentMode.A3: _ModeSpec(
        ("inductive", "deductive"), True, True, True, False
    ),
}


class Pipeline:
    """Run a frozen mode mapping; callers cannot assemble modes with flags."""

    def __init__(
        self,
        *,
        topics: Sequence[TopicCandidate],
        baseline_generator: HypothesisGenerator | None = None,
        skill_generator: HypothesisGenerator | None = None,
        inductive_generator: HypothesisGenerator | None = None,
        deductive_generator: HypothesisGenerator | None = None,
        a1_inductive_generator: HypothesisGenerator | None = None,
        a1_deductive_generator: HypothesisGenerator | None = None,
        critic: HypothesisCritic | None = None,
        refiner: HypothesisRefiner | None = None,
        a1_refiner: HypothesisRefiner | None = None,
        pool: HypothesisPool | None = None,
        max_refinement_rounds: int = 1,
    ) -> None:
        if max_refinement_rounds != 1:
            raise ValueError("Task 7 permits exactly one refinement round")
        topic_ids = [topic.topic_id for topic in topics]
        if len(topic_ids) != len(set(topic_ids)):
            raise PipelineConfigurationError("Frozen topic catalog contains duplicate IDs")
        self._topics = {topic.topic_id: topic for topic in topics}
        self._generators: dict[str, HypothesisGenerator | None] = {
            "baseline": baseline_generator,
            "skill": skill_generator,
            "inductive": inductive_generator,
            "deductive": deductive_generator,
            "a1_inductive": a1_inductive_generator,
            "a1_deductive": a1_deductive_generator,
        }
        self._critic = critic
        self._refiner = refiner
        self._a1_refiner = a1_refiner
        self._pool = pool if pool is not None else HypothesisPool()

    def run(
        self,
        bundle: CaseBundle,
        topic_decision: TopicDecision,
        mode: ExperimentMode,
        context: RunContext,
    ) -> PipelineResult:
        if topic_decision.status is TopicDecisionStatus.ABSTAINED:
            return PipelineResult(
                mode=mode,
                status=PipelineStatus.ABSTAINED,
                topic_id=None,
                hypotheses=[],
                critiques=[],
                degraded=False,
                failed_stages=[],
                revision_count=0,
                reason="Topic selection abstained; hypothesis generation was skipped.",
            )

        topic = self._resolve_topic(topic_decision)
        spec = _MODE_SPECS[mode]
        candidates, failed_stages = self._generate(
            spec.generator_routes,
            bundle,
            topic,
            context,
        )
        degraded = len(spec.generator_routes) > 1 and bool(failed_stages)
        if not candidates:
            return PipelineResult(
                mode=mode,
                status=PipelineStatus.FAILED,
                topic_id=topic.topic_id,
                hypotheses=[],
                critiques=[],
                degraded=degraded,
                failed_stages=failed_stages,
                revision_count=0,
                reason="Every generator required by the explicit mode failed.",
            )

        gate = EvidenceGate(
            expected_topic_id=topic.topic_id,
            require_grounding=spec.require_grounding,
        )
        final_hypotheses: list[Hypothesis] = []
        all_critiques: list[Critique] = []
        revision_count = 0
        for hypothesis in self._pool.consolidate(candidates):
            hypothesis_critiques = gate.check(hypothesis, bundle)
            all_critiques.extend(hypothesis_critiques)
            if _has_blocking(hypothesis_critiques):
                continue

            if spec.adversarial_critique:
                critic = self._required_critic(mode)
                adversarial = critic.review(
                    hypothesis,
                    bundle,
                    context,
                    include_evidence_review=spec.evidence_review,
                )
                hypothesis_critiques.extend(adversarial)
                all_critiques.extend(adversarial)
                if _has_blocking(adversarial):
                    continue

            if spec.refinement and _has_high(hypothesis_critiques):
                refiner = self._required_refiner(mode, use_a1=spec.a1_refiner)
                revised = refiner.revise(
                    hypothesis,
                    hypothesis_critiques,
                    bundle,
                    context,
                )
                revision_count += 1
                post_revision = gate.check(revised, bundle)
                all_critiques.extend(post_revision)
                if _has_blocking(post_revision):
                    continue
                hypothesis = revised
            final_hypotheses.append(hypothesis)

        status = (
            PipelineStatus.COMPLETED if final_hypotheses else PipelineStatus.FAILED
        )
        reason = (
            "Pipeline completed with at least one surviving hypothesis."
            if final_hypotheses
            else "No generated hypothesis survived the required safety gates."
        )
        return PipelineResult(
            mode=mode,
            status=status,
            topic_id=topic.topic_id,
            hypotheses=final_hypotheses,
            critiques=all_critiques,
            degraded=degraded,
            failed_stages=failed_stages,
            revision_count=revision_count,
            reason=reason,
        )

    def _resolve_topic(self, decision: TopicDecision) -> TopicCandidate:
        selected_id = decision.selected_topic_id
        if selected_id is None:
            raise InputReferenceError("A selected topic decision has no selected_topic_id")
        topic = self._topics.get(selected_id)
        if topic is None:
            raise InputReferenceError(
                f"Selected topic {selected_id} is absent from the frozen topic catalog"
            )
        return topic

    def _generate(
        self,
        routes: tuple[str, ...],
        bundle: CaseBundle,
        topic: TopicCandidate,
        context: RunContext,
    ) -> tuple[list[Hypothesis], list[str]]:
        candidates: list[Hypothesis] = []
        failures: list[str] = []
        for route in routes:
            generator = self._generators.get(route)
            if generator is None:
                raise PipelineConfigurationError(
                    f"Experiment mode requires a {route} generator"
                )
            try:
                candidates.extend(generator.generate(bundle, topic, context))
            except Exception:
                failures.append(f"generation_{route}")
        return candidates, failures

    def _required_critic(self, mode: ExperimentMode) -> HypothesisCritic:
        if self._critic is None:
            raise PipelineConfigurationError(f"Experiment mode {mode} requires a critic")
        return self._critic

    def _required_refiner(
        self, mode: ExperimentMode, *, use_a1: bool
    ) -> HypothesisRefiner:
        refiner = self._a1_refiner if use_a1 else self._refiner
        if refiner is None:
            raise PipelineConfigurationError(f"Experiment mode {mode} requires a refiner")
        return refiner


def _has_blocking(critiques: Sequence[Critique]) -> bool:
    return any(critique.severity is Severity.BLOCKING for critique in critiques)


def _has_high(critiques: Sequence[Critique]) -> bool:
    return any(critique.severity is Severity.HIGH for critique in critiques)
