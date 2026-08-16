"""Deterministic gates and evidence-grounded topic selection."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from hypothesis_reasoning.errors import InputReferenceError, TopicSelectionError
from hypothesis_reasoning.io import CaseBundle, write_json_atomic
from hypothesis_reasoning.llm.types import ChatMessage, LLMRequest, LLMResponse
from hypothesis_reasoning.models import (
    TopicCandidate,
    TopicDecision,
    TopicDecisionStatus,
    TopicDimensionScore,
)
from hypothesis_reasoning.skill_adapters.base import HypothesisGenerator, RunContext
from hypothesis_reasoning.topic_discovery import (
    TopicCandidateBatch,
    TopicDiscoverer,
    validate_discovered_candidates,
)

_QUESTION_PREFIX = re.compile(
    r"^\s*(?:what|which|who|when|where|why|how|does|do|is|are)\b",
    re.IGNORECASE,
)
_UNCERTAINTY_MARKER = re.compile(
    r"(?:\?|\b(?:could|would|may|might|whether|can)\b|"
    r"[？]|是否|能否|可能|为何|为什么|如何|什么|哪[一种个])",
    re.IGNORECASE,
)
_WEIGHTS = {
    "evidence_sufficiency": 20,
    "scientific_value_proxy": 15,
    "testability": 20,
    "data_availability": 15,
    "validation_cost": 10,
    "novelty_proxy": 10,
    "uncertainty": 10,
}
_HARD_THRESHOLDS = {
    "evidence_sufficiency": 2,
    "testability": 2,
    "data_availability": 2,
}


class CompletionClient(Protocol):
    def complete_json(self, request: LLMRequest) -> LLMResponse: ...


class TopicScoreBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scores: list[TopicDimensionScore]


class OneShotTopicResult(BaseModel):
    """Shared T0 result envelope written as the same two topic artifacts as T1."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidates: list[TopicCandidate]
    decision: TopicDecision


class TopicSelector:
    """Gate, score, rank, and select topic candidates deterministically."""

    def __init__(
        self,
        fake_llm: CompletionClient | None,
        *,
        prompt: str | None = None,
    ) -> None:
        self._llm = fake_llm
        self._prompt = prompt if prompt is not None else _load_prompt("score.md")

    def validate_candidate(self, candidate: TopicCandidate, bundle: CaseBundle) -> None:
        _require_known_references(
            candidate.evidence_ids,
            {item.evidence_id for item in bundle.evidence},
            "evidence",
        )
        _require_known_references(
            candidate.observation_ids,
            {item.observation_id for item in bundle.observations},
            "observation",
        )
        if not candidate.research_question.strip():
            raise TopicSelectionError("Research question is empty")
        if not (
            _QUESTION_PREFIX.search(candidate.research_question)
            or _UNCERTAINTY_MARKER.search(candidate.research_question)
        ):
            raise TopicSelectionError(
                "Research question claims a result without explicit uncertainty"
            )
        if not any(path.strip() for path in candidate.expected_validation):
            raise TopicSelectionError("Candidate has no declared validation path")

    def select(
        self,
        candidates: Sequence[TopicCandidate],
        bundle: CaseBundle,
        context: RunContext,
    ) -> TopicDecision:
        seen_ids: set[str] = set()
        duplicate_ids: set[str] = set()
        for candidate in candidates:
            if candidate.topic_id in seen_ids:
                duplicate_ids.add(candidate.topic_id)
            seen_ids.add(candidate.topic_id)
        if duplicate_ids:
            raise TopicSelectionError(
                "Duplicate topic candidate IDs: " + ", ".join(sorted(duplicate_ids))
            )

        eligible: list[TopicCandidate] = []
        rejection_reasons: list[dict[str, str]] = []
        for candidate in candidates:
            try:
                self.validate_candidate(candidate, bundle)
            except (InputReferenceError, TopicSelectionError) as error:
                rejection_reasons.append(
                    {"topic_id": candidate.topic_id, "reason": str(error)}
                )
            else:
                eligible.append(candidate)

        if not eligible:
            return TopicDecision(
                selected_topic_id=None,
                ranked_topic_ids=[],
                scores=[],
                selection_reason="No candidate passed the deterministic topic gates.",
                rejection_reasons=rejection_reasons,
                status=TopicDecisionStatus.ABSTAINED,
            )
        if self._llm is None:
            raise TopicSelectionError("Topic scoring requires an injected completion client")

        response = self._llm.complete_json(
            _score_request(
                llm_prompt=self._prompt,
                candidates=eligible,
                bundle=bundle,
                context=context,
            )
        )
        scores = _validate_scores(response.parsed_json, eligible)
        ranked_scores = sorted(scores, key=lambda item: (-_weighted_score(item), item.topic_id))
        passing_ids = {
            item.topic_id for item in ranked_scores if not _hard_threshold_failures(item)
        }
        selected = next(
            (item for item in ranked_scores if item.topic_id in passing_ids),
            None,
        )

        for item in ranked_scores:
            failures = _hard_threshold_failures(item)
            if failures:
                rejection_reasons.append(
                    {
                        "topic_id": item.topic_id,
                        "reason": "Failed hard thresholds: " + ", ".join(failures),
                    }
                )
            elif selected is not None and item.topic_id != selected.topic_id:
                rejection_reasons.append(
                    {
                        "topic_id": item.topic_id,
                        "reason": (
                            f"Ranked below selected topic {selected.topic_id} by weighted score."
                        ),
                    }
                )

        ranked_ids = [item.topic_id for item in ranked_scores]
        if selected is None:
            decision = TopicDecision(
                selected_topic_id=None,
                ranked_topic_ids=ranked_ids,
                scores=ranked_scores,
                selection_reason="Every scored candidate failed at least one hard threshold.",
                rejection_reasons=rejection_reasons,
                status=TopicDecisionStatus.ABSTAINED,
            )
        else:
            decision = TopicDecision(
                selected_topic_id=selected.topic_id,
                ranked_topic_ids=ranked_ids,
                scores=ranked_scores,
                selection_reason=(
                    f"Selected {selected.topic_id} with weighted score "
                    f"{_weighted_score(selected) / 100:.2f}; all hard thresholds passed."
                ),
                rejection_reasons=rejection_reasons,
                status=TopicDecisionStatus.SELECTED,
            )
        _validate_decision_invariants(decision, eligible)
        return decision


class OneShotTopicSelector:
    def __init__(
        self,
        fake_llm: CompletionClient,
        *,
        output_dir: Path,
        prompt: str | None = None,
    ) -> None:
        self._llm = fake_llm
        self._output_dir = output_dir
        self._prompt = prompt if prompt is not None else _load_prompt("one_shot.md")

    def run(
        self, bundle: CaseBundle, context: RunContext
    ) -> tuple[list[TopicCandidate], TopicDecision]:
        payload = _visible_topic_payload(bundle)
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        response = self._llm.complete_json(
            LLMRequest(
                requested_model=context.requested_model,
                messages=(
                    ChatMessage(role="system", content=self._prompt),
                    ChatMessage(
                        role="user",
                        content="Return one-shot topic JSON for this input: " + serialized,
                    ),
                ),
                seed=context.random_seed,
                temperature=context.temperature,
                prompt_version=context.prompt_version,
                input_hashes={
                    "topic_input": hashlib.sha256(serialized.encode()).hexdigest(),
                    "prompt": hashlib.sha256(self._prompt.encode()).hexdigest(),
                },
                response_model=OneShotTopicResult,
                budget_partition=context.budget_partition,
                estimated_cost_cny=context.estimated_cost_cny,
                run_id=context.run_id,
                case_id=context.case_id,
                stage="topic_one_shot",
                experiment_batch=context.experiment_batch,
                adaptations=tuple(context.adaptations),
            )
        )
        try:
            result = OneShotTopicResult.model_validate(response.parsed_json)
        except ValidationError as error:
            raise TopicSelectionError(f"Invalid one-shot topic JSON: {error}") from error
        candidates = validate_discovered_candidates(
            TopicCandidateBatch(candidates=result.candidates).model_dump(mode="json"),
            bundle,
        )
        validator = TopicSelector(None)
        for candidate in candidates:
            validator.validate_candidate(candidate, bundle)
        _validate_decision_invariants(result.decision, candidates)
        _write_topic_artifacts(self._output_dir, candidates, result.decision)
        return candidates, result.decision


class StructuredTopicPipeline:
    def __init__(
        self,
        discoverer: TopicDiscoverer,
        selector: TopicSelector,
        *,
        output_dir: Path,
        downstream: HypothesisGenerator | None = None,
    ) -> None:
        self._discoverer = discoverer
        self._selector = selector
        self._output_dir = output_dir
        self._downstream = downstream

    def run(
        self, bundle: CaseBundle, context: RunContext
    ) -> tuple[list[TopicCandidate], TopicDecision]:
        candidates = self._discoverer.discover(bundle, context)
        decision = self._selector.select(candidates, bundle, context)
        _write_topic_artifacts(self._output_dir, candidates, decision)
        if self._downstream is not None and decision.status is TopicDecisionStatus.SELECTED:
            selected_topic = next(
                candidate
                for candidate in candidates
                if candidate.topic_id == decision.selected_topic_id
            )
            self._downstream.generate(bundle, selected_topic, context)
        return candidates, decision


def _visible_topic_payload(bundle: CaseBundle) -> dict[str, Any]:
    return {
        "context": bundle.context.model_dump(mode="json"),
        "evidence": [item.model_dump(mode="json") for item in bundle.evidence],
        "observations": [item.model_dump(mode="json") for item in bundle.observations],
    }


def _validate_decision_invariants(
    decision: TopicDecision,
    candidates: Sequence[TopicCandidate],
) -> None:
    candidate_ids = {candidate.topic_id for candidate in candidates}
    ranked_ids = decision.ranked_topic_ids
    score_ids = [score.topic_id for score in decision.scores]
    if len(ranked_ids) != len(set(ranked_ids)) or set(ranked_ids) != candidate_ids:
        raise TopicSelectionError("One-shot ranking must contain every candidate exactly once")
    if len(score_ids) != len(set(score_ids)) or set(score_ids) != candidate_ids:
        raise TopicSelectionError("One-shot scores must contain every candidate exactly once")

    expected_ranking = [
        score.topic_id
        for score in sorted(
            decision.scores,
            key=lambda item: (-_weighted_score(item), item.topic_id),
        )
    ]
    if ranked_ids != expected_ranking:
        raise TopicSelectionError(
            "Topic decision weighted ranking is inconsistent with its dimension scores"
        )
    passing_ids = [
        score.topic_id
        for score in sorted(
            decision.scores,
            key=lambda item: (-_weighted_score(item), item.topic_id),
        )
        if not _hard_threshold_failures(score)
    ]
    if decision.status is TopicDecisionStatus.SELECTED:
        if decision.selected_topic_id not in candidate_ids:
            raise TopicSelectionError("One-shot selected topic is not a supplied candidate")
        if not passing_ids or decision.selected_topic_id != passing_ids[0]:
            raise TopicSelectionError(
                "Selected topic must be the highest-ranked candidate that passes every "
                "hard threshold"
            )
    elif decision.selected_topic_id is not None:
        raise TopicSelectionError("An abstained one-shot decision cannot select a topic")
    elif passing_ids:
        raise TopicSelectionError(
            "Topic decision cannot abstain while a candidate passes all hard thresholds"
        )

    required_rejections = candidate_ids - {decision.selected_topic_id}
    rejection_ids = [item.get("topic_id") for item in decision.rejection_reasons]
    if (
        len(rejection_ids) != len(set(rejection_ids))
        or set(rejection_ids) != required_rejections
    ):
        raise TopicSelectionError(
            "Topic decision rejection reasons must cover exactly the unselected candidates"
        )
    if any(not item.get("reason", "").strip() for item in decision.rejection_reasons):
        raise TopicSelectionError(
            "Every rejected topic must retain a nonblank reason"
        )


def _write_topic_artifacts(
    output_dir: Path,
    candidates: Sequence[TopicCandidate],
    decision: TopicDecision,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        output_dir / "topic_candidates.json",
        {"candidates": [candidate.model_dump(mode="json") for candidate in candidates]},
    )
    write_json_atomic(output_dir / "topic_decision.json", decision)


def _score_request(
    *,
    llm_prompt: str,
    candidates: Sequence[TopicCandidate],
    bundle: CaseBundle,
    context: RunContext,
) -> LLMRequest:
    payload = {
        "context": bundle.context.model_dump(mode="json"),
        "evidence": [item.model_dump(mode="json") for item in bundle.evidence],
        "observations": [item.model_dump(mode="json") for item in bundle.observations],
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return LLMRequest(
        requested_model=context.requested_model,
        messages=(
            ChatMessage(role="system", content=llm_prompt),
            ChatMessage(role="user", content="Return JSON scores for this input: " + serialized),
        ),
        seed=context.random_seed,
        temperature=context.temperature,
        prompt_version=context.prompt_version,
        input_hashes={
            "topic_selection_input": hashlib.sha256(serialized.encode()).hexdigest(),
            "prompt": hashlib.sha256(llm_prompt.encode()).hexdigest(),
        },
        response_model=TopicScoreBatch,
        budget_partition=context.budget_partition,
        estimated_cost_cny=context.estimated_cost_cny,
        run_id=context.run_id,
        case_id=context.case_id,
        stage="topic_selection",
        experiment_batch=context.experiment_batch,
        adaptations=tuple(context.adaptations),
    )


def _validate_scores(
    value: object,
    candidates: Sequence[TopicCandidate],
) -> list[TopicDimensionScore]:
    try:
        scores = list(TopicScoreBatch.model_validate(value).scores)
    except ValidationError as error:
        raise TopicSelectionError(f"Invalid topic score JSON: {error}") from error
    score_ids = [item.topic_id for item in scores]
    candidate_ids = [item.topic_id for item in candidates]
    if len(score_ids) != len(set(score_ids)):
        raise TopicSelectionError("Topic scoring returned duplicate topic IDs")
    if len(score_ids) != len(candidate_ids) or set(score_ids) != set(candidate_ids):
        missing = sorted(set(candidate_ids) - set(score_ids))
        unknown = sorted(set(score_ids) - set(candidate_ids))
        raise TopicSelectionError(
            f"Topic score IDs do not match candidates; missing={missing}, unknown={unknown}"
        )
    return scores


def _weighted_score(score: TopicDimensionScore) -> int:
    return sum(getattr(score, field) * weight for field, weight in _WEIGHTS.items())


def _hard_threshold_failures(score: TopicDimensionScore) -> list[str]:
    return [
        f"{field}={getattr(score, field)} < {threshold}"
        for field, threshold in _HARD_THRESHOLDS.items()
        if getattr(score, field) < threshold
    ]


def _require_known_references(
    references: Sequence[str], known_ids: set[str], kind: str
) -> None:
    if not references:
        raise InputReferenceError(f"Candidate has no {kind} references")
    unknown = sorted(set(references) - known_ids)
    if unknown:
        raise InputReferenceError(f"Unknown {kind} references: {', '.join(unknown)}")


def _load_prompt(filename: str) -> str:
    resource = files("hypothesis_reasoning").joinpath("prompts", "topic", filename)
    return resource.read_text(encoding="utf-8")
