"""Evidence-grounded research topic discovery."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from importlib.resources import files
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from hypothesis_reasoning.errors import TopicDiscoveryError
from hypothesis_reasoning.io import CaseBundle
from hypothesis_reasoning.llm.types import ChatMessage, LLMRequest, LLMResponse
from hypothesis_reasoning.models import TopicCandidate
from hypothesis_reasoning.skill_adapters.base import RunContext


class CompletionClient(Protocol):
    def complete_json(self, request: LLMRequest) -> LLMResponse: ...


class TopicCandidateBatch(BaseModel):
    """The JSON envelope shared by discovery and its single repair attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidates: list[TopicCandidate]


class TopicDiscoverer:
    """Discover a fixed-size, grounded candidate set from visible case inputs."""

    def __init__(self, llm: CompletionClient, *, prompt: str | None = None) -> None:
        self._llm = llm
        self._prompt = prompt if prompt is not None else _load_prompt("discover.md")

    def discover(self, bundle: CaseBundle, context: RunContext) -> list[TopicCandidate]:
        payload = _visible_payload(bundle)
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        response = self._llm.complete_json(
            _request(
                context=context,
                prompt=self._prompt,
                user_content="Return JSON for this visible case input: " + serialized,
                serialized_input=serialized,
                stage="topic_discovery",
            )
        )
        try:
            return validate_discovered_candidates(response.parsed_json, bundle)
        except TopicDiscoveryError as first_error:
            invalid_output = json.dumps(response.parsed_json, ensure_ascii=False, sort_keys=True)
            repair_instruction = (
                "Return corrected JSON for the same visible case input. "
                f"The prior output violated this requirement: {first_error}. "
                "Do not introduce any evidence, observations, or facts outside the input. "
                f"Visible input: {serialized}. Prior output: {invalid_output}"
            )
            repaired = self._llm.complete_json(
                _request(
                    context=context,
                    prompt=self._prompt,
                    user_content=repair_instruction,
                    serialized_input=serialized,
                    stage="topic_discovery_repair",
                    invalid_output=invalid_output,
                )
            )
            try:
                return validate_discovered_candidates(repaired.parsed_json, bundle)
            except TopicDiscoveryError as final_error:
                raise final_error from first_error


def _request(
    *,
    context: RunContext,
    prompt: str,
    user_content: str,
    serialized_input: str,
    stage: str,
    invalid_output: str | None = None,
) -> LLMRequest:
    hashes = {
        "case_bundle": hashlib.sha256(serialized_input.encode()).hexdigest(),
        "prompt": hashlib.sha256(prompt.encode()).hexdigest(),
    }
    if invalid_output is not None:
        hashes["invalid_output"] = hashlib.sha256(invalid_output.encode()).hexdigest()
    return LLMRequest(
        requested_model=context.requested_model,
        messages=(
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content=user_content),
        ),
        seed=context.random_seed,
        temperature=context.temperature,
        prompt_version=context.prompt_version,
        input_hashes=hashes,
        response_model=TopicCandidateBatch,
        budget_partition=context.budget_partition,
        estimated_cost_cny=context.estimated_cost_cny,
        run_id=context.run_id,
        case_id=context.case_id,
        stage=stage,
        experiment_batch=context.experiment_batch,
        adaptations=tuple(context.adaptations),
    )


def _visible_payload(bundle: CaseBundle) -> dict[str, Any]:
    return {
        "context": bundle.context.model_dump(mode="json"),
        "evidence": [item.model_dump(mode="json") for item in bundle.evidence],
        "observations": [item.model_dump(mode="json") for item in bundle.observations],
    }


def validate_discovered_candidates(
    value: object, bundle: CaseBundle
) -> list[TopicCandidate]:
    try:
        candidates = list(TopicCandidateBatch.model_validate(value).candidates)
    except ValidationError as error:
        raise TopicDiscoveryError(f"Invalid topic candidate JSON: {error}") from error

    required_count = bundle.context.max_topic_candidates
    if len(candidates) != required_count:
        raise TopicDiscoveryError(
            f"Topic discovery must return exactly {required_count} candidates; "
            f"received {len(candidates)}"
        )
    topic_ids = [candidate.topic_id for candidate in candidates]
    if len(topic_ids) != len(set(topic_ids)):
        raise TopicDiscoveryError("Topic discovery returned duplicate topic IDs")

    known_evidence = {item.evidence_id for item in bundle.evidence}
    known_observations = {item.observation_id for item in bundle.observations}
    for candidate in candidates:
        _require_grounded_references(
            candidate.evidence_ids,
            known_evidence,
            "evidence",
            candidate.topic_id,
        )
        _require_grounded_references(
            candidate.observation_ids,
            known_observations,
            "observation",
            candidate.topic_id,
        )
        if not any(path.strip() for path in candidate.expected_validation):
            raise TopicDiscoveryError(
                f"Topic {candidate.topic_id} has no declared validation path"
            )
        if not any(risk.strip() for risk in candidate.risks):
            raise TopicDiscoveryError(f"Topic {candidate.topic_id} has no declared risks")
    return candidates


def _require_grounded_references(
    references: Sequence[str],
    known_ids: set[str],
    kind: str,
    topic_id: str,
) -> None:
    if not references:
        raise TopicDiscoveryError(f"Topic {topic_id} has no {kind} references")
    unknown_ids = sorted(set(references) - known_ids)
    if unknown_ids:
        raise TopicDiscoveryError(
            f"Unknown {kind} references for topic {topic_id}: {', '.join(unknown_ids)}"
        )


def _load_prompt(filename: str) -> str:
    resource = files("hypothesis_reasoning").joinpath("prompts", "topic", filename)
    return resource.read_text(encoding="utf-8")
