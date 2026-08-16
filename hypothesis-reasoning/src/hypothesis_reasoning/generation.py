"""Independent evidence-grounded hypothesis generation channels."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from hypothesis_reasoning.errors import ContractValidationError, InputReferenceError
from hypothesis_reasoning.io import CaseBundle, validate_hypothesis_references
from hypothesis_reasoning.llm.types import ChatMessage, LLMRequest, LLMResponse
from hypothesis_reasoning.models import Hypothesis, ReasoningType, TopicCandidate
from hypothesis_reasoning.skill_adapters.base import RunContext


class CompletionClient(Protocol):
    def complete_json(self, request: LLMRequest) -> LLMResponse: ...


class PromptProvider(Protocol):
    def get(self, reasoning_type: ReasoningType) -> str: ...


class HypothesisGenerationBatch(BaseModel):
    """Strict response envelope shared by the independent generators."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hypotheses: list[Hypothesis] = Field(min_length=1)


class PromptRegistry:
    """Resolve versioned generation prompts from installed package resources."""

    def get(self, reasoning_type: ReasoningType) -> str:
        resource = files("hypothesis_reasoning").joinpath(
            "prompts", "generation", f"{reasoning_type.value}.md"
        )
        return resource.read_text(encoding="utf-8")


class _IndependentGenerator:
    reasoning_type: ReasoningType

    def __init__(
        self,
        llm: CompletionClient,
        prompt_registry: PromptProvider | None = None,
        *,
        require_grounding: bool = True,
    ) -> None:
        self._llm = llm
        self._prompt_registry = (
            prompt_registry if prompt_registry is not None else PromptRegistry()
        )
        self._require_grounding = require_grounding

    def generate(
        self,
        bundle: CaseBundle,
        topic: TopicCandidate,
        context: RunContext,
    ) -> list[Hypothesis]:
        prompt = self._prompt_registry.get(self.reasoning_type)
        serialized = json.dumps(
            _visible_generation_payload(
                bundle,
                topic,
                require_grounding=self._require_grounding,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
        response = self._llm.complete_json(
            LLMRequest(
                requested_model=context.requested_model,
                messages=(
                    ChatMessage(role="system", content=prompt),
                    ChatMessage(role="user", content=serialized),
                ),
                seed=context.random_seed,
                temperature=context.temperature,
                prompt_version=context.prompt_version,
                input_hashes={
                    "generation_input": _sha256(serialized),
                    "prompt": _sha256(prompt),
                },
                response_model=HypothesisGenerationBatch,
                budget_partition=context.budget_partition,
                estimated_cost_cny=context.estimated_cost_cny,
                run_id=context.run_id,
                case_id=context.case_id,
                stage=f"hypothesis_generation_{self.reasoning_type.value}",
                experiment_batch=context.experiment_batch,
                adaptations=tuple(context.adaptations),
            )
        )
        batch = HypothesisGenerationBatch.model_validate(response.parsed_json)
        return _validated_hypotheses(
            batch.hypotheses,
            bundle=bundle,
            topic=topic,
            reasoning_type=self.reasoning_type,
        )


class InductiveGenerator(_IndependentGenerator):
    """Generate candidates from patterns in the visible case records."""

    reasoning_type = ReasoningType.INDUCTIVE


class DeductiveGenerator(_IndependentGenerator):
    """Generate candidates by deriving predictions from visible premises."""

    reasoning_type = ReasoningType.DEDUCTIVE


def _visible_generation_payload(
    bundle: CaseBundle,
    topic: TopicCandidate,
    *,
    require_grounding: bool,
) -> dict[str, object]:
    return {
        "context": bundle.context.model_dump(mode="json"),
        "evidence": [item.model_dump(mode="json") for item in bundle.evidence],
        "observations": [item.model_dump(mode="json") for item in bundle.observations],
        "topic": topic.model_dump(mode="json"),
        "grounding_policy": {
            "claim_evidence_required": require_grounding,
            "known_identifiers_only": True,
        },
    }


def _validated_hypotheses(
    hypotheses: list[Hypothesis],
    *,
    bundle: CaseBundle,
    topic: TopicCandidate,
    reasoning_type: ReasoningType,
) -> list[Hypothesis]:
    for hypothesis in hypotheses:
        if hypothesis.topic_id != topic.topic_id:
            raise InputReferenceError(
                f"Hypothesis topic reference {hypothesis.topic_id} does not match "
                f"selected topic {topic.topic_id}"
            )
        if hypothesis.reasoning_type is not reasoning_type:
            raise ContractValidationError(
                f"{reasoning_type.value} generator returned "
                f"reasoning_type={hypothesis.reasoning_type.value}"
            )
        validate_hypothesis_references(hypothesis, bundle)
    return list(hypotheses)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
