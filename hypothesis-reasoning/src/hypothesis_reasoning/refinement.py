"""Exactly-once, evidence-bounded hypothesis refinement."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from typing import Protocol

from hypothesis_reasoning.critique import EvidenceGate
from hypothesis_reasoning.errors import (
    RefinementLimitError,
    RefinementValidationError,
)
from hypothesis_reasoning.io import CaseBundle, validate_hypothesis_references
from hypothesis_reasoning.llm.types import ChatMessage, LLMRequest, LLMResponse
from hypothesis_reasoning.models import Critique, Hypothesis, Severity
from hypothesis_reasoning.skill_adapters.base import RunContext


class CompletionClient(Protocol):
    def complete_json(self, request: LLMRequest) -> LLMResponse: ...


class Refiner:
    """Apply one targeted revision to each hypothesis lineage."""

    def __init__(
        self,
        llm: CompletionClient,
        *,
        prompt: str | None = None,
        max_refinement_rounds: int = 1,
        require_grounding: bool = True,
    ) -> None:
        if max_refinement_rounds != 1:
            raise ValueError("Task 7 permits exactly one refinement round")
        self._llm = llm
        self._prompt = prompt if prompt is not None else _load_prompt()
        self._require_grounding = require_grounding
        self._used_lineages: set[tuple[str, str]] = set()
        self._lineage_roots: dict[tuple[str, str], str] = {}

    def revise(
        self,
        hypothesis: Hypothesis,
        critiques: list[Critique],
        bundle: CaseBundle,
        context: RunContext,
    ) -> Hypothesis:
        root_id = self._lineage_roots.get(
            (context.run_id, hypothesis.hypothesis_id),
            hypothesis.parent_id or hypothesis.hypothesis_id,
        )
        lineage_key = (context.run_id, root_id)
        if lineage_key in self._used_lineages:
            raise RefinementLimitError(
                f"Hypothesis lineage {root_id} in run {context.run_id} has already "
                "used its only revision"
            )
        if not critiques:
            raise RefinementValidationError("Refinement requires at least one critique")
        if any(item.hypothesis_id != hypothesis.hypothesis_id for item in critiques):
            raise RefinementValidationError(
                "Every critique must target the hypothesis being revised"
            )

        # A model call is the scarce refinement round, even if its output later fails gates.
        self._used_lineages.add(lineage_key)
        payload = {
            "context": bundle.context.model_dump(mode="json"),
            "evidence": [item.model_dump(mode="json") for item in bundle.evidence],
            "observations": [
                item.model_dump(mode="json") for item in bundle.observations
            ],
            "original_hypothesis": hypothesis.model_dump(mode="json"),
            "critiques": [item.model_dump(mode="json") for item in critiques],
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
                    "refinement_input": _sha256(serialized),
                    "prompt": _sha256(self._prompt),
                },
                response_model=Hypothesis,
                budget_partition=context.budget_partition,
                estimated_cost_cny=context.estimated_cost_cny,
                run_id=context.run_id,
                case_id=context.case_id,
                stage="hypothesis_refinement",
                experiment_batch=context.experiment_batch,
                adaptations=tuple(context.adaptations),
            )
        )
        candidate = Hypothesis.model_validate(response.parsed_json)
        if candidate.parent_id != hypothesis.hypothesis_id:
            raise RefinementValidationError(
                "Revised hypothesis parent_id must equal the original hypothesis_id"
            )
        validate_hypothesis_references(candidate, bundle)

        revised = candidate.model_copy(
            update={
                "hypothesis_id": _revision_id(hypothesis.hypothesis_id, candidate),
                "parent_id": hypothesis.hypothesis_id,
            }
        )
        gate_findings = EvidenceGate(
            expected_topic_id=hypothesis.topic_id,
            require_grounding=self._require_grounding,
        ).check(revised, bundle)
        if any(item.severity is Severity.BLOCKING for item in gate_findings):
            raise RefinementValidationError(
                "Revised hypothesis failed a blocking evidence gate"
            )
        self._lineage_roots[(context.run_id, revised.hypothesis_id)] = root_id
        return revised


def _revision_id(original_id: str, candidate: Hypothesis) -> str:
    payload = candidate.model_dump(mode="json")
    payload.pop("hypothesis_id")
    material = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return f"{original_id}-r1-{_sha256(material)[:12]}"


def _load_prompt() -> str:
    resource = files("hypothesis_reasoning").joinpath(
        "prompts", "refinement", "revise.md"
    )
    return resource.read_text(encoding="utf-8")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
