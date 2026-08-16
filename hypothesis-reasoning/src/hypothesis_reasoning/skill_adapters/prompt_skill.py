"""Prompt-backed adapter for baseline and audited candidate instructions."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from hypothesis_reasoning.errors import InputReferenceError, SkillAdaptationError
from hypothesis_reasoning.io import CaseBundle, validate_hypothesis_references
from hypothesis_reasoning.llm.types import ChatMessage, LLMRequest, LLMResponse
from hypothesis_reasoning.models import Hypothesis, TopicCandidate
from hypothesis_reasoning.skill_adapters.base import AdaptationKind, AdaptationRecord, RunContext

_HOST_EXECUTION_DIRECTION = re.compile(
    r"""
    (?:
        \b(?:use|invoke|call|run|execute|install|spawn|delegate|read|write|edit|
            create|open|launch|download|upload|fetch|search|browse|save|export|
            append|delete|remove|copy|move)\b
        [^\n]{0,200}
        (?:
            \b(?:bash|shell|terminal|powershell|cmd|tool|subagent|agent|worker|pip|npm|
                pnpm|yarn|uv|command|file|package|worktree|python|subprocess|web|
                internet|url|https?|output|results?|path|directory|folder|curl|wget)\b
            | (?:scripts|references|assets)[\\/][A-Za-z0-9_.\\/-]+
            | \b(?:subprocess\.|os\.system|Path\s*\(|open\s*\()
        )
        | ^\s*(?:[A-Z_][A-Z0-9_]*=\S+\s+)*
            (?:python3?|uv|pip3?|curl|wget|git|bash|zsh|sh|powershell|pwsh|rm|
               del|erase|cp|mv|robocopy|xcopy|cmd(?:\.exe)?|node|iex|invoke-expression|
               start-process)\b
        | \b(?:subprocess\.(?:run|Popen|call)|os\.system)\s*\(
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
_FENCE_START = re.compile(r"^\s*(`{3,}|~{3,})")
_RESIDUAL_HOST_ACTION = re.compile(
    r"^\s*(?:`{3,}|~{3,})|"
    r"(?:scripts|references|assets)[\\/][A-Za-z0-9_.\\/-]+|"
    r"\b(?:subprocess\.(?:run|Popen|call)|os\.system|write_text|write_bytes|urlopen)\b|"
    r"^\s*(?:rm|del|erase|cp|mv|curl|wget|python3?|bash|sh|powershell|pwsh|"
    r"invoke-expression|start-process)\b|"
    r"\b(?:download|upload|fetch|browse|search)\b[^\n]{0,120}"
    r"\b(?:web|internet|url|https?|file|dataset|curl|wget)\b|"
    r"\b(?:save|write|append|export|create|edit|delete|remove|copy|move)\b"
    r"[^\n]{0,120}\b(?:file|directory|folder|path|output|results?)\b",
    re.IGNORECASE,
)
_REFERENCE_CONSTRAINTS = (
    "Only use evidence_ids and observation_ids supplied in the input. "
    "Never invent papers, evidence, observations, measurements, or identifiers. "
    "Every hypothesis must use the selected topic_id exactly."
)


class CompletionClient(Protocol):
    def complete_json(self, request: LLMRequest) -> LLMResponse: ...


class HypothesisBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hypotheses: list[Hypothesis] = Field(min_length=1)


class PromptSkillAdapter:
    def __init__(
        self, llm: CompletionClient, *, prompt: str, adapt_skill: bool = True
    ) -> None:
        self._llm = llm
        self._prompt = prompt
        self._adapt_skill = adapt_skill

    def generate(
        self, bundle: CaseBundle, topic: TopicCandidate, context: RunContext
    ) -> list[Hypothesis]:
        system_prompt = self._prompt
        if self._adapt_skill:
            system_prompt = _adapt_skill_prompt(self._prompt, context)
        payload = {
            "context": bundle.context.model_dump(mode="json"),
            "evidence": [item.model_dump(mode="json") for item in bundle.evidence],
            "observations": [item.model_dump(mode="json") for item in bundle.observations],
            "topic": topic.model_dump(mode="json"),
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        response = self._llm.complete_json(
            LLMRequest(
                requested_model=context.requested_model,
                messages=(
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(role="user", content="Return JSON for this input: " + serialized),
                ),
                seed=context.random_seed,
                temperature=context.temperature,
                prompt_version=context.prompt_version,
                input_hashes={
                    "case_bundle": hashlib.sha256(serialized.encode()).hexdigest(),
                    "prompt": hashlib.sha256(system_prompt.encode()).hexdigest(),
                },
                response_model=HypothesisBatch,
                budget_partition=context.budget_partition,
                estimated_cost_cny=context.estimated_cost_cny,
                run_id=context.run_id,
                case_id=context.case_id,
                stage="hypothesis_generation",
                experiment_batch=context.experiment_batch,
                adaptations=tuple(context.adaptations),
            )
        )
        batch = HypothesisBatch.model_validate(response.parsed_json)
        for hypothesis in batch.hypotheses:
            if hypothesis.topic_id != topic.topic_id:
                raise InputReferenceError(
                    f"Hypothesis topic reference {hypothesis.topic_id} does not match "
                    f"selected topic {topic.topic_id}"
                )
            validate_hypothesis_references(hypothesis, bundle)
        return list(batch.hypotheses)


def _adapt_skill_prompt(prompt: str, context: RunContext) -> str:
    kept_lines: list[str] = []
    lines = prompt.splitlines()
    index = 0
    if lines and lines[0].strip() == "---":
        frontmatter_end = next(
            (position for position in range(1, len(lines)) if lines[position].strip() == "---"),
            len(lines) - 1,
        )
        context.adaptations.append(
            AdaptationRecord(
                kind=AdaptationKind.HOST_DIRECTION_REMOVED,
                detail=f"Removed skill frontmatter at source lines 1-{frontmatter_end + 1}.",
            )
        )
        index = frontmatter_end + 1

    while index < len(lines):
        line = lines[index]
        fence_match = _FENCE_START.match(line)
        if fence_match:
            block_start = index
            fence_marker = fence_match.group(1)
            fence_close = re.compile(
                rf"^\s*{re.escape(fence_marker[0])}{{{len(fence_marker)},}}\s*$"
            )
            index += 1
            while index < len(lines) and not fence_close.match(lines[index]):
                index += 1
            if index < len(lines):
                index += 1
            context.adaptations.append(
                AdaptationRecord(
                    kind=AdaptationKind.HOST_DIRECTION_REMOVED,
                    detail=(
                        f"Removed executable-capable fenced block at source lines "
                        f"{block_start + 1}-{index}."
                    ),
                )
            )
            continue
        if _HOST_EXECUTION_DIRECTION.search(line):
            context.adaptations.append(
                AdaptationRecord(
                    kind=AdaptationKind.HOST_DIRECTION_REMOVED,
                    detail=f"Removed host execution direction at source line {index + 1}.",
                )
            )
            index += 1
            while index < len(lines) and _is_directive_continuation(lines[index]):
                index += 1
            continue
        kept_lines.append(line)
        index += 1

    adapted_source = "\n".join(kept_lines).strip()
    residual = next(
        (
            (line_number, line)
            for line_number, line in enumerate(adapted_source.splitlines(), start=1)
            if _RESIDUAL_HOST_ACTION.search(line)
        ),
        None,
    )
    if residual is not None:
        line_number, line = residual
        raise SkillAdaptationError(
            "sanitized skill still contains a host-action signal at adapted line "
            f"{line_number}: {line.strip()[:120]}"
        )

    context.adaptations.append(
        AdaptationRecord(
            kind=AdaptationKind.REFERENCE_CONSTRAINTS_ADDED,
            detail="Added the standard evidence, observation, and topic reference constraints.",
        )
    )
    json_contract = json.dumps(HypothesisBatch.model_json_schema(), sort_keys=True)
    context.adaptations.append(
        AdaptationRecord(
            kind=AdaptationKind.JSON_CONTRACT_ADDED,
            detail="Added the standard HypothesisBatch JSON Schema contract.",
        )
    )
    sections = [adapted_source, _REFERENCE_CONSTRAINTS, json_contract]
    return "\n\n".join(section for section in sections if section)


def _is_directive_continuation(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and (
        line[:1].isspace()
        or stripped.startswith(("--", "- ", "* ", "|", "&&", "||"))
    )
