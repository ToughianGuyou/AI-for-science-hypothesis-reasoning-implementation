"""Deterministic consolidation for competing hypothesis candidates."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

from hypothesis_reasoning.models import Hypothesis

_DUPLICATE_THRESHOLD = 0.90


class HypothesisPool:
    """Preserve competing candidates while consolidating true duplicates."""

    def consolidate(self, candidates: Sequence[Hypothesis]) -> list[Hypothesis]:
        consolidated: list[Hypothesis] = []
        for candidate in candidates:
            for index, existing in enumerate(consolidated):
                if _duplicates(existing, candidate):
                    consolidated[index] = _merge_duplicates(existing, candidate)
                    break
            else:
                consolidated.append(candidate)
        return consolidated


def _duplicates(first: Hypothesis, second: Hypothesis) -> bool:
    return (
        _trigram_jaccard(first.statement, second.statement) >= _DUPLICATE_THRESHOLD
        and _trigram_jaccard(first.mechanism, second.mechanism) >= _DUPLICATE_THRESHOLD
    )


def _trigram_jaccard(first: str, second: str) -> float:
    first_trigrams = _character_trigrams(_normalize(first))
    second_trigrams = _character_trigrams(_normalize(second))
    union = first_trigrams | second_trigrams
    if not union:
        return 1.0
    return len(first_trigrams & second_trigrams) / len(union)


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _character_trigrams(value: str) -> set[str]:
    if len(value) < 3:
        return {value} if value else set()
    return {value[index : index + 3] for index in range(len(value) - 2)}


def _merge_duplicates(first: Hypothesis, second: Hypothesis) -> Hypothesis:
    if _completeness(second) > _completeness(first):
        winner, other = second, first
    else:
        winner, other = first, second
    evidence_ids = _stable_valid_union(winner.evidence_ids, other.evidence_ids)
    payload = winner.model_dump(mode="python")
    payload["evidence_ids"] = evidence_ids
    return Hypothesis.model_validate(payload)


def _completeness(hypothesis: Hypothesis) -> int:
    text_lists = (
        hypothesis.evidence_ids,
        hypothesis.observation_ids,
        hypothesis.assumptions,
        hypothesis.scope_conditions,
        hypothesis.falsification_criteria,
        hypothesis.alternative_explanations,
        hypothesis.validation_plan.baselines,
        hypothesis.validation_plan.metrics,
        hypothesis.validation_plan.required_data,
        hypothesis.unsupported_claims,
    )
    return len(hypothesis.predictions) + sum(
        sum(bool(item.strip()) for item in items) for items in text_lists
    )


def _stable_valid_union(*identifier_lists: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for identifiers in identifier_lists:
        for identifier in identifiers:
            if identifier.strip() and identifier not in seen:
                seen.add(identifier)
                result.append(identifier)
    return result
