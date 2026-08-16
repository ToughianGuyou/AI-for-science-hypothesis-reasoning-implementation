"""Construction helpers for B0 and B1 adapters."""

from __future__ import annotations

from pathlib import Path

from hypothesis_reasoning.research.audit import (
    AuditDecision,
    AuditStatus,
    SkillAuditRecord,
    content_sha256,
)
from hypothesis_reasoning.skill_adapters.prompt_skill import CompletionClient, PromptSkillAdapter


def build_baseline_adapter(llm: CompletionClient, prompt_path: Path) -> PromptSkillAdapter:
    return PromptSkillAdapter(
        llm,
        prompt=prompt_path.read_text(encoding="utf-8"),
        adapt_skill=False,
    )


def build_prompt_skill_adapter(
    llm: CompletionClient, skill_path: Path, audit_record: SkillAuditRecord
) -> PromptSkillAdapter:
    if audit_record.status is not AuditStatus.VERIFIED:
        raise ValueError(f"candidate {audit_record.candidate_name} has not been verified")
    if audit_record.decision not in {AuditDecision.ADAPT, AuditDecision.DIRECT_REUSE}:
        raise ValueError(
            f"candidate {audit_record.candidate_name} is not approved for B1 "
            f"(decision={audit_record.decision.value})"
        )
    actual_hash = content_sha256(skill_path)
    if actual_hash != audit_record.content_sha256:
        raise ValueError(
            f"candidate {audit_record.candidate_name} content does not match its audit record"
        )
    return PromptSkillAdapter(
        llm,
        prompt=skill_path.read_text(encoding="utf-8"),
        adapt_skill=audit_record.decision is AuditDecision.ADAPT,
    )
