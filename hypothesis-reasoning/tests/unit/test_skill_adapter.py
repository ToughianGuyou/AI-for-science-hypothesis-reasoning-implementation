from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import hypothesis_reasoning.skill_adapters.prompt_skill as prompt_skill_module
from hypothesis_reasoning.errors import InputReferenceError, SkillAdaptationError
from hypothesis_reasoning.io import CaseBundle
from hypothesis_reasoning.llm.types import LLMRequest, LLMResponse
from hypothesis_reasoning.models import Hypothesis, RunTrace, TopicCandidate, Usage
from hypothesis_reasoning.research.audit import AuditDecision, AuditStatus, SkillAuditRecord
from hypothesis_reasoning.skill_adapters.base import RunContext
from hypothesis_reasoning.skill_adapters.prompt_skill import PromptSkillAdapter
from hypothesis_reasoning.skill_adapters.registry import (
    build_baseline_adapter,
    build_prompt_skill_adapter,
)
from tests.factories import hypothesis_payload


def hypothesis_json(**overrides: Any) -> dict[str, Any]:
    return {"hypotheses": [hypothesis_payload(**overrides)]}


class FakeLLM:
    def __init__(self) -> None:
        self._payload: dict[str, Any] | None = None
        self.requests: list[LLMRequest] = []

    def returns(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        assert self._payload is not None
        self.requests.append(request)
        return LLMResponse(
            request_id="request-001",
            requested_model=request.requested_model,
            returned_model="qwen-plus-2026-08-01",
            content_hash="c" * 64,
            parsed_json=self._payload,
            called_at=datetime(2026, 8, 15, tzinfo=UTC),
            usage=Usage(
                prompt_tokens=100,
                completion_tokens=200,
                total_tokens=300,
                estimated_cost_cny=0.01,
                latency_ms=25,
            ),
            retries=0,
        )


def run_context(method_id: str = "B1") -> RunContext:
    return RunContext(
        run_id="run-001",
        case_id="case-001",
        method_id=method_id,
        requested_model="qwen-plus",
        random_seed=11,
        temperature=0.2,
        prompt_version="skill-adapter-v1",
        budget_partition="development",
        estimated_cost_cny=Decimal("0.01"),
        experiment_batch="task4-tests",
    )


def test_adapter_cannot_emit_unknown_evidence(
    valid_bundle: CaseBundle,
    selected_topic: dict[str, object],
) -> None:
    fake_llm = FakeLLM()
    fake_llm.returns(hypothesis_json(evidence_ids=["invented-e1"]))

    with pytest.raises(InputReferenceError, match="invented-e1"):
        PromptSkillAdapter(
            fake_llm,
            prompt="Generate one evidence-bound JSON hypothesis.",
        ).generate(
            valid_bundle,
            TopicCandidate.model_validate(selected_topic),
            run_context(),
        )


def test_adapter_rejects_a_hypothesis_for_a_different_topic(
    valid_bundle: CaseBundle,
    selected_topic: dict[str, object],
) -> None:
    fake_llm = FakeLLM()
    fake_llm.returns(hypothesis_json(topic_id="topic-outside-selection"))

    with pytest.raises(InputReferenceError, match="topic-outside-selection"):
        PromptSkillAdapter(fake_llm, prompt="Generate JSON hypotheses.").generate(
            valid_bundle,
            TopicCandidate.model_validate(selected_topic),
            run_context(),
        )


def test_b1_strips_host_execution_directions_and_records_each_adaptation(
    valid_bundle: CaseBundle,
    selected_topic: dict[str, object],
) -> None:
    fake_llm = FakeLLM()
    fake_llm.returns(hypothesis_json())
    context = run_context()
    adapter = PromptSkillAdapter(
        fake_llm,
        prompt=(
            "Analyze the supplied evidence.\n"
            "Use the Bash tool to run `pip install example-package`.\n"
            "Generate competing, falsifiable hypotheses.\n"
        ),
    )

    hypotheses = adapter.generate(
        valid_bundle,
        TopicCandidate.model_validate(selected_topic),
        context,
    )

    system_prompt = fake_llm.requests[0].messages[0].content
    assert hypotheses == [Hypothesis.model_validate(hypothesis_payload())]
    assert "pip install" not in system_prompt
    assert "Bash tool" not in system_prompt
    assert "Generate competing, falsifiable hypotheses." in system_prompt
    assert "Only use evidence_ids and observation_ids supplied in the input" in system_prompt
    assert "hypotheses" in system_prompt
    assert [record.kind.value for record in context.adaptations] == [
        "host_direction_removed",
        "reference_constraints_added",
        "json_contract_added",
    ]
    assert [item.kind for item in fake_llm.requests[0].adaptations] == [
        "host_direction_removed",
        "reference_constraints_added",
        "json_contract_added",
    ]


def test_b1_sanitizer_removes_path_agent_code_block_and_multiline_directions(
    valid_bundle: CaseBundle,
    selected_topic: dict[str, object],
) -> None:
    fake_llm = FakeLLM()
    fake_llm.returns(hypothesis_json())
    context = run_context()
    adapter = PromptSkillAdapter(
        fake_llm,
        prompt=(
            "Keep this scientific reasoning framework.\n"
            "Run python scripts/tree.py cycle.\n"
            "Read references/guide.md before continuing.\n"
            "Delegate the analysis to a worker.\n"
            "Execute subprocess.run(['python', 'worker.py']).\n"
            "```python\n"
            "from pathlib import Path\n"
            "Path('output.json').write_text('{}')\n"
            "```\n"
            "Use this shell command:\n"
            "    --unsafe-continuation value\n"
            "Generate competing, falsifiable hypotheses.\n"
        ),
    )

    adapter.generate(
        valid_bundle,
        TopicCandidate.model_validate(selected_topic),
        context,
    )

    system_prompt = fake_llm.requests[0].messages[0].content
    assert "Keep this scientific reasoning framework." in system_prompt
    assert "Generate competing, falsifiable hypotheses." in system_prompt
    for forbidden in (
        "scripts/tree.py",
        "references/guide.md",
        "worker",
        "subprocess.run",
        "write_text",
        "shell command",
        "unsafe-continuation",
    ):
        assert forbidden not in system_prompt
    removed = [
        record
        for record in context.adaptations
        if record.kind.value == "host_direction_removed"
    ]
    assert len(removed) == 6


def test_b1_sanitizer_removes_alternate_fences_and_host_action_bypasses(
    valid_bundle: CaseBundle,
    selected_topic: dict[str, object],
) -> None:
    fake_llm = FakeLLM()
    fake_llm.returns(hypothesis_json())
    context = run_context()
    adapter = PromptSkillAdapter(
        fake_llm,
        prompt=(
            "Keep this falsifiable scientific reasoning framework.\n"
            "~~~bash\n"
            "echo unsafe\n"
            "~~~\n"
            "rm -rf outputs\n"
            "Download the dataset with wget.\n"
            "Search the web with curl.\n"
            "Save results to a file.\n"
            "Generate competing hypotheses.\n"
        ),
    )

    adapter.generate(
        valid_bundle,
        TopicCandidate.model_validate(selected_topic),
        context,
    )

    system_prompt = fake_llm.requests[0].messages[0].content
    assert "Keep this falsifiable scientific reasoning framework." in system_prompt
    assert "Generate competing hypotheses." in system_prompt
    for forbidden in ("~~~", "echo unsafe", "rm -rf", "wget", "curl", "Save results"):
        assert forbidden not in system_prompt


def test_b1_post_scan_rejects_any_remaining_host_action(
    valid_bundle: CaseBundle,
    selected_topic: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = FakeLLM()
    fake_llm.returns(hypothesis_json())
    monkeypatch.setattr(
        prompt_skill_module,
        "_HOST_EXECUTION_DIRECTION",
        re.compile(r"never-match-this-host-action"),
    )

    with pytest.raises(SkillAdaptationError, match="host-action signal"):
        PromptSkillAdapter(fake_llm, prompt="rm -rf outputs").generate(
            valid_bundle,
            TopicCandidate.model_validate(selected_topic),
            run_context(),
        )

    assert fake_llm.requests == []


def test_b0_uses_only_the_frozen_baseline_prompt(
    tmp_path: Path,
    valid_bundle: CaseBundle,
    selected_topic: dict[str, object],
) -> None:
    prompt_path = tmp_path / "generate.md"
    prompt_path.write_text("FROZEN BASELINE PROMPT WITH JSON CONTRACT", encoding="utf-8")
    fake_llm = FakeLLM()
    fake_llm.returns(hypothesis_json())
    context = run_context(method_id="B0")

    adapter = build_baseline_adapter(fake_llm, prompt_path)
    adapter.generate(
        valid_bundle,
        TopicCandidate.model_validate(selected_topic),
        context,
    )

    assert fake_llm.requests[0].messages[0].content == "FROZEN BASELINE PROMPT WITH JSON CONTRACT"
    assert context.adaptations == []


def test_b1_registry_rejects_design_only_candidate(tmp_path: Path) -> None:
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("# Design reference\n", encoding="utf-8")
    record = SkillAuditRecord.model_construct(
        candidate_name="Design Reference",
        status=AuditStatus.VERIFIED,
        decision=AuditDecision.DESIGN_ONLY,
        content_sha256=hashlib.sha256(skill_path.read_bytes()).hexdigest(),
    )

    with pytest.raises(ValueError, match="not approved for B1"):
        build_prompt_skill_adapter(FakeLLM(), skill_path, record)


def test_direct_reuse_registry_does_not_inject_external_adaptations(
    tmp_path: Path,
    valid_bundle: CaseBundle,
    selected_topic: dict[str, object],
) -> None:
    skill_path = tmp_path / "SKILL.md"
    prompt = "Generate one exact evidence-bound JSON hypothesis."
    skill_path.write_text(prompt, encoding="utf-8")
    record = SkillAuditRecord.model_construct(
        candidate_name="Exact Runtime Prompt",
        status=AuditStatus.VERIFIED,
        decision=AuditDecision.DIRECT_REUSE,
        content_sha256=hashlib.sha256(skill_path.read_bytes()).hexdigest(),
    )
    fake_llm = FakeLLM()
    fake_llm.returns(hypothesis_json())
    context = run_context()

    build_prompt_skill_adapter(fake_llm, skill_path, record).generate(
        valid_bundle,
        TopicCandidate.model_validate(selected_topic),
        context,
    )

    assert fake_llm.requests[0].messages[0].content == prompt
    assert context.adaptations == []


def test_b1_adaptations_can_be_preserved_in_run_trace(
    valid_bundle: CaseBundle,
    selected_topic: dict[str, object],
) -> None:
    fake_llm = FakeLLM()
    fake_llm.returns(hypothesis_json())
    context = run_context()
    PromptSkillAdapter(
        fake_llm,
        prompt="Use the Bash tool to run a command.\nGenerate competing hypotheses.",
    ).generate(valid_bundle, TopicCandidate.model_validate(selected_topic), context)

    trace = RunTrace(
        run_id=context.run_id,
        case_id=context.case_id,
        method_id=context.method_id,
        random_seed=context.random_seed,
        requested_model=context.requested_model,
        returned_model="qwen-plus-2026-08-01",
        parameters={"temperature": context.temperature},
        started_at=datetime(2026, 8, 15, tzinfo=UTC),
        finished_at=datetime(2026, 8, 15, tzinfo=UTC),
        prompt_version=context.prompt_version,
        code_version="task4-test",
        input_hash="d" * 64,
        output_hash="e" * 64,
        usage=Usage(
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            estimated_cost_cny=0.01,
            latency_ms=25,
        ),
        retries=0,
        stage="hypothesis_generation",
        status="succeeded",
        adaptations=context.trace_adaptations(),
    )

    assert [item["kind"] for item in trace.adaptations] == [
        "host_direction_removed",
        "reference_constraints_added",
        "json_contract_added",
    ]
