from __future__ import annotations

import json
from datetime import UTC, datetime

from hypothesis_reasoning.llm.types import TraceEvent
from hypothesis_reasoning.models import Usage
from hypothesis_reasoning.tracing import TraceWriter


def event(run_id: str = "run-001") -> TraceEvent:
    return TraceEvent(
        occurred_at=datetime(2026, 8, 15, tzinfo=UTC),
        run_id=run_id,
        case_id="case-001",
        stage="generation",
        experiment_batch="batch-001",
        random_seed=11,
        prompt_version="prompt-v1",
        input_hashes={"case": "abc"},
        parameters={
            "temperature": 0.2,
            "enable_thinking": False,
            "response_format": {"type": "json_object"},
        },
        requested_model="qwen-plus",
        returned_model="qwen-plus-2026-08-01",
        endpoint_fingerprint="f" * 64,
        request_id="chatcmpl-001",
        content_hash="a" * 64,
        parsed_json={"answer": 42},
        usage=Usage(
            prompt_tokens=120,
            completion_tokens=30,
            total_tokens=150,
            estimated_cost_cny=0.00042,
            latency_ms=25,
        ),
        retries=0,
        status="succeeded",
    )


def test_trace_writer_never_serializes_secrets_or_authorization_headers(tmp_path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    TraceWriter(trace_path).append(event())

    serialized = trace_path.read_text(encoding="utf-8")
    assert "test-secret-api-key" not in serialized
    assert "Authorization" not in serialized
    assert "Bearer " not in serialized
    assert "raw_response" not in serialized


def test_trace_writer_appends_one_valid_json_object_per_event(tmp_path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    writer = TraceWriter(trace_path)

    writer.append(event("run-001"))
    writer.append(event("run-002"))

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["run_id"] for line in lines] == ["run-001", "run-002"]
