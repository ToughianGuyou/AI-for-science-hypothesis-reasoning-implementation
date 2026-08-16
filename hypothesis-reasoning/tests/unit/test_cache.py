from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path

import pytest
from pydantic import ValidationError

from hypothesis_reasoning.errors import ModelChangedError
from hypothesis_reasoning.llm.cache import BatchModelRegistry, ResponseCache, build_cache_key
from hypothesis_reasoning.llm.types import ChatMessage, LLMRequest, LLMResponse
from hypothesis_reasoning.models import Usage


def request(**overrides: object) -> LLMRequest:
    values: dict[str, object] = {
        "requested_model": "qwen-plus",
        "messages": (ChatMessage(role="user", content="Return JSON."),),
        "seed": 11,
        "temperature": 0.2,
        "prompt_version": "prompt-v1",
        "input_hashes": {"case": "abc", "topic": "def"},
        "budget_partition": "development",
        "estimated_cost_cny": "0.01",
        "run_id": "run-001",
        "case_id": "case-001",
        "stage": "generation",
        "experiment_batch": "batch-001",
    }
    values.update(overrides)
    return LLMRequest.model_validate(values)


def response() -> LLMResponse:
    return LLMResponse(
        request_id="chatcmpl-001",
        requested_model="qwen-plus",
        returned_model="qwen-plus-2026-08-01",
        content_hash="a" * 64,
        parsed_json={"answer": 42},
        called_at=datetime(2026, 8, 15, tzinfo=UTC),
        usage=Usage(
            prompt_tokens=120,
            completion_tokens=30,
            total_tokens=150,
            estimated_cost_cny=0.00042,
            latency_ms=25,
        ),
        retries=0,
    )


def _bind_model_in_subprocess(
    path: str,
    model: str,
    ready_queue,
    start_event,
    result_queue,
) -> None:
    original_load = BatchModelRegistry._load

    def slow_initial_load(registry: BatchModelRegistry) -> dict[str, dict[str, str]]:
        bindings = original_load(registry)
        if not bindings:
            time.sleep(0.1)
        return bindings

    BatchModelRegistry._load = slow_initial_load
    registry = BatchModelRegistry(Path(path))
    ready_queue.put("ready")
    start_event.wait()
    try:
        registry.bind("batch-001", "qwen-plus", model)
    except ModelChangedError:
        result_queue.put("changed")
    else:
        result_queue.put("bound")


@pytest.mark.parametrize("field", ["repair_request_id", "repair_returned_model"])
def test_llm_response_rejects_partial_repair_metadata(field: str) -> None:
    payload = response().model_dump(mode="python")
    payload[field] = "repair-value"

    with pytest.raises(ValidationError, match="Repair status"):
        LLMResponse.model_validate(payload)


def test_cache_key_is_order_independent_and_sensitive_to_request_behavior() -> None:
    first = request(input_hashes={"case": "abc", "topic": "def"})
    reordered = request(input_hashes={"topic": "def", "case": "abc"})
    changed_seed = request(seed=23)

    assert build_cache_key(first) == build_cache_key(reordered)
    assert len(build_cache_key(first)) == 64
    assert build_cache_key(first) != build_cache_key(changed_seed)


def test_response_cache_round_trips_only_parsed_metadata(tmp_path) -> None:
    cache = ResponseCache(tmp_path)
    cache_key = build_cache_key(request())

    cache.put(cache_key, response())

    cached = cache.get(cache_key)
    assert cached == response()
    serialized = (tmp_path / f"{cache_key}.json").read_text(encoding="utf-8")
    assert "qwen-plus-2026-08-01" in serialized
    assert "raw_response" not in serialized


def test_response_cache_treats_invalid_entries_as_misses(tmp_path) -> None:
    cache = ResponseCache(tmp_path)
    cache_key = build_cache_key(request())
    (tmp_path / f"{cache_key}.json").write_text("{not json", encoding="utf-8")

    assert cache.get(cache_key) is None


def test_batch_model_registry_persists_bindings_across_instances(tmp_path) -> None:
    path = tmp_path / "batch-models.json"
    BatchModelRegistry(path).bind("batch-001", "qwen-plus", "qwen-plus-build-a")

    restarted = BatchModelRegistry(path)

    restarted.bind("batch-001", "qwen-plus", "qwen-plus-build-a")
    with pytest.raises(ModelChangedError):
        restarted.bind("batch-001", "qwen-plus", "qwen-plus-build-b")


def test_batch_model_registry_serializes_concurrent_instances(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "concurrent-batch-models.json"
    first = BatchModelRegistry(path)
    second = BatchModelRegistry(path)
    original_load = BatchModelRegistry._load

    def slow_initial_load(registry: BatchModelRegistry) -> dict[str, dict[str, str]]:
        bindings = original_load(registry)
        if not bindings:
            time.sleep(0.05)
        return bindings

    monkeypatch.setattr(BatchModelRegistry, "_load", slow_initial_load)

    def bind(registry: BatchModelRegistry, model: str) -> str:
        try:
            registry.bind("batch-001", "qwen-plus", model)
        except ModelChangedError:
            return "changed"
        return "bound"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda item: bind(*item),
                ((first, "qwen-plus-build-a"), (second, "qwen-plus-build-b")),
            )
        )

    assert sorted(results) == ["bound", "changed"]


def test_batch_model_registry_serializes_competing_processes(tmp_path: Path) -> None:
    path = tmp_path / "process-batch-models.json"
    context = get_context("spawn")
    ready_queue = context.Queue()
    result_queue = context.Queue()
    start_event = context.Event()
    processes = [
        context.Process(
            target=_bind_model_in_subprocess,
            args=(str(path), model, ready_queue, start_event, result_queue),
        )
        for model in ("qwen-plus-build-a", "qwen-plus-build-b")
    ]

    for process in processes:
        process.start()
    assert [ready_queue.get(timeout=10) for _ in processes] == ["ready", "ready"]
    start_event.set()
    results = [result_queue.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert sorted(results) == ["bound", "changed"]
