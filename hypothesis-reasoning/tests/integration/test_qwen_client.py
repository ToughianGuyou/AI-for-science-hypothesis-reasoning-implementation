from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from pydantic import BaseModel

import hypothesis_reasoning.llm.qwen_client as qwen_client_module
from hypothesis_reasoning.errors import (
    BudgetExceededError,
    GatewayConfigurationError,
    GatewayResponseError,
    ModelChangedError,
    PricingConfigurationError,
    StructuredOutputError,
)
from hypothesis_reasoning.llm.budget import BudgetLedger
from hypothesis_reasoning.llm.cache import BatchModelRegistry, ResponseCache
from hypothesis_reasoning.llm.qwen_client import QwenClient
from hypothesis_reasoning.llm.types import ChatMessage, LLMRequest, ModelProfile, PriceTier
from hypothesis_reasoning.tracing import TraceWriter

BASE_URL = "https://workspace.example.invalid/compatible-mode/v1"
CHAT_URL = f"{BASE_URL}/chat/completions"


class Answer(BaseModel):
    answer: int


def model_profile(alias: str) -> ModelProfile:
    return ModelProfile(
        requested_alias=alias,
        pricing_checked_on=date(2026, 8, 15),
        input_token_tiers=(
            PriceTier(
                max_input_tokens=None,
                input_cny_per_unit=Decimal("2"),
                output_cny_per_unit=Decimal("6"),
            ),
        ),
    )


def valid_request(**overrides: Any) -> LLMRequest:
    values: dict[str, Any] = {
        "requested_model": "qwen-plus",
        "messages": (
            ChatMessage(role="system", content="Return one JSON object."),
            ChatMessage(role="user", content="Give the integer answer in JSON."),
        ),
        "seed": 11,
        "temperature": 0.2,
        "prompt_version": "prompt-v1",
        "input_hashes": {"case": "abc", "topic": "def"},
        "response_model": Answer,
        "budget_partition": "development",
        "estimated_cost_cny": Decimal("0.01"),
        "run_id": "run-001",
        "case_id": "case-001",
        "stage": "generation",
        "experiment_batch": "batch-001",
    }
    values.update(overrides)
    return LLMRequest.model_validate(values)


def fake_qwen_response(
    *,
    content: str = '{"answer": 42}',
    request_id: str = "chatcmpl-001",
    model: str | None = "qwen-plus-2026-08-01",
    prompt_tokens: int = 120,
    completion_tokens: int = 30,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": request_id,
        "object": "chat.completion",
        "created": 1786723200,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    if model is not None:
        payload["model"] = model
    return payload


@pytest.fixture
def ledger() -> BudgetLedger:
    return BudgetLedger({"development": Decimal("30")})


def make_client(
    tmp_path: Path,
    ledger: BudgetLedger,
    *,
    cache: ResponseCache | None = None,
    batch_registry: BatchModelRegistry | None = None,
    trace_name: str = "trace.jsonl",
    profiles: dict[str, ModelProfile] | None = None,
    date_provider: Callable[[], date] = lambda: date(2026, 8, 15),
    http_client: httpx.Client | None = None,
) -> QwenClient:
    return QwenClient(
        api_key="test-secret-api-key",
        base_url=BASE_URL,
        profiles=profiles
        or {
            "qwen-plus": model_profile("qwen-plus"),
            "qwen-turbo": model_profile("qwen-turbo"),
        },
        repair_model="qwen-turbo",
        budget=ledger,
        cache=cache or ResponseCache(tmp_path / "cache"),
        batch_registry=batch_registry or BatchModelRegistry(tmp_path / "batch-models.json"),
        trace_writer=TraceWriter(tmp_path / trace_name),
        date_provider=date_provider,
        sleeper=lambda _: None,
        http_client=http_client,
    )


@pytest.fixture
def client(tmp_path: Path, ledger: BudgetLedger) -> QwenClient:
    return make_client(tmp_path, ledger)


@respx.mock
def test_qwen_client_requests_json_mode_and_records_usage(
    client: QwenClient, ledger: BudgetLedger, tmp_path: Path
) -> None:
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=fake_qwen_response())
    )

    response = client.complete_json(valid_request())

    sent = json.loads(route.calls[0].request.content)
    assert response.usage.prompt_tokens == 120
    assert response.parsed_json == {"answer": 42}
    assert route.calls[0].request.headers["Authorization"].startswith("Bearer ")
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["seed"] == 11
    assert sent["enable_thinking"] is False
    assert "max_tokens" not in sent
    assert ledger.spent("development") == Decimal("0.000420")
    trace_event = json.loads(
        (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert trace_event["prompt_version"] == "prompt-v1"
    assert trace_event["input_hashes"] == {"case": "abc", "topic": "def"}
    assert trace_event["parameters"] == {
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
    }


@respx.mock
def test_qwen_client_persists_adapter_adaptations_in_trace(
    client: QwenClient,
    tmp_path: Path,
) -> None:
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=fake_qwen_response()))
    adaptations = (
        {
            "kind": "host_direction_removed",
            "detail": "Removed host direction at source line 7.",
        },
        {
            "kind": "json_contract_added",
            "detail": "Added the standard HypothesisBatch JSON Schema contract.",
        },
    )

    client.complete_json(valid_request(adaptations=adaptations))

    trace_event = json.loads(
        (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert trace_event["adaptations"] == list(adaptations)


@respx.mock
def test_qwen_client_uses_cache_without_network_or_duplicate_spend(
    client: QwenClient, ledger: BudgetLedger, tmp_path: Path
) -> None:
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(429, json={"error": {"message": "limited"}}),
            httpx.Response(200, json=fake_qwen_response()),
        ]
    )
    request = valid_request()

    first = client.complete_json(request)
    second = client.complete_json(request)

    assert route.call_count == 2
    assert first.usage.cache_hit is False
    assert second.usage.cache_hit is True
    assert second.retries == 0
    assert second.usage.estimated_cost_cny == 0.0
    assert ledger.spent("development") == Decimal("0.000420")
    trace_events = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert trace_events[0]["retries"] == 1
    assert trace_events[1]["retries"] == 0


@respx.mock
def test_qwen_client_treats_schema_invalid_cached_output_as_a_miss(
    tmp_path: Path,
) -> None:
    cache = ResponseCache(tmp_path / "schema-cache")
    registry = BatchModelRegistry(tmp_path / "schema-batch-models.json")
    first_client = make_client(
        tmp_path,
        BudgetLedger({"development": Decimal("30")}),
        cache=cache,
        batch_registry=registry,
        trace_name="schema-first-trace.jsonl",
    )
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, json=fake_qwen_response(content='{"answer": 42}')),
            httpx.Response(200, json=fake_qwen_response(content='{"answer": 7}')),
        ]
    )
    request = valid_request()
    first_client.complete_json(request)
    cache_file = next((tmp_path / "schema-cache").glob("*.json"))
    cached_payload = json.loads(cache_file.read_text(encoding="utf-8"))
    cached_payload["parsed_json"] = {"answer": "not-an-integer"}
    cache_file.write_text(json.dumps(cached_payload), encoding="utf-8")
    restarted = make_client(
        tmp_path,
        BudgetLedger({"development": Decimal("30")}),
        cache=cache,
        batch_registry=registry,
        trace_name="schema-second-trace.jsonl",
    )

    response = restarted.complete_json(request)

    assert response.parsed_json == {"answer": 7}
    assert response.usage.cache_hit is False
    assert route.call_count == 2


@respx.mock
def test_qwen_client_retries_retryable_statuses_only(client: QwenClient) -> None:
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(429, json={"error": {"message": "limited"}}),
            httpx.Response(200, json=fake_qwen_response()),
        ]
    )

    response = client.complete_json(valid_request())

    assert route.call_count == 2
    assert response.retries == 1


@respx.mock
def test_qwen_client_retries_transport_errors(client: QwenClient) -> None:
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.ConnectError(
                "connection interrupted",
                request=httpx.Request("POST", CHAT_URL),
            ),
            httpx.Response(200, json=fake_qwen_response()),
        ]
    )

    response = client.complete_json(valid_request())

    assert route.call_count == 2
    assert response.retries == 1


@respx.mock
def test_qwen_client_does_not_retry_non_retryable_status(client: QwenClient) -> None:
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(400, json={"error": {"message": "bad request"}})
    )

    with pytest.raises(GatewayResponseError):
        client.complete_json(valid_request())

    assert route.call_count == 1


@respx.mock
def test_qwen_client_stops_after_three_retryable_failures(
    client: QwenClient,
    tmp_path: Path,
) -> None:
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(500, json={"error": {"message": "unavailable"}})
    )

    with pytest.raises(GatewayResponseError):
        client.complete_json(valid_request())

    assert route.call_count == 3
    trace_event = json.loads(
        (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert trace_event["retries"] == 2


@respx.mock
def test_qwen_client_repairs_invalid_json_exactly_once(client: QwenClient) -> None:
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, json=fake_qwen_response(content="not-json")),
            httpx.Response(
                200,
                json=fake_qwen_response(
                    content='{ "answer": 7 }',
                    request_id="chatcmpl-repair",
                    model="qwen-turbo-2026-08-01",
                    prompt_tokens=60,
                    completion_tokens=10,
                ),
            ),
        ]
    )

    response = client.complete_json(valid_request())

    repair_payload = json.loads(route.calls[1].request.content)
    assert route.call_count == 2
    assert repair_payload["model"] == "qwen-turbo"
    assert response.parsed_json == {"answer": 7}
    assert response.repaired is True
    assert response.repair_returned_model == "qwen-turbo-2026-08-01"


@respx.mock
@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_qwen_client_repairs_non_finite_json_constants(
    client: QwenClient,
    constant: str,
) -> None:
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                json=fake_qwen_response(content=f'{{"value": {constant}}}'),
            ),
            httpx.Response(
                200,
                json=fake_qwen_response(
                    content='{"value": 1.5}',
                    request_id="chatcmpl-repair",
                    model="qwen-turbo-2026-08-01",
                ),
            ),
        ]
    )

    response = client.complete_json(valid_request(response_model=None))

    assert response.parsed_json == {"value": 1.5}
    assert response.repaired is True
    assert route.call_count == 2


@respx.mock
def test_qwen_client_repairs_schema_failure_exactly_once(client: QwenClient) -> None:
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, json=fake_qwen_response(content='{"answer": "wrong"}')),
            httpx.Response(
                200,
                json=fake_qwen_response(
                    content='{"answer": 9}',
                    request_id="chatcmpl-repair",
                    model="qwen-turbo-2026-08-01",
                ),
            ),
        ]
    )

    response = client.complete_json(valid_request())

    assert route.call_count == 2
    assert response.parsed_json == {"answer": 9}


@respx.mock
def test_qwen_client_traces_repair_budget_preflight_failure(tmp_path: Path) -> None:
    ledger = BudgetLedger({"development": Decimal("0.0005")})
    client = make_client(tmp_path, ledger)
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=fake_qwen_response(content="not-json"))
    )

    with pytest.raises(BudgetExceededError):
        client.complete_json(
            valid_request(estimated_cost_cny=Decimal("0.0002"))
        )

    assert route.call_count == 1
    trace_events = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(trace_events) == 2
    assert trace_events[1]["stage"] == "generation:repair"
    assert trace_events[1]["requested_model"] == "qwen-turbo"
    assert trace_events[1]["status"] == "failed"
    assert "budget" in trace_events[1]["error_summary"].lower()


@respx.mock
def test_qwen_client_preserves_original_output_error_when_repair_fails(
    client: QwenClient,
    tmp_path: Path,
) -> None:
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, json=fake_qwen_response(content="not-json")),
            httpx.Response(
                200,
                json=fake_qwen_response(
                    content="still-not-json",
                    request_id="chatcmpl-repair",
                    model="qwen-turbo-2026-08-01",
                ),
            ),
        ]
    )

    with pytest.raises(StructuredOutputError, match="original response"):
        client.complete_json(valid_request())

    assert route.call_count == 2
    trace_events = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert trace_events[1]["requested_model"] == "qwen-turbo"
    assert trace_events[1]["returned_model"] == "qwen-turbo-2026-08-01"
    assert trace_events[1]["request_id"] == "chatcmpl-repair"
    assert trace_events[1]["usage"]["total_tokens"] == 150


@respx.mock
def test_qwen_client_accounts_for_billable_response_missing_model_identifier(
    client: QwenClient,
    ledger: BudgetLedger,
    tmp_path: Path,
) -> None:
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=fake_qwen_response(model=None))
    )

    with pytest.raises(GatewayResponseError, match="model identifier"):
        client.complete_json(valid_request())

    assert route.call_count == 1
    assert ledger.spent("development") == Decimal("0.000420")
    trace_event = json.loads(
        (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert trace_event["request_id"] == "chatcmpl-001"
    assert trace_event["returned_model"] is None
    assert trace_event["usage"]["total_tokens"] == 150


@respx.mock
def test_qwen_client_accounts_for_billable_response_missing_total_tokens(
    client: QwenClient,
    ledger: BudgetLedger,
    tmp_path: Path,
) -> None:
    payload = fake_qwen_response()
    del payload["usage"]["total_tokens"]
    route = respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=payload))

    with pytest.raises(GatewayResponseError, match="total_tokens"):
        client.complete_json(valid_request())

    assert route.call_count == 1
    assert ledger.spent("development") == Decimal("0.000420")
    trace_event = json.loads(
        (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert trace_event["usage"]["prompt_tokens"] == 120
    assert trace_event["usage"]["completion_tokens"] == 30
    assert trace_event["usage"]["total_tokens"] == 150


@respx.mock
def test_billable_malformed_primary_response_still_locks_batch_model(
    client: QwenClient,
) -> None:
    malformed = fake_qwen_response(model="qwen-plus-build-a")
    del malformed["usage"]["total_tokens"]
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, json=malformed),
            httpx.Response(
                200,
                json=fake_qwen_response(model="qwen-plus-build-b"),
            ),
        ]
    )

    with pytest.raises(GatewayResponseError, match="total_tokens"):
        client.complete_json(valid_request())
    with pytest.raises(ModelChangedError):
        client.complete_json(valid_request(seed=23, run_id="run-002"))

    assert route.call_count == 2


@respx.mock
def test_billable_malformed_repair_response_still_locks_batch_model(
    client: QwenClient,
) -> None:
    malformed_repair = fake_qwen_response(
        content='{"answer": 7}',
        request_id="chatcmpl-repair-a",
        model="qwen-turbo-build-a",
    )
    del malformed_repair["usage"]["total_tokens"]
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                json=fake_qwen_response(content="not-json", request_id="primary-a"),
            ),
            httpx.Response(200, json=malformed_repair),
            httpx.Response(
                200,
                json=fake_qwen_response(content="not-json", request_id="primary-b"),
            ),
            httpx.Response(
                200,
                json=fake_qwen_response(
                    content='{"answer": 8}',
                    request_id="chatcmpl-repair-b",
                    model="qwen-turbo-build-b",
                ),
            ),
        ]
    )

    with pytest.raises(StructuredOutputError, match="repair attempt failed"):
        client.complete_json(valid_request())
    with pytest.raises(ModelChangedError):
        client.complete_json(valid_request(seed=23, run_id="run-002"))

    assert route.call_count == 4


@respx.mock
def test_billable_malformed_primary_model_change_is_explicit_in_trace(
    tmp_path: Path,
) -> None:
    registry = BatchModelRegistry(tmp_path / "primary-change-batch-models.json")
    registry.bind("batch-001", "qwen-plus", "qwen-plus-build-a")
    client = make_client(
        tmp_path,
        BudgetLedger({"development": Decimal("30")}),
        batch_registry=registry,
    )
    malformed = fake_qwen_response(model="qwen-plus-build-b")
    del malformed["usage"]["total_tokens"]
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=malformed))

    with pytest.raises(ModelChangedError):
        client.complete_json(valid_request())

    trace_event = json.loads(
        (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert trace_event["returned_model"] == "qwen-plus-build-b"
    assert trace_event["usage"]["total_tokens"] == 150
    assert "Returned model changed" in trace_event["error_summary"]


@respx.mock
def test_billable_malformed_repair_model_change_is_explicit_in_trace(
    tmp_path: Path,
) -> None:
    registry = BatchModelRegistry(tmp_path / "repair-change-batch-models.json")
    registry.bind("batch-001", "qwen-turbo", "qwen-turbo-build-a")
    client = make_client(
        tmp_path,
        BudgetLedger({"development": Decimal("30")}),
        batch_registry=registry,
    )
    malformed_repair = fake_qwen_response(
        content='{"answer": 7}',
        request_id="chatcmpl-repair-b",
        model="qwen-turbo-build-b",
    )
    del malformed_repair["usage"]["total_tokens"]
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, json=fake_qwen_response(content="not-json")),
            httpx.Response(200, json=malformed_repair),
        ]
    )

    with pytest.raises(ModelChangedError):
        client.complete_json(valid_request())

    assert route.call_count == 2
    trace_events = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert trace_events[1]["stage"] == "generation:repair"
    assert trace_events[1]["returned_model"] == "qwen-turbo-build-b"
    assert trace_events[1]["usage"]["total_tokens"] == 150
    assert "Returned model changed" in trace_events[1]["error_summary"]


@respx.mock
def test_qwen_client_preserves_retry_count_for_malformed_2xx_envelope(
    client: QwenClient,
    tmp_path: Path,
) -> None:
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(429, json={"error": {"message": "limited"}}),
            httpx.Response(200, text="not-json"),
        ]
    )

    with pytest.raises(GatewayResponseError, match="non-JSON response envelope"):
        client.complete_json(valid_request())

    assert route.call_count == 2
    trace_event = json.loads(
        (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert trace_event["retries"] == 1


@respx.mock
def test_qwen_client_preserves_retry_count_for_post_retry_pricing_failure(
    tmp_path: Path,
) -> None:
    limited_profiles = {
        alias: ModelProfile(
            requested_alias=alias,
            pricing_checked_on=date(2026, 8, 15),
            input_token_tiers=(
                PriceTier(
                    max_input_tokens=100,
                    input_cny_per_unit=Decimal("2"),
                    output_cny_per_unit=Decimal("6"),
                ),
            ),
        )
        for alias in ("qwen-plus", "qwen-turbo")
    }
    client = make_client(
        tmp_path,
        BudgetLedger({"development": Decimal("30")}),
        profiles=limited_profiles,
    )
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(429, json={"error": {"message": "limited"}}),
            httpx.Response(200, json=fake_qwen_response(prompt_tokens=120)),
        ]
    )

    with pytest.raises(PricingConfigurationError, match="No price tier"):
        client.complete_json(valid_request())

    assert route.call_count == 2
    trace_event = json.loads(
        (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert trace_event["retries"] == 1


@respx.mock
def test_qwen_client_stops_batch_when_returned_model_changes(client: QwenClient) -> None:
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, json=fake_qwen_response(model="qwen-plus-build-a")),
            httpx.Response(200, json=fake_qwen_response(model="qwen-plus-build-b")),
        ]
    )

    client.complete_json(valid_request(seed=11))
    with pytest.raises(ModelChangedError):
        client.complete_json(valid_request(seed=23, run_id="run-002"))

    assert route.call_count == 2


@respx.mock
def test_cache_hit_restores_batch_model_lock_after_client_restart(tmp_path: Path) -> None:
    shared_cache = ResponseCache(tmp_path / "shared-cache")
    first_client = make_client(
        tmp_path,
        BudgetLedger({"development": Decimal("30")}),
        cache=shared_cache,
        trace_name="first-trace.jsonl",
    )
    second_client = make_client(
        tmp_path,
        BudgetLedger({"development": Decimal("30")}),
        cache=shared_cache,
        trace_name="second-trace.jsonl",
    )
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, json=fake_qwen_response(model="qwen-plus-build-a")),
            httpx.Response(200, json=fake_qwen_response(model="qwen-plus-build-b")),
        ]
    )

    first_client.complete_json(valid_request(seed=11))
    cached = second_client.complete_json(valid_request(seed=11))

    assert cached.usage.cache_hit is True
    with pytest.raises(ModelChangedError):
        second_client.complete_json(valid_request(seed=23, run_id="run-002"))
    assert route.call_count == 2


@respx.mock
def test_persisted_batch_lock_rejects_changed_model_without_cache_replay(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "persistent-batch-models.json"
    first_client = make_client(
        tmp_path,
        BudgetLedger({"development": Decimal("30")}),
        cache=ResponseCache(tmp_path / "first-cache"),
        batch_registry=BatchModelRegistry(registry_path),
        trace_name="first-direct-trace.jsonl",
    )
    second_client = make_client(
        tmp_path,
        BudgetLedger({"development": Decimal("30")}),
        cache=ResponseCache(tmp_path / "second-cache"),
        batch_registry=BatchModelRegistry(registry_path),
        trace_name="second-direct-trace.jsonl",
    )
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, json=fake_qwen_response(model="qwen-plus-build-a")),
            httpx.Response(200, json=fake_qwen_response(model="qwen-plus-build-b")),
        ]
    )

    first_client.complete_json(valid_request(seed=11))

    with pytest.raises(ModelChangedError):
        second_client.complete_json(valid_request(seed=23, run_id="run-002"))
    assert route.call_count == 2


@respx.mock
def test_qwen_client_rejects_budget_before_network(
    tmp_path: Path,
) -> None:
    client = QwenClient(
        api_key="test-secret-api-key",
        base_url=BASE_URL,
        profiles={
            "qwen-plus": model_profile("qwen-plus"),
            "qwen-turbo": model_profile("qwen-turbo"),
        },
        repair_model="qwen-turbo",
        budget=BudgetLedger({"development": Decimal("0.001")}),
        cache=ResponseCache(tmp_path / "cache"),
        batch_registry=BatchModelRegistry(tmp_path / "budget-batch-models.json"),
        trace_writer=TraceWriter(tmp_path / "trace.jsonl"),
        date_provider=lambda: date(2026, 8, 15),
    )
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=fake_qwen_response())
    )

    with pytest.raises(BudgetExceededError):
        client.complete_json(valid_request(estimated_cost_cny=Decimal("0.01")))

    assert route.call_count == 0


@respx.mock
def test_qwen_client_records_actual_cost_that_exceeds_reserved_estimate(
    tmp_path: Path,
) -> None:
    ledger = BudgetLedger({"development": Decimal("0.0003")})
    client = make_client(tmp_path, ledger)
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=fake_qwen_response())
    )

    response = client.complete_json(
        valid_request(estimated_cost_cny=Decimal("0.0002"))
    )

    assert response.usage.estimated_cost_cny == 0.00042
    assert ledger.spent("development") == Decimal("0.000420")
    with pytest.raises(BudgetExceededError):
        ledger.assert_can_spend("development", Decimal("0"))
    assert route.call_count == 1


def test_qwen_client_rejects_shared_or_non_https_endpoints(tmp_path: Path) -> None:
    common = {
        "api_key": "test-secret-api-key",
        "profiles": {
            "qwen-plus": model_profile("qwen-plus"),
            "qwen-turbo": model_profile("qwen-turbo"),
        },
        "repair_model": "qwen-turbo",
        "budget": BudgetLedger({"development": Decimal("30")}),
        "cache": ResponseCache(tmp_path / "cache"),
        "batch_registry": BatchModelRegistry(tmp_path / "endpoint-batch-models.json"),
        "date_provider": lambda: date(2026, 8, 15),
    }

    with pytest.raises(GatewayConfigurationError):
        QwenClient(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", **common
        )
    with pytest.raises(GatewayConfigurationError):
        QwenClient(
            base_url="http://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
            **common,
        )


def test_qwen_client_context_closes_internally_owned_http_client(
    tmp_path: Path,
    ledger: BudgetLedger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned_client = httpx.Client()
    monkeypatch.setattr(qwen_client_module.httpx, "Client", lambda: owned_client)

    with make_client(tmp_path, ledger):
        assert owned_client.is_closed is False

    assert owned_client.is_closed is True


def test_qwen_client_close_preserves_externally_owned_http_client(
    tmp_path: Path,
    ledger: BudgetLedger,
) -> None:
    external_client = httpx.Client()
    try:
        client = make_client(tmp_path, ledger, http_client=external_client)

        client.close()

        assert external_client.is_closed is False
    finally:
        external_client.close()


@respx.mock
def test_qwen_client_rejects_json_mode_without_json_in_messages(client: QwenClient) -> None:
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=fake_qwen_response())
    )
    request = valid_request(
        messages=(ChatMessage(role="user", content="Give the integer answer."),)
    )

    with pytest.raises(GatewayConfigurationError, match="JSON"):
        client.complete_json(request)

    assert route.call_count == 0


@respx.mock
def test_qwen_client_uses_current_date_provider_and_traces_stale_pricing(
    tmp_path: Path,
) -> None:
    stale_profiles = {
        alias: ModelProfile(
            requested_alias=alias,
            pricing_checked_on=date(2026, 8, 7),
            input_token_tiers=(
                PriceTier(
                    max_input_tokens=None,
                    input_cny_per_unit=Decimal("2"),
                    output_cny_per_unit=Decimal("6"),
                ),
            ),
        )
        for alias in ("qwen-plus", "qwen-turbo")
    }
    client = make_client(
        tmp_path,
        BudgetLedger({"development": Decimal("30")}),
        profiles=stale_profiles,
        date_provider=lambda: date(2026, 8, 15),
        trace_name="stale-pricing-trace.jsonl",
    )
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=fake_qwen_response())
    )

    with pytest.raises(PricingConfigurationError):
        client.complete_json(valid_request())

    assert route.call_count == 0
    trace_event = json.loads(
        (tmp_path / "stale-pricing-trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert trace_event["status"] == "failed"
    assert trace_event["error_summary"] == "Pricing is older than seven days for qwen-plus"


class SecretBearingBudgetLedger(BudgetLedger):
    def assert_can_spend(self, partition: str, estimated_cny: Decimal) -> None:
        raise GatewayConfigurationError(
            "Authorization: Bearer test-secret-api-key; key=test-secret-api-key"
        )


@respx.mock
def test_qwen_client_redacts_secret_bearing_failures_before_tracing(tmp_path: Path) -> None:
    client = make_client(
        tmp_path,
        SecretBearingBudgetLedger({"development": Decimal("30")}),
        trace_name="secret-failure-trace.jsonl",
    )
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=fake_qwen_response())
    )

    with pytest.raises(GatewayConfigurationError):
        client.complete_json(valid_request())

    serialized = (tmp_path / "secret-failure-trace.jsonl").read_text(encoding="utf-8")
    assert route.call_count == 0
    assert "test-secret-api-key" not in serialized
    assert "Bearer [REDACTED]" in serialized
