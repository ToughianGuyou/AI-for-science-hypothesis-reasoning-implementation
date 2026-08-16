"""Synchronous OpenAI-compatible Qwen JSON gateway with hard local controls."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from types import TracebackType
from typing import Any, Literal, NoReturn
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from hypothesis_reasoning.errors import (
    GatewayConfigurationError,
    GatewayResponseError,
    ModelChangedError,
    PricingConfigurationError,
    StructuredOutputError,
)
from hypothesis_reasoning.llm.budget import BudgetLedger
from hypothesis_reasoning.llm.cache import BatchModelRegistry, ResponseCache, build_cache_key
from hypothesis_reasoning.llm.types import (
    ChatMessage,
    LLMRequest,
    LLMResponse,
    ModelProfile,
    TraceEvent,
)
from hypothesis_reasoning.models import Usage
from hypothesis_reasoning.tracing import TraceWriter

_RETRYABLE_STATUS_CODES = {408, 429}
_SHARED_HOSTS = {
    "dashscope.aliyuncs.com",
    "dashscope-intl.aliyuncs.com",
    "coding.dashscope.aliyuncs.com",
}
_DISALLOWED_DEDICATED_PREFIXES = {"trial", "token-plan", "coding"}


@dataclass(frozen=True, slots=True)
class _RawCompletion:
    request_id: str
    requested_model: str
    returned_model: str
    content: str
    content_hash: str
    called_at: datetime
    usage: Usage
    cost_cny: Decimal
    retries: int


class _BillableResponseError(GatewayResponseError):
    """A failed 2xx response whose trustworthy usage has already been settled."""

    def __init__(
        self,
        message: str,
        *,
        requested_model: str,
        request_id: str | None,
        returned_model: str | None,
        content_hash: str | None,
        usage: Usage,
        retries: int,
    ) -> None:
        super().__init__(message)
        self.requested_model = requested_model
        self.request_id = request_id
        self.returned_model = returned_model
        self.content_hash = content_hash
        self.usage = usage
        self.retries = retries


class _GatewayCallError(GatewayResponseError):
    """A pre-envelope call failure with an accurate retry count."""

    def __init__(self, message: str, *, retries: int) -> None:
        super().__init__(message)
        self.retries = retries


class _GatewayPricingError(PricingConfigurationError):
    """A post-response pricing failure with an accurate retry count."""

    def __init__(self, message: str, *, retries: int) -> None:
        super().__init__(message)
        self.retries = retries


class QwenClient:
    """Call a Workspace Qwen endpoint while enforcing reproducibility and cost gates."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        profiles: Mapping[str, ModelProfile],
        repair_model: str,
        budget: BudgetLedger,
        cache: ResponseCache,
        batch_registry: BatchModelRegistry,
        trace_writer: TraceWriter | None = None,
        max_attempts: int = 3,
        timeout_seconds: float = 60.0,
        date_provider: Callable[[], date] = date.today,
        sleeper: Callable[[float], object] = time.sleep,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key.strip():
            raise GatewayConfigurationError("DASHSCOPE_API_KEY is required")
        if max_attempts < 1 or max_attempts > 3:
            raise GatewayConfigurationError("Qwen calls allow between one and three attempts")
        if timeout_seconds <= 0:
            raise GatewayConfigurationError("Request timeout must be positive")
        normalized_base_url = _validate_workspace_base_url(base_url)
        normalized_profiles = dict(profiles)
        if repair_model not in normalized_profiles:
            raise GatewayConfigurationError(
                f"Repair model profile is not configured: {repair_model!r}"
            )
        for alias, profile in normalized_profiles.items():
            if alias != profile.requested_alias:
                raise GatewayConfigurationError(
                    f"Model profile key {alias!r} does not match {profile.requested_alias!r}"
                )

        self._api_key = api_key
        self._base_url = normalized_base_url
        self._endpoint = f"{normalized_base_url}/chat/completions"
        self._endpoint_fingerprint = hashlib.sha256(
            normalized_base_url.encode("utf-8")
        ).hexdigest()
        self._profiles = normalized_profiles
        self._repair_model = repair_model
        self._budget = budget
        self._cache = cache
        self._batch_registry = batch_registry
        self._date_provider = date_provider
        self._trace_writer = trace_writer
        self._max_attempts = max_attempts
        self._timeout_seconds = timeout_seconds
        self._sleeper = sleeper
        self._owns_http_client = http_client is None
        self._http_client = http_client if http_client is not None else httpx.Client()

    def close(self) -> None:
        """Close the internally owned HTTP connection pool."""

        if self._owns_http_client:
            self._http_client.close()

    def __enter__(self) -> QwenClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        """Return one validated JSON object, using at most one repair completion."""

        try:
            self._assert_json_instruction(request.messages)
        except Exception as error:
            self._trace_request_failure(request, error)
            raise
        cache_key = build_cache_key(request)
        cached = self._validated_cached_response(request, self._cache.get(cache_key))
        if cached is not None:
            try:
                self._remember_returned_model(
                    request.experiment_batch,
                    request.requested_model,
                    cached.returned_model,
                )
                if cached.repair_returned_model is not None:
                    self._remember_returned_model(
                        request.experiment_batch,
                        self._repair_model,
                        cached.repair_returned_model,
                    )
            except ModelChangedError as error:
                self._trace_request_failure(request, error)
                raise
            hit = self._as_cache_hit(cached)
            self._trace_cache_hit(request, hit)
            return hit

        try:
            self._profile(request.requested_model).assert_pricing_current(
                as_of=self._date_provider()
            )
            self._budget.assert_can_spend(
                request.budget_partition,
                request.estimated_cost_cny,
            )
        except Exception as error:
            self._trace_request_failure(request, error)
            raise
        try:
            primary = self._call_api(
                requested_model=request.requested_model,
                messages=request.messages,
                seed=request.seed,
                temperature=request.temperature,
                enable_thinking=request.enable_thinking,
                budget_partition=request.budget_partition,
                estimated_cost_cny=request.estimated_cost_cny,
            )
        except _BillableResponseError as error:
            try:
                self._remember_billable_returned_model(request, error)
            except ModelChangedError as model_error:
                self._trace_request_failure(
                    request,
                    error,
                    summary_error=model_error,
                )
                raise
            self._trace_request_failure(request, error)
            raise
        except Exception as error:
            self._trace_request_failure(request, error)
            raise

        try:
            self._remember_returned_model(
                request.experiment_batch,
                request.requested_model,
                primary.returned_model,
            )
        except ModelChangedError as error:
            self._trace_completion(request, primary, status="failed", error=error)
            raise

        try:
            parsed = self._parse_content(primary.content, request)
        except StructuredOutputError as original_error:
            self._trace_completion(request, primary, status="failed", error=original_error)
            return self._repair_once(request, primary, original_error, cache_key)

        self._trace_completion(request, primary, status="succeeded", parsed_json=parsed)
        response = LLMResponse(
            request_id=primary.request_id,
            requested_model=primary.requested_model,
            returned_model=primary.returned_model,
            content_hash=primary.content_hash,
            parsed_json=parsed,
            called_at=primary.called_at,
            usage=primary.usage,
            retries=primary.retries,
        )
        self._cache.put(cache_key, response)
        return response

    def _repair_once(
        self,
        request: LLMRequest,
        primary: _RawCompletion,
        original_error: StructuredOutputError,
        cache_key: str,
    ) -> LLMResponse:
        repair_stage = f"{request.stage}:repair"
        repair_prompt_version = f"{request.prompt_version}:repair"
        repair_input_hashes = {
            **request.input_hashes,
            "repair_source": primary.content_hash,
        }
        try:
            self._profile(self._repair_model).assert_pricing_current(
                as_of=self._date_provider()
            )
            self._budget.assert_can_spend(
                request.budget_partition,
                request.estimated_cost_cny,
            )
        except Exception as error:
            self._trace_request_failure(
                request,
                error,
                requested_model=self._repair_model,
                stage=repair_stage,
                temperature=0.0,
                enable_thinking=False,
                prompt_version=repair_prompt_version,
                input_hashes=repair_input_hashes,
            )
            raise
        repair_messages = self._repair_messages(primary.content, request)
        try:
            repair = self._call_api(
                requested_model=self._repair_model,
                messages=repair_messages,
                seed=request.seed,
                temperature=0.0,
                enable_thinking=False,
                budget_partition=request.budget_partition,
                estimated_cost_cny=request.estimated_cost_cny,
            )
        except _BillableResponseError as repair_call_error:
            try:
                self._remember_billable_returned_model(request, repair_call_error)
            except ModelChangedError as model_error:
                self._trace_request_failure(
                    request,
                    repair_call_error,
                    requested_model=self._repair_model,
                    stage=repair_stage,
                    temperature=0.0,
                    enable_thinking=False,
                    prompt_version=repair_prompt_version,
                    input_hashes=repair_input_hashes,
                    summary_error=model_error,
                )
                raise
            self._trace_request_failure(
                request,
                repair_call_error,
                requested_model=self._repair_model,
                stage=repair_stage,
                temperature=0.0,
                enable_thinking=False,
                prompt_version=repair_prompt_version,
                input_hashes=repair_input_hashes,
            )
            raise StructuredOutputError(
                "The original response was invalid and the single JSON repair attempt failed"
            ) from original_error
        except Exception as repair_call_error:
            self._trace_request_failure(
                request,
                repair_call_error,
                requested_model=self._repair_model,
                stage=repair_stage,
                temperature=0.0,
                enable_thinking=False,
                prompt_version=repair_prompt_version,
                input_hashes=repair_input_hashes,
            )
            raise StructuredOutputError(
                "The original response was invalid and the single JSON repair attempt failed"
            ) from original_error

        try:
            self._remember_returned_model(
                request.experiment_batch,
                self._repair_model,
                repair.returned_model,
            )
        except ModelChangedError as error:
            self._trace_completion(
                request,
                repair,
                status="failed",
                error=error,
                stage=repair_stage,
                temperature=0.0,
                enable_thinking=False,
                prompt_version=repair_prompt_version,
                input_hashes=repair_input_hashes,
            )
            raise

        try:
            parsed = self._parse_content(repair.content, request)
        except StructuredOutputError as repair_output_error:
            self._trace_completion(
                request,
                repair,
                status="failed",
                error=repair_output_error,
                stage=repair_stage,
                temperature=0.0,
                enable_thinking=False,
                prompt_version=repair_prompt_version,
                input_hashes=repair_input_hashes,
            )
            raise StructuredOutputError(
                "The original response was invalid and the single JSON repair attempt failed"
            ) from original_error

        self._trace_completion(
            request,
            repair,
            status="succeeded",
            parsed_json=parsed,
            stage=repair_stage,
            temperature=0.0,
            enable_thinking=False,
            prompt_version=repair_prompt_version,
            input_hashes=repair_input_hashes,
        )
        usage = _combine_usage(primary.usage, repair.usage)
        response = LLMResponse(
            request_id=primary.request_id,
            requested_model=primary.requested_model,
            returned_model=primary.returned_model,
            content_hash=repair.content_hash,
            parsed_json=parsed,
            called_at=primary.called_at,
            usage=usage,
            retries=primary.retries + repair.retries,
            repaired=True,
            repair_request_id=repair.request_id,
            repair_returned_model=repair.returned_model,
        )
        self._cache.put(cache_key, response)
        return response

    def _call_api(
        self,
        *,
        requested_model: str,
        messages: Sequence[ChatMessage],
        seed: int,
        temperature: float,
        enable_thinking: bool,
        budget_partition: str,
        estimated_cost_cny: Decimal,
    ) -> _RawCompletion:
        profile = self._profile(requested_model)
        reservation = self._budget.reserve(budget_partition, estimated_cost_cny)
        reservation_settled = False
        payload = {
            "model": requested_model,
            "messages": [message.model_dump(mode="json") for message in messages],
            "temperature": temperature,
            "seed": seed,
            "enable_thinking": enable_thinking,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            started = time.perf_counter()
            response: httpx.Response | None = None
            retries = 0
            for attempt in range(1, self._max_attempts + 1):
                try:
                    response = self._http_client.post(
                        self._endpoint,
                        headers=headers,
                        json=payload,
                        timeout=self._timeout_seconds,
                    )
                except httpx.TransportError as error:
                    if attempt == self._max_attempts:
                        raise _GatewayCallError(
                            f"Qwen transport failed after {attempt} attempts",
                            retries=retries,
                        ) from error
                    retries += 1
                    self._sleeper(float(2 ** (attempt - 1)))
                    continue

                if _is_retryable(response.status_code):
                    if attempt == self._max_attempts:
                        raise _GatewayCallError(
                            f"Qwen returned retryable HTTP {response.status_code} "
                            f"for all {attempt} attempts",
                            retries=retries,
                        )
                    retries += 1
                    self._sleeper(float(2 ** (attempt - 1)))
                    continue
                if response.status_code < 200 or response.status_code >= 300:
                    raise _GatewayCallError(
                        f"Qwen request failed with HTTP {response.status_code}",
                        retries=retries,
                    )
                break

            if response is None:
                raise _GatewayCallError(
                    "Qwen request produced no HTTP response",
                    retries=retries,
                )
            latency_ms = max(0, round((time.perf_counter() - started) * 1000))
            try:
                payload = _decode_response_payload(response)
                prompt_tokens, completion_tokens = _decode_cost_token_usage(payload)
            except GatewayResponseError as error:
                raise _GatewayCallError(str(error), retries=retries) from error
            try:
                cost = profile.calculate_cost(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            except PricingConfigurationError as error:
                raise _GatewayPricingError(str(error), retries=retries) from error
            total_tokens_error: GatewayResponseError | None = None
            try:
                total_tokens = _decode_total_tokens(payload)
            except GatewayResponseError as error:
                # Prompt and completion counts are sufficient to settle the bill.  Keep
                # the derived total only for the failure trace; the malformed envelope
                # must still be rejected after accounting.
                total_tokens = prompt_tokens + completion_tokens
                total_tokens_error = error
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost_cny=float(cost),
                latency_ms=latency_ms,
                cache_hit=False,
            )
            self._budget.settle(reservation, cost, usage)
            reservation_settled = True
            if total_tokens_error is not None:
                raise _BillableResponseError(
                    str(total_tokens_error),
                    requested_model=requested_model,
                    request_id=_optional_string(payload.get("id")),
                    returned_model=_optional_string(payload.get("model")),
                    content_hash=None,
                    usage=usage,
                    retries=retries,
                ) from total_tokens_error
            try:
                envelope = _decode_response_envelope(
                    payload,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
            except GatewayResponseError as error:
                observed_content = _optional_response_content(payload)
                raise _BillableResponseError(
                    str(error),
                    requested_model=requested_model,
                    request_id=_optional_string(payload.get("id")),
                    returned_model=_optional_string(payload.get("model")),
                    content_hash=(
                        hashlib.sha256(observed_content.encode("utf-8")).hexdigest()
                        if observed_content is not None
                        else None
                    ),
                    usage=usage,
                    retries=retries,
                ) from error
            return _RawCompletion(
                request_id=envelope.request_id,
                requested_model=requested_model,
                returned_model=envelope.returned_model,
                content=envelope.content,
                content_hash=hashlib.sha256(envelope.content.encode("utf-8")).hexdigest(),
                called_at=datetime.now(UTC),
                usage=usage,
                cost_cny=cost,
                retries=retries,
            )
        finally:
            if not reservation_settled:
                self._budget.cancel(reservation)

    def _parse_content(self, content: str, request: LLMRequest) -> dict[str, Any]:
        try:
            parsed = json.loads(content, parse_constant=_reject_non_finite_json)
            if not isinstance(parsed, dict):
                raise TypeError("JSON output must be an object")
            if request.response_model is None:
                return parsed
            validated = request.response_model.model_validate(parsed)
            return validated.model_dump(mode="json")
        except (ValueError, TypeError, ValidationError) as error:
            raise StructuredOutputError(
                "The original response was not a valid object for the requested JSON contract"
            ) from error

    def _repair_messages(
        self,
        content: str,
        request: LLMRequest,
    ) -> tuple[ChatMessage, ChatMessage]:
        schema = (
            request.response_model.model_json_schema()
            if request.response_model is not None
            else {"type": "object"}
        )
        return (
            ChatMessage(
                role="system",
                content=(
                    "Repair the supplied content. Return exactly one valid JSON object and "
                    "do not add facts."
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    "Return repaired JSON matching this schema: "
                    f"{json.dumps(schema, ensure_ascii=False, sort_keys=True)}\n"
                    f"Content to repair:\n{content}"
                ),
            ),
        )

    def _profile(self, alias: str) -> ModelProfile:
        try:
            return self._profiles[alias]
        except KeyError as error:
            raise GatewayConfigurationError(
                f"Requested model alias is not configured: {alias!r}"
            ) from error

    def _remember_returned_model(
        self,
        experiment_batch: str,
        requested_model: str,
        returned_model: str,
    ) -> None:
        self._batch_registry.bind(experiment_batch, requested_model, returned_model)

    def _remember_billable_returned_model(
        self,
        request: LLMRequest,
        error: _BillableResponseError,
    ) -> None:
        if error.returned_model is not None:
            self._remember_returned_model(
                request.experiment_batch,
                error.requested_model,
                error.returned_model,
            )

    @staticmethod
    def _assert_json_instruction(messages: Sequence[ChatMessage]) -> None:
        if not any("json" in message.content.casefold() for message in messages):
            raise GatewayConfigurationError(
                'JSON Object mode requires at least one message containing the word "JSON"'
            )

    def _as_cache_hit(self, response: LLMResponse) -> LLMResponse:
        usage = response.usage.model_copy(
            update={
                "estimated_cost_cny": 0.0,
                "latency_ms": 0,
                "cache_hit": True,
            }
        )
        return response.model_copy(update={"usage": usage, "retries": 0})

    @staticmethod
    def _validated_cached_response(
        request: LLMRequest,
        response: LLMResponse | None,
    ) -> LLMResponse | None:
        if response is None or response.requested_model != request.requested_model:
            return None
        if request.response_model is None:
            return response
        try:
            validated = request.response_model.model_validate(response.parsed_json)
        except ValidationError:
            return None
        return response.model_copy(update={"parsed_json": validated.model_dump(mode="json")})

    def _trace_cache_hit(self, request: LLMRequest, response: LLMResponse) -> None:
        self._append_trace(
            TraceEvent(
                occurred_at=datetime.now(UTC),
                run_id=request.run_id,
                case_id=request.case_id,
                stage=request.stage,
                experiment_batch=request.experiment_batch,
                random_seed=request.seed,
                prompt_version=request.prompt_version,
                input_hashes=request.input_hashes,
                parameters=_trace_parameters(request.temperature, request.enable_thinking),
                requested_model=request.requested_model,
                returned_model=response.returned_model,
                endpoint_fingerprint=self._endpoint_fingerprint,
                request_id=response.request_id,
                content_hash=response.content_hash,
                parsed_json=response.parsed_json,
                usage=response.usage,
                retries=response.retries,
                status="cache_hit",
                adaptations=request.adaptations,
            )
        )

    def _trace_completion(
        self,
        request: LLMRequest,
        completion: _RawCompletion,
        *,
        status: Literal["succeeded", "failed"],
        parsed_json: dict[str, Any] | None = None,
        error: Exception | None = None,
        stage: str | None = None,
        temperature: float | None = None,
        enable_thinking: bool | None = None,
        prompt_version: str | None = None,
        input_hashes: dict[str, str] | None = None,
    ) -> None:
        self._append_trace(
            TraceEvent(
                occurred_at=datetime.now(UTC),
                run_id=request.run_id,
                case_id=request.case_id,
                stage=stage or request.stage,
                experiment_batch=request.experiment_batch,
                random_seed=request.seed,
                prompt_version=prompt_version or request.prompt_version,
                input_hashes=input_hashes or request.input_hashes,
                parameters=_trace_parameters(
                    request.temperature if temperature is None else temperature,
                    request.enable_thinking if enable_thinking is None else enable_thinking,
                ),
                requested_model=completion.requested_model,
                returned_model=completion.returned_model,
                endpoint_fingerprint=self._endpoint_fingerprint,
                request_id=completion.request_id,
                content_hash=completion.content_hash,
                parsed_json=parsed_json,
                usage=completion.usage,
                retries=completion.retries,
                status=status,
                error_summary=self._safe_error(error),
                adaptations=request.adaptations,
            )
        )

    def _trace_request_failure(
        self,
        request: LLMRequest,
        error: Exception,
        *,
        requested_model: str | None = None,
        stage: str | None = None,
        temperature: float | None = None,
        enable_thinking: bool | None = None,
        prompt_version: str | None = None,
        input_hashes: dict[str, str] | None = None,
        summary_error: Exception | None = None,
    ) -> None:
        if isinstance(error, _BillableResponseError):
            self._append_trace(
                TraceEvent(
                    occurred_at=datetime.now(UTC),
                    run_id=request.run_id,
                    case_id=request.case_id,
                    stage=stage or request.stage,
                    experiment_batch=request.experiment_batch,
                    random_seed=request.seed,
                    prompt_version=prompt_version or request.prompt_version,
                    input_hashes=input_hashes or request.input_hashes,
                    parameters=_trace_parameters(
                        request.temperature if temperature is None else temperature,
                        request.enable_thinking if enable_thinking is None else enable_thinking,
                    ),
                    requested_model=error.requested_model,
                    returned_model=error.returned_model,
                    endpoint_fingerprint=self._endpoint_fingerprint,
                    request_id=error.request_id,
                    content_hash=error.content_hash,
                    usage=error.usage,
                    retries=error.retries,
                    status="failed",
                    error_summary=self._safe_error(
                        error if summary_error is None else summary_error
                    ),
                    adaptations=request.adaptations,
                )
            )
            return
        self._append_trace(
            TraceEvent(
                occurred_at=datetime.now(UTC),
                run_id=request.run_id,
                case_id=request.case_id,
                stage=stage or request.stage,
                experiment_batch=request.experiment_batch,
                random_seed=request.seed,
                prompt_version=prompt_version or request.prompt_version,
                input_hashes=input_hashes or request.input_hashes,
                parameters=_trace_parameters(
                    request.temperature if temperature is None else temperature,
                    request.enable_thinking if enable_thinking is None else enable_thinking,
                ),
                requested_model=requested_model or request.requested_model,
                endpoint_fingerprint=self._endpoint_fingerprint,
                retries=getattr(error, "retries", 0),
                status="failed",
                error_summary=self._safe_error(
                    error if summary_error is None else summary_error
                ),
                adaptations=request.adaptations,
            )
        )

    def _safe_error(self, error: Exception | None) -> str | None:
        if error is None:
            return None
        summary = str(error).replace(self._api_key, "[REDACTED]")
        return re.sub(r"(?i)Bearer\s+\S+", "Bearer [REDACTED]", summary)[:500]

    def _append_trace(self, event: TraceEvent) -> None:
        if self._trace_writer is not None:
            self._trace_writer.append(event)


@dataclass(frozen=True, slots=True)
class _ResponseEnvelope:
    request_id: str
    returned_model: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def _decode_response_payload(response: httpx.Response) -> dict[str, object]:
    try:
        value: object = response.json()
    except ValueError as error:
        raise GatewayResponseError("Qwen returned a non-JSON response envelope") from error
    if not isinstance(value, dict):
        raise GatewayResponseError("Qwen response envelope must be a JSON object")
    return value


def _decode_cost_token_usage(payload: Mapping[str, object]) -> tuple[int, int]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        raise GatewayResponseError("Qwen response is missing token usage")
    return (
        _required_nonnegative_int(usage, "prompt_tokens"),
        _required_nonnegative_int(usage, "completion_tokens"),
    )


def _decode_total_tokens(payload: Mapping[str, object]) -> int:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        raise GatewayResponseError("Qwen response is missing token usage")
    return _required_nonnegative_int(usage, "total_tokens")


def _decode_response_envelope(
    payload: Mapping[str, object],
    *,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> _ResponseEnvelope:

    request_id = payload.get("id")
    returned_model = payload.get("model")
    choices = payload.get("choices")
    if not isinstance(request_id, str) or not request_id:
        raise GatewayResponseError("Qwen response is missing a request identifier")
    if not isinstance(returned_model, str) or not returned_model:
        raise GatewayResponseError("Qwen response is missing a model identifier")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise GatewayResponseError("Qwen response is missing choices[0]")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise GatewayResponseError("Qwen response is missing assistant content")
    content = message["content"]
    if not isinstance(content, str):
        raise GatewayResponseError("Qwen response is missing assistant content")
    return _ResponseEnvelope(
        request_id=request_id,
        returned_model=returned_model,
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _optional_response_content(payload: Mapping[str, object]) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    return _optional_string(message.get("content"))


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _required_nonnegative_int(value: Mapping[str, object], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool) or item < 0:
        raise GatewayResponseError(f"Qwen usage is missing nonnegative integer {key}")
    return item


def _is_retryable(status_code: int) -> bool:
    return status_code in _RETRYABLE_STATUS_CODES or status_code >= 500


def _reject_non_finite_json(constant: str) -> NoReturn:
    raise ValueError(f"Non-finite JSON constant: {constant}")


def _combine_usage(primary: Usage, repair: Usage) -> Usage:
    return Usage(
        prompt_tokens=primary.prompt_tokens + repair.prompt_tokens,
        completion_tokens=primary.completion_tokens + repair.completion_tokens,
        total_tokens=primary.total_tokens + repair.total_tokens,
        estimated_cost_cny=primary.estimated_cost_cny + repair.estimated_cost_cny,
        latency_ms=primary.latency_ms + repair.latency_ms,
        cache_hit=False,
    )


def _trace_parameters(temperature: float, enable_thinking: bool) -> dict[str, object]:
    return {
        "temperature": temperature,
        "enable_thinking": enable_thinking,
        "response_format": {"type": "json_object"},
    }


def _validate_workspace_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    parsed = urlsplit(normalized)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != "/compatible-mode/v1"
    ):
        raise GatewayConfigurationError(
            "DASHSCOPE_BASE_URL must be an HTTPS Workspace-compatible /compatible-mode/v1 URL"
        )

    hostname = parsed.hostname.casefold()
    if hostname in _SHARED_HOSTS:
        raise GatewayConfigurationError("Shared DashScope endpoints are not allowed")
    labels = hostname.split(".")
    official_workspace = (
        hostname.endswith(".maas.aliyuncs.com")
        and len(labels) >= 5
        and labels[0] not in _DISALLOWED_DEDICATED_PREFIXES
        and "{" not in labels[0]
        and "}" not in labels[0]
    )
    reserved_test_workspace = hostname == "workspace.example.invalid"
    if not official_workspace and not reserved_test_workspace:
        raise GatewayConfigurationError("A Workspace-dedicated Model Studio endpoint is required")
    return normalized
