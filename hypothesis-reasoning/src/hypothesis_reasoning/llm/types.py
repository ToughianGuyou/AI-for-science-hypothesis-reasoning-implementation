"""Strict contracts used by the Qwen gateway and its local support services."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hypothesis_reasoning.errors import PricingConfigurationError
from hypothesis_reasoning.models import Usage


class LLMContract(BaseModel):
    """Strict, immutable base class for gateway contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class TraceAdaptation(LLMContract):
    """One auditable prompt adaptation carried through the gateway trace."""

    kind: str = Field(min_length=1)
    detail: str = Field(min_length=1)


class ChatMessage(LLMContract):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class PriceTier(LLMContract):
    """Per-token-unit prices selected by the request's input-token count."""

    max_input_tokens: int | None = Field(default=None, ge=1)
    input_cny_per_unit: Decimal = Field(ge=0)
    output_cny_per_unit: Decimal = Field(ge=0)


class ModelProfile(LLMContract):
    """A requested alias and its recently verified ordered price tiers."""

    requested_alias: str = Field(min_length=1)
    pricing_checked_on: date | None = None
    unit_tokens: int = Field(default=1_000_000, ge=1)
    input_token_tiers: tuple[PriceTier, ...]

    @model_validator(mode="after")
    def validate_tier_order(self) -> ModelProfile:
        finite_limits = [
            tier.max_input_tokens
            for tier in self.input_token_tiers
            if tier.max_input_tokens is not None
        ]
        if finite_limits != sorted(finite_limits) or len(finite_limits) != len(set(finite_limits)):
            raise ValueError("input-token price tiers must have unique ascending limits")
        unbounded_positions = [
            index
            for index, tier in enumerate(self.input_token_tiers)
            if tier.max_input_tokens is None
        ]
        if unbounded_positions and unbounded_positions != [len(self.input_token_tiers) - 1]:
            raise ValueError("only the final input-token price tier may be unbounded")
        return self

    def assert_pricing_current(self, *, as_of: date) -> None:
        if not self.input_token_tiers:
            raise PricingConfigurationError(
                f"No verified prices are configured for {self.requested_alias}"
            )
        if self.pricing_checked_on is None:
            raise PricingConfigurationError(
                f"Pricing has not been verified for {self.requested_alias}"
            )
        if self.pricing_checked_on > as_of:
            raise PricingConfigurationError(
                f"Pricing verification date is in the future for {self.requested_alias}"
            )
        if as_of - self.pricing_checked_on > timedelta(days=7):
            raise PricingConfigurationError(
                f"Pricing is older than seven days for {self.requested_alias}"
            )

    def calculate_cost(self, *, prompt_tokens: int, completion_tokens: int) -> Decimal:
        if prompt_tokens < 0 or completion_tokens < 0:
            raise PricingConfigurationError("Token counts cannot be negative")
        tier = next(
            (
                candidate
                for candidate in self.input_token_tiers
                if candidate.max_input_tokens is None
                or prompt_tokens <= candidate.max_input_tokens
            ),
            None,
        )
        if tier is None:
            raise PricingConfigurationError(
                f"No price tier covers {prompt_tokens} input tokens for {self.requested_alias}"
            )
        unit = Decimal(self.unit_tokens)
        return (
            Decimal(prompt_tokens) * tier.input_cny_per_unit
            + Decimal(completion_tokens) * tier.output_cny_per_unit
        ) / unit


class LLMRequest(LLMContract):
    """One reproducible JSON-mode completion request."""

    requested_model: str = Field(min_length=1)
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    seed: int
    temperature: float = Field(ge=0.0, le=2.0)
    prompt_version: str = Field(min_length=1)
    input_hashes: dict[str, str] = Field(min_length=1)
    response_model: type[BaseModel] | None = Field(default=None, exclude=True)
    budget_partition: str = Field(min_length=1)
    estimated_cost_cny: Decimal = Field(ge=0)
    run_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    experiment_batch: str = Field(min_length=1)
    enable_thinking: bool = False
    adaptations: tuple[TraceAdaptation, ...] = ()


class LLMResponse(LLMContract):
    """Parsed response metadata; raw API bodies and credentials are intentionally absent."""

    request_id: str = Field(min_length=1)
    requested_model: str = Field(min_length=1)
    returned_model: str = Field(min_length=1)
    content_hash: str = Field(min_length=64, max_length=64)
    parsed_json: dict[str, Any]
    called_at: datetime
    usage: Usage
    retries: int = Field(ge=0)
    repaired: bool = False
    repair_request_id: str | None = None
    repair_returned_model: str | None = None

    @model_validator(mode="after")
    def validate_repair_metadata(self) -> LLMResponse:
        has_request_id = self.repair_request_id is not None
        has_returned_model = self.repair_returned_model is not None
        if self.repaired and not (has_request_id and has_returned_model):
            raise ValueError("Repair status and repair response metadata must agree")
        if not self.repaired and (has_request_id or has_returned_model):
            raise ValueError("Repair status and repair response metadata must agree")
        return self


class TraceEvent(LLMContract):
    """Safe append-only event containing hashes and parsed output, never request secrets."""

    occurred_at: datetime
    run_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    experiment_batch: str = Field(min_length=1)
    random_seed: int
    prompt_version: str = Field(min_length=1)
    input_hashes: dict[str, str] = Field(min_length=1)
    parameters: dict[str, Any]
    requested_model: str = Field(min_length=1)
    returned_model: str | None = None
    endpoint_fingerprint: str = Field(min_length=64, max_length=64)
    request_id: str | None = None
    content_hash: str | None = Field(default=None, min_length=64, max_length=64)
    parsed_json: dict[str, Any] | None = None
    usage: Usage | None = None
    retries: int = Field(ge=0)
    status: Literal["succeeded", "failed", "cache_hit"]
    error_summary: str | None = None
    adaptations: tuple[TraceAdaptation, ...] = ()
