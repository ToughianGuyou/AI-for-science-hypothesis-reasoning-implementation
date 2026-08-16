from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from hypothesis_reasoning.errors import (
    BudgetConfigurationError,
    BudgetExceededError,
    PricingConfigurationError,
)
from hypothesis_reasoning.llm.budget import BudgetLedger
from hypothesis_reasoning.llm.types import ModelProfile, PriceTier
from hypothesis_reasoning.models import Usage


def usage() -> Usage:
    return Usage(
        prompt_tokens=120,
        completion_tokens=30,
        total_tokens=150,
        estimated_cost_cny=0.0,
        latency_ms=25,
    )


def test_budget_rejects_partition_overrun() -> None:
    ledger = BudgetLedger({"formal": Decimal("180")})
    ledger.record("formal", Decimal("179.50"))

    with pytest.raises(BudgetExceededError):
        ledger.assert_can_spend("formal", Decimal("0.51"))


def test_budget_records_usage_without_exceeding_partition() -> None:
    ledger = BudgetLedger({"development": Decimal("30")})

    ledger.record("development", Decimal("1.25"), usage())

    assert ledger.spent("development") == Decimal("1.25")
    assert ledger.remaining("development") == Decimal("28.75")
    assert ledger.entries[0].usage == usage()


def test_budget_reservations_prevent_concurrent_overcommit() -> None:
    ledger = BudgetLedger({"formal": Decimal("1.00")})

    reservation = ledger.reserve("formal", Decimal("0.60"))

    with pytest.raises(BudgetExceededError):
        ledger.reserve("formal", Decimal("0.41"))
    ledger.cancel(reservation)
    replacement = ledger.reserve("formal", Decimal("1.00"))
    ledger.cancel(replacement)


def test_budget_settlement_records_actual_overrun_and_blocks_future_spend() -> None:
    ledger = BudgetLedger({"formal": Decimal("1.00")})
    reservation = ledger.reserve("formal", Decimal("0.40"))

    ledger.settle(reservation, Decimal("1.20"), usage())

    assert ledger.spent("formal") == Decimal("1.20")
    assert ledger.entries[-1].actual_cny == Decimal("1.20")
    with pytest.raises(BudgetExceededError):
        ledger.assert_can_spend("formal", Decimal("0"))


def test_budget_rejects_unknown_partition_and_negative_amounts() -> None:
    ledger = BudgetLedger({"formal": Decimal("180")})

    with pytest.raises(BudgetConfigurationError):
        ledger.assert_can_spend("missing", Decimal("1"))
    with pytest.raises(BudgetConfigurationError):
        ledger.record("formal", Decimal("-0.01"))


def test_model_profile_uses_ordered_input_token_price_tiers() -> None:
    profile = ModelProfile(
        requested_alias="qwen-plus",
        pricing_checked_on=date(2026, 8, 15),
        unit_tokens=1_000_000,
        input_token_tiers=(
            PriceTier(
                max_input_tokens=100,
                input_cny_per_unit=Decimal("2"),
                output_cny_per_unit=Decimal("6"),
            ),
            PriceTier(
                max_input_tokens=None,
                input_cny_per_unit=Decimal("4"),
                output_cny_per_unit=Decimal("12"),
            ),
        ),
    )

    assert profile.calculate_cost(prompt_tokens=100, completion_tokens=50) == Decimal(
        "0.000500"
    )
    assert profile.calculate_cost(prompt_tokens=101, completion_tokens=50) == Decimal(
        "0.001004"
    )


@pytest.mark.parametrize("checked_on", [None, date(2026, 8, 7)])
def test_model_profile_rejects_missing_or_stale_pricing(checked_on: date | None) -> None:
    profile = ModelProfile(
        requested_alias="qwen-plus",
        pricing_checked_on=checked_on,
        input_token_tiers=(
            PriceTier(
                max_input_tokens=None,
                input_cny_per_unit=Decimal("2"),
                output_cny_per_unit=Decimal("6"),
            ),
        ),
    )

    with pytest.raises(PricingConfigurationError):
        profile.assert_pricing_current(as_of=date(2026, 8, 15))


def test_model_profile_rejects_missing_price_tiers() -> None:
    profile = ModelProfile(
        requested_alias="qwen-plus",
        pricing_checked_on=date(2026, 8, 15),
        input_token_tiers=(),
    )

    with pytest.raises(PricingConfigurationError):
        profile.assert_pricing_current(as_of=date(2026, 8, 15))
