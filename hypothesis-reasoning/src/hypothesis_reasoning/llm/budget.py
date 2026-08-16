"""Fail-closed in-memory budget partitions for model calls."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from threading import RLock
from types import MappingProxyType

from hypothesis_reasoning.errors import BudgetConfigurationError, BudgetExceededError
from hypothesis_reasoning.models import Usage


@dataclass(frozen=True, slots=True)
class BudgetEntry:
    partition: str
    actual_cny: Decimal
    usage: Usage | None


@dataclass(frozen=True, slots=True)
class BudgetReservation:
    reservation_id: int
    partition: str
    reserved_cny: Decimal


class BudgetLedger:
    """Track actual API cost against independent hard partitions."""

    def __init__(self, limits: Mapping[str, Decimal]) -> None:
        if not limits:
            raise BudgetConfigurationError("At least one budget partition is required")
        normalized = {name: Decimal(str(limit)) for name, limit in limits.items()}
        if any(not name or limit < 0 for name, limit in normalized.items()):
            raise BudgetConfigurationError("Budget names must be nonempty and limits nonnegative")
        self._limits = MappingProxyType(normalized)
        self._spent = {name: Decimal("0") for name in normalized}
        self._reserved = {name: Decimal("0") for name in normalized}
        self._reservations: dict[int, BudgetReservation] = {}
        self._next_reservation_id = 1
        self._exhausted: set[str] = set()
        self._entries: list[BudgetEntry] = []
        self._lock = RLock()

    @property
    def entries(self) -> tuple[BudgetEntry, ...]:
        with self._lock:
            return tuple(self._entries)

    def assert_can_spend(self, partition: str, estimated_cny: Decimal) -> None:
        amount = self._normalize_amount(estimated_cny)
        with self._lock:
            self._assert_known(partition)
            if (
                partition in self._exhausted
                or self._spent[partition] + self._reserved[partition] + amount
                > self._limits[partition]
            ):
                raise BudgetExceededError(
                    f"Budget partition {partition!r} would exceed its "
                    f"{self._limits[partition]} CNY limit"
                )

    def reserve(self, partition: str, estimated_cny: Decimal) -> BudgetReservation:
        amount = self._normalize_amount(estimated_cny)
        with self._lock:
            self.assert_can_spend(partition, amount)
            reservation = BudgetReservation(
                reservation_id=self._next_reservation_id,
                partition=partition,
                reserved_cny=amount,
            )
            self._next_reservation_id += 1
            self._reservations[reservation.reservation_id] = reservation
            self._reserved[partition] += amount
            return reservation

    def settle(
        self,
        reservation: BudgetReservation,
        actual_cny: Decimal,
        usage: Usage | None = None,
    ) -> None:
        amount = self._normalize_amount(actual_cny)
        with self._lock:
            active = self._pop_reservation(reservation)
            self._reserved[active.partition] -= active.reserved_cny
            self._record_actual(active.partition, amount, usage)

    def cancel(self, reservation: BudgetReservation) -> None:
        with self._lock:
            active = self._pop_reservation(reservation)
            self._reserved[active.partition] -= active.reserved_cny

    def record(
        self,
        partition: str,
        actual_cny: Decimal,
        usage: Usage | None = None,
    ) -> None:
        amount = self._normalize_amount(actual_cny)
        with self._lock:
            self._assert_known(partition)
            self._record_actual(partition, amount, usage)

    def spent(self, partition: str) -> Decimal:
        with self._lock:
            self._assert_known(partition)
            return self._spent[partition]

    def remaining(self, partition: str) -> Decimal:
        with self._lock:
            self._assert_known(partition)
            remaining = (
                self._limits[partition]
                - self._spent[partition]
                - self._reserved[partition]
            )
            return max(Decimal("0"), remaining)

    def _pop_reservation(self, reservation: BudgetReservation) -> BudgetReservation:
        active = self._reservations.pop(reservation.reservation_id, None)
        if active != reservation:
            raise BudgetConfigurationError("Unknown or already settled budget reservation")
        return active

    def _record_actual(
        self,
        partition: str,
        amount: Decimal,
        usage: Usage | None,
    ) -> None:
        self._spent[partition] += amount
        self._entries.append(BudgetEntry(partition, amount, usage))
        if self._spent[partition] >= self._limits[partition]:
            self._exhausted.add(partition)

    def _assert_known(self, partition: str) -> None:
        if partition not in self._limits:
            raise BudgetConfigurationError(f"Unknown budget partition: {partition!r}")

    @staticmethod
    def _normalize_amount(value: Decimal) -> Decimal:
        amount = Decimal(str(value))
        if not amount.is_finite() or amount < 0:
            raise BudgetConfigurationError("Budget amounts must be finite and nonnegative")
        return amount
