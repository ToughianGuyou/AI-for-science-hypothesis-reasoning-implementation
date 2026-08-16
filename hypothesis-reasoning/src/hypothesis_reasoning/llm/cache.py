"""Deterministic local cache for parsed model responses."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock, RLock
from typing import Any, BinaryIO

if sys.platform == "win32":
    import msvcrt

    def _lock_file(handle: BinaryIO) -> None:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)

    def _unlock_file(handle: BinaryIO) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock_file(handle: BinaryIO) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

    def _unlock_file(handle: BinaryIO) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

from pydantic import ValidationError

from hypothesis_reasoning.errors import GatewayConfigurationError, ModelChangedError
from hypothesis_reasoning.io import write_json_atomic
from hypothesis_reasoning.llm.types import LLMRequest, LLMResponse

_PATH_LOCKS: dict[Path, RLock] = {}
_PATH_LOCKS_GUARD = Lock()


def _thread_lock_for(path: Path) -> RLock:
    resolved = path.resolve()
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(resolved, RLock())


def _open_initialized_lock_file(path: Path) -> BinaryIO:
    """Open a nonempty lock file without racing a buffered Windows write."""

    deadline = time.monotonic() + 5.0
    while True:
        handle = path.open("a+b", buffering=0)
        handle.seek(0, os.SEEK_END)
        if handle.tell() > 0:
            return handle
        try:
            handle.write(b"\0")
            handle.flush()
        except PermissionError:
            handle.close()
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)
            continue
        return handle


@contextmanager
def _process_file_lock(path: Path) -> Iterator[None]:
    """Hold an OS-backed exclusive lock for one registry path."""

    handle = _open_initialized_lock_file(path)
    locked = False
    try:
        _lock_file(handle)
        locked = True
        yield
    finally:
        try:
            if locked:
                _unlock_file(handle)
        finally:
            handle.close()


def build_cache_key(request: LLMRequest) -> str:
    """Hash every request field that can change model behavior."""

    response_schema: dict[str, Any] | None = None
    if request.response_model is not None:
        response_schema = request.response_model.model_json_schema()
    canonical = {
        "requested_model": request.requested_model,
        "messages": [message.model_dump(mode="json") for message in request.messages],
        "response_format": {"type": "json_object"},
        "temperature": request.temperature,
        "seed": request.seed,
        "prompt_version": request.prompt_version,
        "input_hashes": request.input_hashes,
        "enable_thinking": request.enable_thinking,
        "response_schema": response_schema,
    }
    serialized = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class ResponseCache:
    """One validated metadata-only JSON document per cache key."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)

    def get(self, cache_key: str) -> LLMResponse | None:
        path = self._path(cache_key)
        try:
            return LLMResponse.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, ValidationError):
            return None

    def put(self, cache_key: str, response: LLMResponse) -> None:
        write_json_atomic(self._path(cache_key), response)

    def _path(self, cache_key: str) -> Path:
        return self._directory / f"{cache_key}.json"


class BatchModelRegistry:
    """Persist the concrete model bound to each experiment-batch alias."""

    def __init__(self, path: Path) -> None:
        self._path = path.resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self._path.with_name(f"{self._path.name}.lock")
        self._lock = _thread_lock_for(self._path)

    def bind(self, experiment_batch: str, requested_alias: str, returned_model: str) -> None:
        if not experiment_batch or not requested_alias or not returned_model:
            raise GatewayConfigurationError("Batch model bindings require nonempty identifiers")
        with self._lock, _process_file_lock(self._lock_path):
            bindings = self._load()
            batch = bindings.setdefault(experiment_batch, {})
            previous = batch.setdefault(requested_alias, returned_model)
            if previous != returned_model:
                raise ModelChangedError(
                    f"Returned model changed inside batch {experiment_batch!r} for "
                    f"alias {requested_alias!r}"
                )
            serializable_bindings: dict[object, object] = {
                batch_id: aliases for batch_id, aliases in bindings.items()
            }
            write_json_atomic(self._path, serializable_bindings)

    def _load(self) -> dict[str, dict[str, str]]:
        if not self._path.exists():
            return {}
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise GatewayConfigurationError("Batch model registry is unreadable") from error
        if not isinstance(payload, dict):
            raise GatewayConfigurationError("Batch model registry must contain a JSON object")
        normalized: dict[str, dict[str, str]] = {}
        for batch_id, aliases in payload.items():
            if not isinstance(batch_id, str) or not isinstance(aliases, dict):
                raise GatewayConfigurationError("Batch model registry has invalid entries")
            normalized_aliases: dict[str, str] = {}
            for alias, returned_model in aliases.items():
                if not isinstance(alias, str) or not isinstance(returned_model, str):
                    raise GatewayConfigurationError("Batch model registry has invalid bindings")
                normalized_aliases[alias] = returned_model
            normalized[batch_id] = normalized_aliases
        return normalized
