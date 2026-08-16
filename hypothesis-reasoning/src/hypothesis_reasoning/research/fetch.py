"""Pinned, non-executing archive fetching for static skill audits."""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urlparse

from hypothesis_reasoning.errors import AuditValidationError

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_MAX_ARCHIVE_FILES = 20_000
_MAX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024


def fetch_repository_archive(
    *, repository: str, commit: str, cache_root: Path, client: Any
) -> Path:
    """Download a GitHub ZIP only after *commit* is an immutable SHA."""

    _require_commit_sha(commit)
    owner, name = _parse_github_repository(repository)
    repository_hash = hashlib.sha256(repository.rstrip("/").encode("utf-8")).hexdigest()[:16]
    destination = cache_root / repository_hash / commit
    completion_marker = destination / ".audit-fetch-complete"
    if completion_marker.is_file():
        return destination

    response = client.get(f"https://codeload.github.com/{owner}/{name}/zip/{commit}")
    response.raise_for_status()
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    extract_zip_archive(response.content, temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    (temporary / ".audit-fetch-complete").write_text(commit + "\n", encoding="utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)
    return destination


def resolve_commit(repository: str, reference: str, client: Any) -> str:
    """Resolve a mutable GitHub reference to the exact commit used for an audit."""

    owner, name = _parse_github_repository(repository)
    if not reference or any(character.isspace() for character in reference):
        raise AuditValidationError("repository reference must be a non-empty Git reference")
    response = client.get(
        f"https://api.github.com/repos/{owner}/{name}/commits/{reference}"
    )
    response.raise_for_status()
    payload = response.json()
    commit = payload.get("sha") if isinstance(payload, dict) else None
    if not isinstance(commit, str):
        raise AuditValidationError("GitHub commit response did not contain a SHA")
    _require_commit_sha(commit)
    return commit


def fetch_repository_snapshot(
    *,
    repository: str,
    commit: str,
    candidate_slugs: set[str],
    cache_root: Path,
    client: Any,
) -> Path:
    """Fetch only exact candidate directories from one immutable GitHub tree."""

    _require_commit_sha(commit)
    owner, name = _parse_github_repository(repository)
    if any(not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug) for slug in candidate_slugs):
        raise AuditValidationError("candidate slugs must be exact lowercase directory names")
    repository_hash = hashlib.sha256(repository.rstrip("/").encode("utf-8")).hexdigest()[:16]
    destination = cache_root / repository_hash / commit
    manifest_path = destination / ".audit-fetch-manifest.json"
    expected_slugs = sorted(candidate_slugs)
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = None
        if _cached_snapshot_is_valid(
            manifest,
            destination=destination,
            repository=repository.rstrip("/"),
            commit=commit,
            candidate_slugs=expected_slugs,
        ):
            return destination

    tree_response = client.get(
        f"https://api.github.com/repos/{owner}/{name}/git/trees/{commit}?recursive=1"
    )
    tree_response.raise_for_status()
    tree_payload = tree_response.json()
    if not isinstance(tree_payload, dict) or tree_payload.get("truncated") is not False:
        raise AuditValidationError("GitHub repository tree is missing or truncated")
    raw_entries = tree_payload.get("tree")
    if not isinstance(raw_entries, list) or len(raw_entries) > _MAX_ARCHIVE_FILES:
        raise AuditValidationError("GitHub repository tree is invalid or too large")
    entries = [_validate_tree_blob(entry) for entry in raw_entries]

    skill_roots: list[PurePosixPath] = []
    for slug in expected_slugs:
        matches = [
            PurePosixPath(entry["path"]).parent
            for entry in entries
            if PurePosixPath(entry["path"]).name == "SKILL.md"
            and PurePosixPath(entry["path"]).parent.name == slug
        ]
        if len(matches) > 1:
            raise AuditValidationError(f"multiple exact skill directories matched {slug}")
        if matches:
            skill_roots.append(matches[0])

    selected = [
        entry
        for entry in entries
        if _is_root_license(entry["path"])
        or any(_path_is_within(entry["path"], root) for root in skill_roots)
    ]
    if sum(entry["size"] for entry in selected) > _MAX_UNCOMPRESSED_BYTES:
        raise AuditValidationError("selected candidate files exceed the audit size limit")

    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    for entry in sorted(selected, key=lambda item: item["path"]):
        raw_path = entry["path"]
        response = client.get(
            f"https://raw.githubusercontent.com/{owner}/{name}/{commit}/"
            f"{quote(raw_path, safe='/')}"
        )
        response.raise_for_status()
        content = response.content
        if len(content) != entry["size"] or _git_blob_sha(content) != entry["sha"]:
            raise AuditValidationError(f"Git blob verification failed for {raw_path}")
        target = temporary.joinpath(*PurePosixPath(raw_path).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    temporary.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "repository": repository.rstrip("/"),
        "commit": commit,
        "candidate_slugs": expected_slugs,
        "files": [
            {"path": entry["path"], "size": entry["size"], "sha": entry["sha"]}
            for entry in selected
        ],
    }
    (temporary / ".audit-fetch-manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)
    return destination


def _validate_tree_blob(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("type") != "blob":
        return {"path": "", "size": 0, "sha": "0" * 40}
    path = value.get("path")
    size = value.get("size")
    sha = value.get("sha")
    if (
        not isinstance(path, str)
        or not isinstance(size, int)
        or size < 0
        or not isinstance(sha, str)
        or not _COMMIT_SHA.fullmatch(sha)
    ):
        raise AuditValidationError("GitHub repository tree contains an invalid blob entry")
    parts = PurePosixPath(path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise AuditValidationError(f"unsafe Git tree path: {path}")
    return {"path": path, "size": size, "sha": sha}


def _is_root_license(path: str) -> bool:
    parsed = PurePosixPath(path)
    return len(parsed.parts) == 1 and parsed.name.lower().startswith("license")


def _path_is_within(path: str, root: PurePosixPath) -> bool:
    parsed = PurePosixPath(path)
    return parsed == root or root in parsed.parents


def _git_blob_sha(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def _cached_snapshot_is_valid(
    manifest: object,
    *,
    destination: Path,
    repository: str,
    commit: str,
    candidate_slugs: list[str],
) -> bool:
    if not isinstance(manifest, dict):
        return False
    if (
        manifest.get("repository") != repository
        or manifest.get("commit") != commit
        or manifest.get("candidate_slugs") != candidate_slugs
    ):
        return False
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        return False
    expected_paths: set[str] = set()
    validated_entries: list[tuple[Path, int, str]] = []
    for value in files:
        if not isinstance(value, dict):
            return False
        path = value.get("path")
        size = value.get("size")
        sha = value.get("sha")
        if (
            not isinstance(path, str)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(sha, str)
            or not _COMMIT_SHA.fullmatch(sha)
        ):
            return False
        parts = PurePosixPath(path).parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            return False
        expected_paths.add(path)
        validated_entries.append((destination.joinpath(*parts), size, sha))

    actual_paths = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file() and path != destination / ".audit-fetch-manifest.json"
    }
    if actual_paths != expected_paths:
        return False
    for path, size, sha in validated_entries:
        try:
            content = path.read_bytes()
        except OSError:
            return False
        if len(content) != size or _git_blob_sha(content) != sha:
            return False
    return True


def extract_zip_archive(payload: bytes, destination: Path) -> Path:
    """Safely extract a GitHub-style ZIP after validating every member path."""

    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as error:
        raise AuditValidationError("source archive is not a valid ZIP file") from error

    with archive:
        members = archive.infolist()
        if len(members) > _MAX_ARCHIVE_FILES:
            raise AuditValidationError("source archive contains too many files")
        if sum(member.file_size for member in members) > _MAX_UNCOMPRESSED_BYTES:
            raise AuditValidationError("source archive exceeds the uncompressed size limit")
        validated = [_validate_archive_member(member) for member in members]
        top_levels = {parts[0] for parts in validated if parts}
        strip_top_level = len(top_levels) == 1

        for member, parts in zip(members, validated, strict=True):
            relative_parts = parts[1:] if strip_top_level else parts
            if not relative_parts:
                continue
            target = destination.joinpath(*relative_parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    return destination


def _require_commit_sha(commit: str) -> None:
    if not _COMMIT_SHA.fullmatch(commit):
        raise AuditValidationError("commit must be a lowercase 40-character commit SHA")


def _parse_github_repository(repository: str) -> tuple[str, str]:
    parsed = urlparse(repository.rstrip("/"))
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.scheme != "https" or parsed.netloc.lower() != "github.com" or len(parts) != 2:
        raise AuditValidationError("repository must be an HTTPS GitHub owner/repository URL")
    name = parts[1][:-4] if parts[1].endswith(".git") else parts[1]
    return parts[0], name


def _validate_archive_member(member: zipfile.ZipInfo) -> tuple[str, ...]:
    path = PurePosixPath(member.filename.replace("\\", "/"))
    parts = path.parts
    is_symlink = (member.external_attr >> 16) & 0o170000 == 0o120000
    if (
        not parts
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in parts)
        or ":" in parts[0]
        or is_symlink
    ):
        raise AuditValidationError(f"unsafe archive path: {member.filename}")
    return parts
