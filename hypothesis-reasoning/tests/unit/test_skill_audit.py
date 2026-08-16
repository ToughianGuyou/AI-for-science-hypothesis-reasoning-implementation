from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import validate
from typer.testing import CliRunner

from hypothesis_reasoning.cli import app
from hypothesis_reasoning.errors import AuditValidationError
from hypothesis_reasoning.research.audit import (
    CandidateEvaluationTarget,
    DecisionGates,
    SkillAuditRecord,
    SkillAuditReport,
    audit_inventory,
    determine_audit_decision,
    find_exact_skill_path,
    scan_skill_tree,
)
from hypothesis_reasoning.research.fetch import (
    extract_zip_archive,
    fetch_repository_archive,
    fetch_repository_snapshot,
)

PINNED_COMMIT = "a" * 40
CONTENT_HASH = "b" * 64


def audit_record_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "candidate_name": "Hypothesis Generation",
        "repository": "https://github.com/K-Dense-AI/scientific-agent-skills",
        "commit": PINNED_COMMIT,
        "relative_path": "scientific-skills/hypothesis-generation/SKILL.md",
        "content_sha256": CONTENT_HASH,
        "license": "MIT",
        "dependencies": [],
        "mechanism": "Structured hypothesis generation instructions.",
        "input_contract": "Unstructured user prompt.",
        "output_contract": "Markdown instructions without a required JSON schema.",
        "evidence_control": "Mentions evidence but does not enforce supplied IDs.",
        "loop_control": "Single pass unless the host repeats the workflow.",
        "observability": "No machine-readable trace contract.",
        "security_risks": [],
        "scanned_files": ["SKILL.md"],
        "decision_gates": {
            "license_compatible": True,
            "intended_for_b1": True,
            "scientific_reasoning_detected": True,
            "exact_hypothesis_contract_detected": False,
            "supplied_reference_enforcement_detected": False,
            "bounded_loop_contract_detected": False,
            "run_trace_contract_detected": False,
            "host_actions_detected": False,
        },
        "integration_effort": "medium",
        "decision": "adapt",
        "decision_reason": "The standard JSON and reference contract must be injected.",
        "status": "verified",
    }
    payload.update(overrides)
    return payload


def test_audit_rejects_unpinned_source() -> None:
    with pytest.raises(AuditValidationError, match="40-character commit SHA"):
        SkillAuditRecord.model_validate(audit_record_payload(commit="main"))


def test_decision_gates_are_evidence_based_not_candidate_name_based() -> None:
    gates = DecisionGates(
        license_compatible=True,
        intended_for_b1=True,
        scientific_reasoning_detected=True,
        exact_hypothesis_contract_detected=False,
        supplied_reference_enforcement_detected=False,
        bounded_loop_contract_detected=False,
        run_trace_contract_detected=False,
        host_actions_detected=False,
    )

    assert determine_audit_decision(gates).value == "adapt"
    assert determine_audit_decision(
        gates.model_copy(update={"intended_for_b1": False})
    ).value == "design_only"
    assert CandidateEvaluationTarget.B1_PROMPT.value == "b1_prompt"


def test_not_found_record_keeps_pinned_repository_without_inventing_a_path() -> None:
    record = SkillAuditRecord.model_validate(
        audit_record_payload(
            candidate_name="Missing Candidate",
            relative_path=None,
            content_sha256=None,
            mechanism="Candidate name was not found at the pinned commit.",
            input_contract="Not available.",
            output_contract="Not available.",
            evidence_control="Not available.",
            loop_control="Not available.",
            observability="Not available.",
            scanned_files=[],
            decision_gates=None,
            integration_effort="unknown",
            decision="not_found",
            decision_reason="No exact skill path matched the approved candidate name.",
            status="not_found",
        )
    )

    assert record.relative_path is None
    assert record.decision.value == "not_found"


def test_verified_record_requires_a_content_hash() -> None:
    with pytest.raises(AuditValidationError, match="content_sha256"):
        SkillAuditRecord.model_validate(audit_record_payload(content_sha256=None))


def test_exact_path_resolution_does_not_fuzzy_map_a_missing_name(tmp_path: Path) -> None:
    near_match = tmp_path / "scientific-skills" / "conscious-council"
    near_match.mkdir(parents=True)
    (near_match / "SKILL.md").write_text("# Conscious Council\n", encoding="utf-8")

    assert find_exact_skill_path(tmp_path, "consciousness-council") is None


def test_static_scan_includes_referenced_files_and_reports_risky_actions(tmp_path: Path) -> None:
    skill_dir = tmp_path / "scientific-skills" / "candidate"
    reference_dir = skill_dir / "references"
    reference_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "# Candidate\n"
        "See [the workflow](references/workflow.md).\n"
        "Install it with `pip install example-package`.\n"
        "Fetch inputs with `curl https://example.invalid/data`.\n",
        encoding="utf-8",
    )
    (reference_dir / "workflow.md").write_text(
        "Run `python analyze.py` in a shell and write results to output.json.\n",
        encoding="utf-8",
    )

    scan = scan_skill_tree(skill_dir / "SKILL.md")

    assert {finding.kind.value for finding in scan.security_findings} == {
        "file_write",
        "network_call",
        "package_install",
        "shell_command",
    }
    assert {path.as_posix() for path in scan.scanned_files} == {
        "SKILL.md",
        "references/workflow.md",
    }


def test_static_scan_covers_plain_path_references_and_script_side_effects(
    tmp_path: Path,
) -> None:
    skill_dir = tmp_path / "skills" / "candidate"
    (skill_dir / "references").mkdir(parents=True)
    (skill_dir / "scripts").mkdir()
    (skill_dir / "SKILL.md").write_text(
        "Run python scripts/tool.py after reading references/setup.md.\n",
        encoding="utf-8",
    )
    (skill_dir / "references" / "setup.md").write_text(
        "See https://example.invalid/docs for documentation only.\n"
        "Install locally with `uv pip install -e .`.\n",
        encoding="utf-8",
    )
    (skill_dir / "scripts" / "tool.py").write_text(
        "import subprocess\n"
        "import urllib.request\n"
        "from pathlib import Path\n"
        "subprocess.run(['python', 'worker.py'], check=True)\n"
        "urllib.request.urlopen('https://example.invalid/data')\n"
        "Path('output.json').write_text('{}', encoding='utf-8')\n",
        encoding="utf-8",
    )

    scan = scan_skill_tree(skill_dir / "SKILL.md")

    assert {path.as_posix() for path in scan.scanned_files} == {
        "SKILL.md",
        "references/setup.md",
        "scripts/tool.py",
    }
    findings_by_file = {
        (finding.relative_path, finding.kind.value)
        for finding in scan.security_findings
    }
    assert ("references/setup.md", "package_install") in findings_by_file
    assert ("scripts/tool.py", "shell_command") in findings_by_file
    assert ("scripts/tool.py", "network_call") in findings_by_file
    assert ("scripts/tool.py", "file_write") in findings_by_file
    assert not any(
        finding.kind.value == "network_call"
        and finding.relative_path == "references/setup.md"
        for finding in scan.security_findings
    )


def test_fetch_rejects_a_branch_before_making_a_network_request(tmp_path: Path) -> None:
    class NeverCalledClient:
        call_count = 0

        def get(self, _url: str) -> None:
            self.call_count += 1
            raise AssertionError("network must not be called for an unpinned source")

    client = NeverCalledClient()
    with pytest.raises(AuditValidationError, match="40-character commit SHA"):
        fetch_repository_archive(
            repository="https://github.com/K-Dense-AI/scientific-agent-skills",
            commit="main",
            cache_root=tmp_path,
            client=client,
        )

    assert client.call_count == 0


def test_archive_extraction_rejects_parent_directory_escape(tmp_path: Path) -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, mode="w") as archive:
        archive.writestr("repository/SKILL.md", "safe")
        archive.writestr("repository/../../escape.txt", "unsafe")

    with pytest.raises(AuditValidationError, match="unsafe archive path"):
        extract_zip_archive(payload.getvalue(), tmp_path / "destination")

    assert not (tmp_path / "escape.txt").exists()


def test_selective_fetch_downloads_only_exact_candidate_trees(tmp_path: Path) -> None:
    blobs = {
        "LICENSE.md": b"MIT License",
        "skills/hypothesis-generation/SKILL.md": b"# Hypothesis Generation\n",
        "skills/hypothesis-generation/references/guide.md": b"# Guide\n",
        "skills/unrelated/SKILL.md": b"# Unrelated\n",
        "docs/images/large.png": b"not-needed",
    }

    def git_blob_sha(content: bytes) -> str:
        header = f"blob {len(content)}\0".encode()
        return hashlib.sha1(header + content).hexdigest()

    tree = {
        "truncated": False,
        "tree": [
            {
                "path": path,
                "type": "blob",
                "size": len(content),
                "sha": git_blob_sha(content),
            }
            for path, content in blobs.items()
        ],
    }

    class FakeResponse:
        def __init__(self, *, content: bytes = b"", payload: object = None) -> None:
            self.content = content
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> object:
            return self._payload

    class FakeClient:
        def __init__(self) -> None:
            self.urls: list[str] = []

        def get(self, url: str) -> FakeResponse:
            self.urls.append(url)
            if "/git/trees/" in url:
                return FakeResponse(payload=tree)
            raw_path = url.split(f"/{PINNED_COMMIT}/", maxsplit=1)[1]
            return FakeResponse(content=blobs[raw_path])

    client = FakeClient()
    snapshot = fetch_repository_snapshot(
        repository="https://github.com/K-Dense-AI/scientific-agent-skills",
        commit=PINNED_COMMIT,
        candidate_slugs={"hypothesis-generation"},
        cache_root=tmp_path,
        client=client,
    )

    assert (snapshot / "LICENSE.md").is_file()
    assert (snapshot / "skills/hypothesis-generation/SKILL.md").is_file()
    assert (snapshot / "skills/hypothesis-generation/references/guide.md").is_file()
    assert not (snapshot / "skills/unrelated/SKILL.md").exists()
    assert not (snapshot / "docs/images/large.png").exists()
    assert all("unrelated" not in url and "large.png" not in url for url in client.urls)


def test_selective_fetch_revalidates_cached_blob_hashes_before_reuse(tmp_path: Path) -> None:
    blobs = {
        "LICENSE.md": b"MIT License",
        "skills/hypothesis-generation/SKILL.md": b"# Trusted content\n",
    }

    def git_blob_sha(content: bytes) -> str:
        return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()

    tree = {
        "truncated": False,
        "tree": [
            {
                "path": path,
                "type": "blob",
                "size": len(content),
                "sha": git_blob_sha(content),
            }
            for path, content in blobs.items()
        ],
    }

    class FakeResponse:
        def __init__(self, *, content: bytes = b"", payload: object = None) -> None:
            self.content = content
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> object:
            return self._payload

    class FakeClient:
        def __init__(self) -> None:
            self.urls: list[str] = []

        def get(self, url: str) -> FakeResponse:
            self.urls.append(url)
            if "/git/trees/" in url:
                return FakeResponse(payload=tree)
            raw_path = url.split(f"/{PINNED_COMMIT}/", maxsplit=1)[1]
            return FakeResponse(content=blobs[raw_path])

    client = FakeClient()
    kwargs = {
        "repository": "https://github.com/K-Dense-AI/scientific-agent-skills",
        "commit": PINNED_COMMIT,
        "candidate_slugs": {"hypothesis-generation"},
        "cache_root": tmp_path,
        "client": client,
    }
    snapshot = fetch_repository_snapshot(**kwargs)
    skill_path = snapshot / "skills/hypothesis-generation/SKILL.md"
    skill_path.write_text("# Tampered content\n", encoding="utf-8")
    client.urls.clear()

    reused = fetch_repository_snapshot(**kwargs)

    assert reused == snapshot
    assert skill_path.read_bytes() == blobs["skills/hypothesis-generation/SKILL.md"]
    assert any("/git/trees/" in url for url in client.urls)


def test_inventory_audit_pins_source_and_writes_valid_artifacts(tmp_path: Path) -> None:
    inventory_path = tmp_path / "research" / "candidate_skills.yaml"
    inventory_path.parent.mkdir(parents=True)
    inventory_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "repository": "https://github.com/K-Dense-AI/scientific-agent-skills",
                "reference": "main",
                "candidates": [
                    {
                        "name": "Hypothesis Generation",
                        "expected_slug": "hypothesis-generation",
                        "intended_role": "structured hypothesis generation",
                        "evaluation_target": "b1_prompt",
                        "status": "unverified",
                    },
                    {
                        "name": "Missing Candidate",
                        "expected_slug": "missing-candidate",
                        "intended_role": "missing design lead",
                        "evaluation_target": "design_reference",
                        "status": "unverified",
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    blobs = {
        "LICENSE.md": b"MIT License",
        "skills/hypothesis-generation/SKILL.md": (
            b"---\nname: hypothesis-generation\ndescription: Generate hypotheses.\n---\n"
            b"# Hypothesis Generation\n"
            b"Use evidence to generate competing and falsifiable hypotheses.\n"
        ),
        "skills/hypothesis-generation/references/non-runtime-contract.md": (
            b"hypothesis_id topic_id hypotheses evidence_ids observation_ids\n"
            b"Maximum 1 iteration. Emit RunTrace.\n"
        ),
    }

    def git_blob_sha(content: bytes) -> str:
        return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()

    tree = {
        "truncated": False,
        "tree": [
            {
                "path": path,
                "type": "blob",
                "size": len(content),
                "sha": git_blob_sha(content),
            }
            for path, content in blobs.items()
        ],
    }

    class FakeResponse:
        def __init__(self, *, payload: object = None, content: bytes = b""):
            self._payload = payload
            self.content = content

        def raise_for_status(self) -> None:
            return None

        def json(self) -> object:
            assert self._payload is not None
            return self._payload

    class FakeClient:
        def __init__(self) -> None:
            self.urls: list[str] = []

        def get(self, url: str) -> FakeResponse:
            self.urls.append(url)
            if "/commits/" in url:
                return FakeResponse(payload={"sha": PINNED_COMMIT})
            if "/git/trees/" in url:
                return FakeResponse(payload=tree)
            raw_path = url.split(f"/{PINNED_COMMIT}/", maxsplit=1)[1]
            return FakeResponse(content=blobs[raw_path])

    output_path = tmp_path / "results" / "skill-audit.json"
    schema_path = tmp_path / "research" / "audit_schema.json"
    comparison_path = tmp_path / "docs" / "skill-comparison.md"
    client = FakeClient()

    report = audit_inventory(
        inventory_path=inventory_path,
        output_path=output_path,
        schema_path=schema_path,
        comparison_path=comparison_path,
        cache_root=tmp_path / ".cache" / "skills",
        client=client,
    )

    reloaded = SkillAuditReport.model_validate_json(output_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validate(reloaded.model_dump(mode="json"), schema)
    table = comparison_path.read_text(encoding="utf-8")
    assert reloaded == report
    assert report.commit == PINNED_COMMIT
    assert [record.status.value for record in report.records] == ["verified", "not_found"]
    assert report.records[0].decision.value == "adapt"
    assert schema["title"] == "SkillAuditReport"
    assert table.count("| Hypothesis Generation |") == 1
    assert table.count("| Missing Candidate |") == 1
    assert "| not_found |" in table
    assert client.urls == [
        "https://api.github.com/repos/K-Dense-AI/scientific-agent-skills/commits/main",
        "https://api.github.com/repos/K-Dense-AI/scientific-agent-skills/git/trees/"
        f"{PINNED_COMMIT}?recursive=1",
        "https://raw.githubusercontent.com/K-Dense-AI/scientific-agent-skills/"
        f"{PINNED_COMMIT}/LICENSE.md",
        "https://raw.githubusercontent.com/K-Dense-AI/scientific-agent-skills/"
        f"{PINNED_COMMIT}/skills/hypothesis-generation/SKILL.md",
        "https://raw.githubusercontent.com/K-Dense-AI/scientific-agent-skills/"
        f"{PINNED_COMMIT}/skills/hypothesis-generation/references/"
        "non-runtime-contract.md",
    ]


def test_audit_skills_cli_runs_the_offline_controlled_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory_path = tmp_path / "candidate_skills.yaml"
    inventory_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "repository": "https://github.com/K-Dense-AI/scientific-agent-skills",
                "reference": "main",
                "candidates": [
                    {
                        "name": "Hypothesis Generation",
                        "expected_slug": "hypothesis-generation",
                        "intended_role": "structured hypothesis generation",
                        "evaluation_target": "b1_prompt",
                        "status": "unverified",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    blobs = {
        "LICENSE.md": b"MIT License",
        "skills/hypothesis-generation/SKILL.md": (
            b"# Hypothesis Generation\nGenerate evidence-bound hypotheses.\n"
        ),
    }
    tree = {
        "truncated": False,
        "tree": [
            {
                "path": path,
                "type": "blob",
                "size": len(content),
                "sha": hashlib.sha1(
                    f"blob {len(content)}\0".encode() + content
                ).hexdigest(),
            }
            for path, content in blobs.items()
        ],
    }

    class FakeResponse:
        def __init__(self, url: str) -> None:
            self.url = url
            if "/raw.githubusercontent.com/" in url:
                raw_path = url.split(f"/{PINNED_COMMIT}/", maxsplit=1)[1]
                self.content = blobs[raw_path]
            else:
                self.content = b""

        def raise_for_status(self) -> None:
            return None

        def json(self) -> object:
            if "/commits/" in self.url:
                return {"sha": PINNED_COMMIT}
            return tree

    monkeypatch.setattr(
        "httpx.Client.get", lambda _client, url: FakeResponse(str(url))
    )
    output_path = tmp_path / "skill-audit.json"
    schema_path = tmp_path / "audit_schema.json"
    comparison_path = tmp_path / "skill-comparison.md"

    result = CliRunner().invoke(
        app,
        [
            "audit-skills",
            "--inventory",
            str(inventory_path),
            "--out",
            str(output_path),
            "--schema",
            str(schema_path),
            "--comparison",
            str(comparison_path),
            "--cache-root",
            str(tmp_path / ".cache" / "skills"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert PINNED_COMMIT in result.stdout
    assert output_path.is_file()
    assert schema_path.is_file()
    assert comparison_path.is_file()
