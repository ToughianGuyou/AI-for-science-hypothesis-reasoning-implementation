"""Candidate skill static-audit contracts and deterministic source scanning."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hypothesis_reasoning.errors import AuditValidationError
from hypothesis_reasoning.research.fetch import fetch_repository_snapshot, resolve_commit

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_CONTENT_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PACKAGE_INSTALL = re.compile(
    r"\b(?:(?:pip|pip3|uv|conda)\s+install|(?:npm|pnpm|yarn)\s+(?:install|add)|"
    r"apt(?:-get)?\s+install)\b",
    re.IGNORECASE,
)
_NETWORK_CALL = re.compile(
    r"\b(?:curl|wget)\b|\b(?:requests|httpx)\.(?:get|post|put|patch|delete|request)\b|"
    r"\burllib\.request\.(?:urlopen|Request)\b|\baiohttp\.ClientSession\b|"
    r"\burlopen\s*\(",
    re.IGNORECASE,
)
_SHELL_COMMAND = re.compile(
    r"\bsubprocess\.(?:run|Popen|call|check_call|check_output)\b|\bos\.system\s*\(|"
    r"\bshell\s*=\s*True\b|"
    r"(?:^|[`$>]\s*|\b(?:run|execute)\s+)"
    r"(?:[A-Z_][A-Z0-9_]*=\S+\s+)*(?:python3?|uv|pip3?|bash|zsh|sh|powershell|pwsh|"
    r"cmd(?:\.exe)?|node|ruby|perl)\b",
    re.IGNORECASE,
)
_FILE_WRITE = re.compile(
    r"\.(?:write_text|write_bytes)\s*\(|\b(?:atomic_write|write_json|write_csv)\w*\s*\(|"
    r"\bopen\s*\([^\n]{0,160}[\"'][wax][+bt]?[\"']|"
    r"\bos\.open\s*\([^\n]{0,160}\bO_(?:WRONLY|RDWR|CREAT|APPEND)\b|"
    r"\b(?:write|save|create|export|append|output)\b[^\n]{0,80}"
    r"(?:\bfile\b|\bdirectory\b|\bfolder\b|\.[a-z0-9]{1,8}\b)",
    re.IGNORECASE,
)
_AUDIT_TEXT_SUFFIXES = {
    ".bat",
    ".cmd",
    ".csv",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_MAX_AUDIT_TEXT_BYTES = 5 * 1024 * 1024


class AuditDecision(StrEnum):
    DIRECT_REUSE = "direct_reuse"
    ADAPT = "adapt"
    DESIGN_ONLY = "design_only"
    REJECT = "reject"
    NOT_FOUND = "not_found"


class AuditStatus(StrEnum):
    VERIFIED = "verified"
    NOT_FOUND = "not_found"


class CandidateEvaluationTarget(StrEnum):
    """Manually declared evaluation scope; it is not an audit conclusion."""

    B1_PROMPT = "b1_prompt"
    DESIGN_REFERENCE = "design_reference"


class IntegrationEffort(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class SecurityFindingKind(StrEnum):
    SHELL_COMMAND = "shell_command"
    PACKAGE_INSTALL = "package_install"
    NETWORK_CALL = "network_call"
    FILE_WRITE = "file_write"


class SecurityFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: SecurityFindingKind
    relative_path: str = Field(min_length=1)
    line: int = Field(ge=1)
    summary: str = Field(min_length=1)


class SkillScan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    scanned_files: list[Path]
    security_findings: list[SecurityFinding]


class DecisionGates(BaseModel):
    """Machine-readable evidence used to derive a preliminary static decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    license_compatible: bool
    intended_for_b1: bool
    scientific_reasoning_detected: bool
    exact_hypothesis_contract_detected: bool
    supplied_reference_enforcement_detected: bool
    bounded_loop_contract_detected: bool
    run_trace_contract_detected: bool
    host_actions_detected: bool


class SkillAuditRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_name: str = Field(min_length=1)
    repository: str = Field(pattern=r"^https://github\.com/[^/]+/[^/]+/?$")
    commit: str
    relative_path: str | None
    content_sha256: str | None
    license: str = Field(min_length=1)
    dependencies: list[str]
    mechanism: str = Field(min_length=1)
    input_contract: str = Field(min_length=1)
    output_contract: str = Field(min_length=1)
    evidence_control: str = Field(min_length=1)
    loop_control: str = Field(min_length=1)
    observability: str = Field(min_length=1)
    security_risks: list[SecurityFinding]
    scanned_files: list[str]
    decision_gates: DecisionGates | None
    integration_effort: IntegrationEffort
    decision: AuditDecision
    decision_reason: str = Field(min_length=1)
    status: AuditStatus

    @model_validator(mode="after")
    def validate_pinned_source(self) -> SkillAuditRecord:
        if not _COMMIT_SHA.fullmatch(self.commit):
            raise AuditValidationError("commit must be a lowercase 40-character commit SHA")
        if self.status is AuditStatus.NOT_FOUND:
            if (
                self.relative_path is not None
                or self.content_sha256 is not None
                or self.scanned_files
                or self.decision_gates is not None
            ):
                raise AuditValidationError(
                    "not_found records cannot invent paths, hashes, scans, or decision gates"
                )
            if self.decision is not AuditDecision.NOT_FOUND:
                raise AuditValidationError("not_found status requires decision=not_found")
            return self
        if self.decision is AuditDecision.NOT_FOUND:
            raise AuditValidationError("verified records cannot use decision=not_found")
        if self.relative_path is None or not self.relative_path.endswith("/SKILL.md"):
            raise AuditValidationError("verified records require a relative_path to SKILL.md")
        if self.content_sha256 is None or not _CONTENT_SHA256.fullmatch(self.content_sha256):
            raise AuditValidationError("verified records require a lowercase content_sha256")
        if "SKILL.md" not in self.scanned_files:
            raise AuditValidationError("verified records must include SKILL.md in scanned_files")
        if self.decision_gates is None:
            raise AuditValidationError("verified records require explicit decision_gates")
        if determine_audit_decision(self.decision_gates) is not self.decision:
            raise AuditValidationError("decision must be derivable from decision_gates")
        return self


class SkillAuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    repository: str = Field(pattern=r"^https://github\.com/[^/]+/[^/]+/?$")
    commit: str
    generated_at: datetime
    records: list[SkillAuditRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_report_commit(self) -> SkillAuditReport:
        if not _COMMIT_SHA.fullmatch(self.commit):
            raise AuditValidationError("report commit must be a lowercase 40-character commit SHA")
        if any(record.repository != self.repository for record in self.records):
            raise AuditValidationError("all records must use the report repository")
        if any(record.commit != self.commit for record in self.records):
            raise AuditValidationError("all records must use the report commit")
        return self


class CandidateHint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    expected_slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    intended_role: str = Field(min_length=1)
    evaluation_target: CandidateEvaluationTarget
    status: Literal["unverified"]


class CandidateInventory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    repository: str = Field(pattern=r"^https://github\.com/[^/]+/[^/]+/?$")
    reference: str = Field(min_length=1)
    candidates: list[CandidateHint] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_candidates(self) -> CandidateInventory:
        names = [candidate.name for candidate in self.candidates]
        slugs = [candidate.expected_slug for candidate in self.candidates]
        if len(names) != len(set(names)) or len(slugs) != len(set(slugs)):
            raise AuditValidationError("candidate names and expected slugs must be unique")
        return self


def audit_inventory(
    *,
    inventory_path: Path,
    output_path: Path,
    schema_path: Path,
    comparison_path: Path,
    cache_root: Path,
    client: Any,
) -> SkillAuditReport:
    """Resolve, fetch, statically audit, and serialize one candidate inventory."""

    inventory = _load_inventory(inventory_path)
    commit = resolve_commit(inventory.repository, inventory.reference, client)
    repository_root = fetch_repository_snapshot(
        repository=inventory.repository,
        commit=commit,
        candidate_slugs={candidate.expected_slug for candidate in inventory.candidates},
        cache_root=cache_root,
        client=client,
    )
    repository_license = _detect_repository_license(repository_root)
    records = [
        _audit_candidate(
            candidate=candidate,
            repository=inventory.repository,
            commit=commit,
            repository_root=repository_root,
            repository_license=repository_license,
        )
        for candidate in inventory.candidates
    ]
    report = SkillAuditReport(
        schema_version=1,
        repository=inventory.repository,
        commit=commit,
        generated_at=datetime.now(UTC),
        records=records,
    )
    _write_json(output_path, report.model_dump(mode="json"))
    _write_json(schema_path, SkillAuditReport.model_json_schema())
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(_render_comparison(report), encoding="utf-8")
    return report


def find_exact_skill_path(repository_root: Path, expected_slug: str) -> Path | None:
    matches = sorted(
        path
        for path in repository_root.rglob("SKILL.md")
        if path.parent.name == expected_slug
    )
    if len(matches) > 1:
        relative_matches = ", ".join(
            path.relative_to(repository_root).as_posix() for path in matches
        )
        raise AuditValidationError(
            f"multiple exact skill paths matched {expected_slug}: {relative_matches}"
        )
    return matches[0] if matches else None


def scan_skill_tree(skill_path: Path) -> SkillScan:
    skill_path = skill_path.resolve()
    skill_root = skill_path.parent
    files = sorted(
        path.resolve()
        for path in skill_root.rglob("*")
        if path.is_file() and path.suffix.lower() in _AUDIT_TEXT_SUFFIXES
    )
    if skill_path not in files:
        raise AuditValidationError(f"skill instruction is not an auditable text file: {skill_path}")
    findings: list[SecurityFinding] = []
    for current in files:
        if not current.is_relative_to(skill_root):
            raise AuditValidationError(f"audit file escapes skill directory: {current}")
        if current.stat().st_size > _MAX_AUDIT_TEXT_BYTES:
            raise AuditValidationError(f"audit text file exceeds size limit: {current}")
        try:
            text = current.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise AuditValidationError(f"audit text file is not UTF-8: {current}") from error
        relative_path = current.relative_to(skill_root)
        for line_number, line in enumerate(text.splitlines(), start=1):
            findings.extend(_scan_line(relative_path, line_number, line))

    return SkillScan(
        scanned_files=[path.relative_to(skill_root) for path in files],
        security_findings=_deduplicate_findings(findings),
    )


def content_sha256(path: Path) -> str:
    """Hash one fetched instruction file without interpreting or executing it."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_inventory(path: Path) -> CandidateInventory:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return CandidateInventory.model_validate(payload)
    except (OSError, yaml.YAMLError, ValueError) as error:
        if isinstance(error, AuditValidationError):
            raise
        raise AuditValidationError(f"invalid candidate inventory at {path}: {error}") from error


def _audit_candidate(
    *,
    candidate: CandidateHint,
    repository: str,
    commit: str,
    repository_root: Path,
    repository_license: str,
) -> SkillAuditRecord:
    skill_path = find_exact_skill_path(repository_root, candidate.expected_slug)
    if skill_path is None:
        return SkillAuditRecord(
            candidate_name=candidate.name,
            repository=repository,
            commit=commit,
            relative_path=None,
            content_sha256=None,
            license=repository_license,
            dependencies=[],
            mechanism="Candidate name was not found at the pinned commit.",
            input_contract="Not available.",
            output_contract="Not available.",
            evidence_control="Not available.",
            loop_control="Not available.",
            observability="Not available.",
            security_risks=[],
            scanned_files=[],
            decision_gates=None,
            integration_effort=IntegrationEffort.UNKNOWN,
            decision=AuditDecision.NOT_FOUND,
            decision_reason="No exact skill directory matched the approved candidate slug.",
            status=AuditStatus.NOT_FOUND,
        )

    scan = scan_skill_tree(skill_path)
    lowered = skill_path.read_text(encoding="utf-8").lower()
    dependencies = sorted(
        {
            finding.summary
            for finding in scan.security_findings
            if finding.kind is SecurityFindingKind.PACKAGE_INSTALL
        }
    )
    risky_kinds = {finding.kind for finding in scan.security_findings}
    output_mentions_json = "json" in lowered or "schema" in lowered
    evidence_terms = any(term in lowered for term in ("evidence", "citation", "source"))
    loop_terms = any(
        term in lowered for term in ("iteration", "iterate", "loop", "refine", "repeat")
    )
    observable_terms = any(term in lowered for term in ("json", "report", "output", "log"))
    decision_gates = DecisionGates(
        license_compatible=repository_license == "MIT",
        intended_for_b1=candidate.evaluation_target is CandidateEvaluationTarget.B1_PROMPT,
        scientific_reasoning_detected=any(
            term in lowered
            for term in (
                "hypothesis",
                "falsifiable",
                "scientific",
                "evidence",
                "counterfactual",
                "scholar",
            )
        ),
        exact_hypothesis_contract_detected=all(
            term in lowered for term in ("hypothesis_id", "topic_id", "hypotheses")
        ),
        supplied_reference_enforcement_detected=all(
            term in lowered for term in ("evidence_ids", "observation_ids")
        ),
        bounded_loop_contract_detected=bool(
            re.search(
                r"\b(?:bounded|max(?:imum)?|at most|up to)\b[^\n]{0,80}"
                r"\b(?:iteration|iterations|cycle|cycles|round|rounds|loop|loops)\b",
                lowered,
            )
        ),
        run_trace_contract_detected=any(
            term in lowered for term in ("runtrace", "run_trace", "run trace")
        ),
        host_actions_detected=bool(risky_kinds),
    )
    decision = determine_audit_decision(decision_gates)
    integration_effort = (
        IntegrationEffort.LOW
        if decision is AuditDecision.DIRECT_REUSE
        else IntegrationEffort.HIGH
        if decision_gates.host_actions_detected or decision is AuditDecision.REJECT
        else IntegrationEffort.MEDIUM
    )
    decision_reason = _decision_reason(decision, decision_gates)
    return SkillAuditRecord(
        candidate_name=candidate.name,
        repository=repository,
        commit=commit,
        relative_path=skill_path.relative_to(repository_root).as_posix(),
        content_sha256=content_sha256(skill_path),
        license=repository_license,
        dependencies=dependencies,
        mechanism=(
            f"Inventory hypothesis ({candidate.evaluation_target.value}): "
            f"{candidate.intended_role}."
        ),
        input_contract=(
            "Static inspection found no exact prototype CaseBundle input contract."
        ),
        output_contract=(
            "Exact prototype Hypothesis field names were detected."
            if decision_gates.exact_hypothesis_contract_detected
            else "Mentions structured or JSON output, but not the prototype's exact "
            "Hypothesis fields."
            if output_mentions_json
            else "Natural-language/Markdown output without the prototype's Hypothesis JSON schema."
        ),
        evidence_control=(
            "Exact supplied evidence_ids and observation_ids fields were detected."
            if decision_gates.supplied_reference_enforcement_detected
            else "Mentions evidence or sources but exact supplied evidence_ids and "
            "observation_ids enforcement was not detected."
            if evidence_terms
            else "No explicit supplied-ID evidence constraint was detected."
        ),
        loop_control=(
            "A bounded iteration/cycle phrase was detected by the static gate."
            if decision_gates.bounded_loop_contract_detected
            else "Contains iteration/refinement language, but no bounded-loop phrase was detected."
            if loop_terms
            else "No explicit bounded iteration contract was detected."
        ),
        observability=(
            "A RunTrace term was detected; semantic compatibility still needs manual review."
            if decision_gates.run_trace_contract_detected
            else "Mentions outputs or reports, but no prototype RunTrace term was detected."
            if observable_terms
            else "No machine-readable trace contract was detected."
        ),
        security_risks=scan.security_findings,
        scanned_files=[path.as_posix() for path in scan.scanned_files],
        decision_gates=decision_gates,
        integration_effort=integration_effort,
        decision=decision,
        decision_reason=decision_reason,
        status=AuditStatus.VERIFIED,
    )


def determine_audit_decision(gates: DecisionGates) -> AuditDecision:
    """Derive the preliminary decision solely from declared scope and audit gates."""

    if not gates.license_compatible or not gates.scientific_reasoning_detected:
        return AuditDecision.REJECT
    if not gates.intended_for_b1:
        return AuditDecision.DESIGN_ONLY
    direct_reuse_gates = (
        gates.exact_hypothesis_contract_detected,
        gates.supplied_reference_enforcement_detected,
        gates.bounded_loop_contract_detected,
        gates.run_trace_contract_detected,
        not gates.host_actions_detected,
    )
    return (
        AuditDecision.DIRECT_REUSE
        if all(direct_reuse_gates)
        else AuditDecision.ADAPT
    )


def _decision_reason(decision: AuditDecision, gates: DecisionGates) -> str:
    if decision is AuditDecision.REJECT:
        failed = []
        if not gates.license_compatible:
            failed.append("compatible license")
        if not gates.scientific_reasoning_detected:
            failed.append("scientific reasoning signal")
        return "Preliminary reject because these required static gates failed: " + ", ".join(failed)
    if decision is AuditDecision.DESIGN_ONLY:
        return (
            "The inventory explicitly scopes this candidate as a design reference; the name "
            "was not used to infer B1 eligibility."
        )
    if decision is AuditDecision.DIRECT_REUSE:
        return (
            "All direct-reuse static gates passed; runtime smoke and manual review "
            "remain pending."
        )
    missing = []
    if not gates.exact_hypothesis_contract_detected:
        missing.append("exact Hypothesis contract")
    if not gates.supplied_reference_enforcement_detected:
        missing.append("supplied-reference enforcement")
    if not gates.bounded_loop_contract_detected:
        missing.append("bounded loop contract")
    if not gates.run_trace_contract_detected:
        missing.append("RunTrace contract")
    if gates.host_actions_detected:
        missing.append("host-action removal")
    return "B1-targeted candidate requires adaptation for: " + ", ".join(missing) + "."


def _detect_repository_license(repository_root: Path) -> str:
    license_paths = sorted(
        path for path in repository_root.iterdir() if path.name.lower().startswith("license")
    )
    if not license_paths:
        return "unknown"
    text = license_paths[0].read_text(encoding="utf-8", errors="replace").lower()
    return "MIT" if "mit license" in text else f"See {license_paths[0].name}"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(path)


def _render_comparison(report: SkillAuditReport) -> str:
    lines = [
        "# Candidate Skill Static Audit",
        "",
        f"Pinned source: `{report.repository}@{report.commit}`",
        "",
        "This table reports static inspection only. No fetched instruction or script was "
        "executed.",
        "Runtime quality, latency, and cost remain unverified until the later controlled "
        "smoke test.",
        "Evaluation target is a manually declared inventory scope, not a conclusion inferred "
        "from the candidate name. Decisions are preliminary and require manual review before "
        "runtime use.",
        "",
        "| Candidate | Status | Path | Mechanism | Evidence control | Loop control | "
        "Observability | Scanned files | Security findings | Direct gates | Integration effort | "
        "Decision |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for record in report.records:
        direct_gates = _direct_gate_score(record.decision_gates)
        cells = (
            record.candidate_name,
            record.status.value,
            record.relative_path or "—",
            record.mechanism,
            record.evidence_control,
            record.loop_control,
            record.observability,
            str(len(record.scanned_files)),
            str(len(record.security_risks)),
            direct_gates,
            record.integration_effort.value,
            record.decision.value,
        )
        lines.append("| " + " | ".join(_escape_table_cell(cell) for cell in cells) + " |")
    lines.extend(
        [
            "",
            "`adapt` means the candidate requires the externally added Hypothesis JSON contract, "
            "reference validation, bounded control flow, and RunTrace integration. It is therefore "
            "not classified as direct reuse.",
            "",
        ]
    )
    return "\n".join(lines)


def _direct_gate_score(gates: DecisionGates | None) -> str:
    if gates is None:
        return "—"
    values = (
        gates.license_compatible,
        gates.scientific_reasoning_detected,
        gates.exact_hypothesis_contract_detected,
        gates.supplied_reference_enforcement_detected,
        gates.bounded_loop_contract_detected,
        gates.run_trace_contract_detected,
        not gates.host_actions_detected,
    )
    return f"{sum(values)}/{len(values)}"


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _scan_line(relative_path: Path, line_number: int, line: str) -> list[SecurityFinding]:
    patterns = (
        (SecurityFindingKind.PACKAGE_INSTALL, _PACKAGE_INSTALL),
        (SecurityFindingKind.NETWORK_CALL, _NETWORK_CALL),
        (SecurityFindingKind.SHELL_COMMAND, _SHELL_COMMAND),
        (SecurityFindingKind.FILE_WRITE, _FILE_WRITE),
    )
    return [
        SecurityFinding(
            kind=kind,
            relative_path=relative_path.as_posix(),
            line=line_number,
            summary=line.strip()[:240],
        )
        for kind, pattern in patterns
        if pattern.search(line)
    ]


def _deduplicate_findings(findings: list[SecurityFinding]) -> list[SecurityFinding]:
    unique: dict[tuple[str, str, int], SecurityFinding] = {}
    for finding in findings:
        key = (finding.kind.value, finding.relative_path, finding.line)
        unique[key] = finding
    return [unique[key] for key in sorted(unique)]
