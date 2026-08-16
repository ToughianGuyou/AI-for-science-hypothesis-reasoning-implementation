"""Command-line entry point for the hypothesis-reasoning prototype."""

from pathlib import Path
from typing import Annotated

import httpx
import typer

from hypothesis_reasoning.errors import HypothesisReasoningError
from hypothesis_reasoning.research.audit import audit_inventory

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@app.callback()
def main() -> None:
    """Evidence-grounded hypothesis reasoning utilities."""


@app.command("audit-skills")
def audit_skills(
    inventory: Annotated[
        Path,
        typer.Option(
            "--inventory",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Machine-readable candidate inventory.",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Static audit JSON output."),
    ] = Path("results/skill-audit.json"),
    schema: Annotated[
        Path,
        typer.Option("--schema", help="Generated SkillAuditReport JSON Schema."),
    ] = Path("research/audit_schema.json"),
    comparison: Annotated[
        Path,
        typer.Option("--comparison", help="Human-readable comparison table."),
    ] = Path("docs/skill-comparison.md"),
    cache_root: Annotated[
        Path,
        typer.Option(
            "--cache-root",
            help="Ignored cache for pinned, non-executed external sources.",
        ),
    ] = Path(".cache/skills"),
) -> None:
    """Resolve, fetch, and statically audit candidate scientific skills."""

    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(30.0),
            headers={"Accept": "application/vnd.github+json"},
        ) as client:
            report = audit_inventory(
                inventory_path=inventory,
                output_path=out,
                schema_path=schema,
                comparison_path=comparison,
                cache_root=cache_root,
                client=client,
            )
    except (HypothesisReasoningError, httpx.HTTPError, OSError) as error:
        typer.echo(f"skill audit failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    verified = sum(record.status.value == "verified" for record in report.records)
    not_found = sum(record.status.value == "not_found" for record in report.records)
    typer.echo(
        f"audited {len(report.records)} candidates at {report.commit}: "
        f"verified={verified}, not_found={not_found}"
    )


if __name__ == "__main__":
    app()
