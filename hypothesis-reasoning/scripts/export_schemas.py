"""Export deterministic JSON Schemas from the Pydantic contract models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from hypothesis_reasoning.io import write_json_atomic  # noqa: E402
from hypothesis_reasoning.models import (  # noqa: E402
    Critique,
    EvidenceItem,
    Hypothesis,
    ObservationRecord,
    ResearchContext,
    RunTrace,
    ScoreCard,
    TopicCandidate,
    TopicDecision,
)

SCHEMAS = {
    "research-context": ResearchContext,
    "evidence-item": EvidenceItem,
    "observation-record": ObservationRecord,
    "topic-candidate": TopicCandidate,
    "topic-decision": TopicDecision,
    "hypothesis": Hypothesis,
    "critique": Critique,
    "score-card": ScoreCard,
    "run-trace": RunTrace,
}


def export_schemas(output_dir: Path = PROJECT_DIR / "schemas") -> None:
    """Write one canonical JSON Schema file for each public contract model."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in SCHEMAS.items():
        write_json_atomic(output_dir / f"{name}.schema.json", model.model_json_schema())


def main() -> None:
    """Export schemas to the committed directory or an explicit output directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", nargs="?", type=Path, default=PROJECT_DIR / "schemas")
    arguments = parser.parse_args()
    export_schemas(arguments.output_dir)


if __name__ == "__main__":
    main()
