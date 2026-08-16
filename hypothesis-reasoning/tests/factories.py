"""Reusable payload factories for contract and integration tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def context_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "context_id": "case-001",
        "domain_scope": "基于公开论文和观测数据的天文学研究",
        "topic_constraints": ["仅使用公开论文和观测摘要", "不得预设唯一研究问题"],
        "available_resources": {"budget_cny": 300, "evidence": True, "observations": True},
        "max_topic_candidates": 3,
        "selection_policy_id": "selection-v1",
    }
    payload.update(overrides)
    return payload


def evidence_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "evidence_id": "evidence-001",
        "paper_title": "A public astronomy study",
        "identifier": "doi:10.0000/example",
        "location": {"page": 2, "section": "Results"},
        "source_text": "The measured signal changes over time.",
        "extracted_fact": "The signal changes over time.",
        "scope": "The observed sample only.",
        "verification_status": "source_located",
    }
    payload.update(overrides)
    return payload


def observation_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "observation_id": "observation-001",
        "target_id": "target-001",
        "data_source": "public-survey",
        "modality": "time_series",
        "summary": "A periodic signal contains an unexplained deviation.",
        "features": [{"name": "period_days", "value": 3.2, "unit": "day", "uncertainty": 0.1}],
        "anomaly_tags": ["timing-deviation"],
        "method": "upstream feature extraction",
        "provenance": {"dataset_version": "v1", "processing_version": "extractor-1"},
    }
    payload.update(overrides)
    return payload


def topic_candidate_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "topic_id": "topic-001",
        "research_question": "What mechanism could explain the observed timing deviation?",
        "gap_type": "anomaly",
        "motivation": "The deviation is not explained by the supplied evidence.",
        "evidence_ids": ["evidence-001"],
        "observation_ids": ["observation-001"],
        "expected_validation": ["Compare competing timing models."],
        "risks": ["The observation may contain an unmodeled systematic effect."],
    }
    payload.update(overrides)
    return payload


def topic_decision_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "selected_topic_id": "topic-001",
        "ranked_topic_ids": ["topic-001", "topic-002"],
        "scores": [
            {
                "topic_id": "topic-001",
                "evidence_sufficiency": 3,
                "scientific_value_proxy": 3,
                "testability": 4,
                "data_availability": 3,
                "validation_cost": 2,
                "novelty_proxy": 2,
                "uncertainty": 1,
                "reason": "The question has direct evidence and a testable validation path.",
            }
        ],
        "selection_reason": "The selected topic has the strongest evidence and validation path.",
        "rejection_reasons": [{"topic_id": "topic-002", "reason": "Higher validation cost."}],
        "status": "selected",
    }
    payload.update(overrides)
    return payload


def hypothesis_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "hypothesis_id": "hypothesis-001",
        "parent_id": None,
        "topic_id": "topic-001",
        "reasoning_type": "hybrid",
        "statement": "The timing deviation is caused by an additional perturbing body.",
        "mechanism": "A perturbing body changes the expected timing pattern.",
        "evidence_ids": ["evidence-001"],
        "observation_ids": ["observation-001"],
        "assumptions": ["The extracted timing feature is reliable."],
        "scope_conditions": ["The hypothesis applies to target-001."],
        "predictions": [
            {
                "description": "The timing residual should show a periodic trend.",
                "variable": "timing_residual",
                "expected_direction": "periodic",
                "falsifiable": True,
            }
        ],
        "falsification_criteria": [
            "No periodic residual remains after the proposed model is fitted."
        ],
        "alternative_explanations": ["An instrumental systematic could cause the deviation."],
        "validation_plan": {
            "method": "Fit competing timing models and compare residuals.",
            "baselines": ["constant-period model"],
            "metrics": ["residual_error"],
            "required_data": ["time-series observations"],
        },
        "unsupported_claims": [],
        "confidence": 0.55,
        "status": "proposed",
    }
    payload.update(overrides)
    return payload


def write_case_fixture(directory: Path, **overrides: Any) -> Path:
    """Write a minimal case bundle for later loader tests."""

    directory.mkdir(parents=True, exist_ok=True)
    evidence = overrides.get("evidence", [evidence_payload()])
    observations = overrides.get("observations", [observation_payload()])
    if isinstance(evidence, dict):
        evidence = [evidence_payload(**evidence)]
    if isinstance(observations, dict):
        observations = [observation_payload(**observations)]
    files = {
        "research_context.json": context_payload(**overrides.get("context", {})),
        "evidence_items.json": evidence,
        "observation_records.json": observations,
    }
    for filename, payload in files.items():
        (directory / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return directory
