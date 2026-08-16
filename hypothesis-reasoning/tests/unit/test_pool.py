from __future__ import annotations

from decimal import Decimal

from hypothesis_reasoning.models import Hypothesis
from hypothesis_reasoning.skill_adapters.base import RunContext
from tests.factories import hypothesis_payload


def run_context() -> RunContext:
    return RunContext(
        run_id="run-task6-pool",
        case_id="case-001",
        method_id="B2",
        requested_model="qwen-max",
        random_seed=23,
        temperature=0.2,
        prompt_version="generation-v1",
        budget_partition="development",
        estimated_cost_cny=Decimal("0.10"),
        experiment_batch="task6-tests",
    )


def stellar_activity_hypothesis() -> Hypothesis:
    return Hypothesis.model_validate(
        hypothesis_payload(
            hypothesis_id="hypothesis-stellar-activity",
            reasoning_type="inductive",
            statement="The observed dimming pattern has an astrophysical origin.",
            mechanism=(
                "Rotating starspots modulate the stellar surface brightness and "
                "produce the observed dimming pattern."
            ),
            evidence_ids=["evidence-stellar-activity"],
            falsification_criteria=[
                "The dimming remains phase-stable while independent activity "
                "indicators show no corresponding modulation."
            ],
        )
    )


def blend_hypothesis() -> Hypothesis:
    return Hypothesis.model_validate(
        hypothesis_payload(
            hypothesis_id="hypothesis-background-blend",
            reasoning_type="deductive",
            statement="The observed dimming pattern has an astrophysical origin.",
            mechanism=(
                "Flux from a background eclipsing binary is blended with the target "
                "and dilutes its eclipse into the observed dimming pattern."
            ),
            evidence_ids=["evidence-background-blend"],
            falsification_criteria=[
                "High-resolution imaging excludes contaminating sources throughout "
                "the photometric aperture."
            ],
        )
    )


def test_same_statement_different_mechanism_is_preserved() -> None:
    from hypothesis_reasoning.pool import HypothesisPool

    result = HypothesisPool().consolidate(
        [stellar_activity_hypothesis(), blend_hypothesis()]
    )

    assert result == [stellar_activity_hypothesis(), blend_hypothesis()]


def test_unicode_and_whitespace_equivalent_candidates_are_merged() -> None:
    from hypothesis_reasoning.pool import HypothesisPool

    first = Hypothesis.model_validate(
        hypothesis_payload(
            hypothesis_id="hypothesis-spacing-a",
            statement="A  periodic\u00a0signal marks a timing deviation.",
            mechanism="A companion produces Ａ periodic perturbation.",
        )
    )
    second = Hypothesis.model_validate(
        hypothesis_payload(
            hypothesis_id="hypothesis-spacing-b",
            statement="A periodic signal marks a timing deviation.",
            mechanism="A companion produces A periodic perturbation.",
        )
    )

    result = HypothesisPool().consolidate([first, second])

    assert result == [first]


def test_candidates_at_both_similarity_thresholds_are_merged() -> None:
    from hypothesis_reasoning.pool import HypothesisPool

    statement = "".join(chr(0x4E00 + index) for index in range(59))
    changed_statement = statement[:29] + chr(0x9FFF) + statement[30:]
    first = Hypothesis.model_validate(
        hypothesis_payload(
            hypothesis_id="hypothesis-threshold-a",
            statement=statement,
            mechanism=statement,
        )
    )
    second = Hypothesis.model_validate(
        hypothesis_payload(
            hypothesis_id="hypothesis-threshold-b",
            statement=changed_statement,
            mechanism=changed_statement,
        )
    )

    result = HypothesisPool().consolidate([first, second])

    assert result == [first]


def test_candidate_below_either_similarity_threshold_is_preserved() -> None:
    from hypothesis_reasoning.pool import HypothesisPool

    statement = "".join(chr(0x4E00 + index) for index in range(58))
    changed_statement = statement[:29] + chr(0x9FFF) + statement[30:]
    first = Hypothesis.model_validate(
        hypothesis_payload(
            hypothesis_id="hypothesis-below-threshold-a",
            statement=statement,
            mechanism="The same mechanism applies in both candidates.",
        )
    )
    second = Hypothesis.model_validate(
        hypothesis_payload(
            hypothesis_id="hypothesis-below-threshold-b",
            statement=changed_statement,
            mechanism="The same mechanism applies in both candidates.",
        )
    )

    result = HypothesisPool().consolidate([first, second])

    assert result == [first, second]


def test_duplicate_preserves_more_complete_candidate_and_unions_evidence() -> None:
    from hypothesis_reasoning.pool import HypothesisPool

    sparse = Hypothesis.model_validate(
        hypothesis_payload(
            hypothesis_id="hypothesis-sparse",
            evidence_ids=["evidence-sparse"],
            assumptions=[],
            scope_conditions=[],
            predictions=[],
            falsification_criteria=[],
            alternative_explanations=[],
            validation_plan={
                "method": "Compare the candidates.",
                "baselines": [],
                "metrics": [],
                "required_data": [],
            },
        )
    )
    complete = Hypothesis.model_validate(
        hypothesis_payload(
            hypothesis_id="hypothesis-complete",
            evidence_ids=["evidence-complete"],
        )
    )
    expected_payload = complete.model_dump(mode="python")
    expected_payload["evidence_ids"] = ["evidence-complete", "evidence-sparse"]
    expected = Hypothesis.model_validate(expected_payload)

    result = HypothesisPool().consolidate([sparse, complete])

    assert result == [expected]
