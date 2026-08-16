from __future__ import annotations

from collections import Counter
from decimal import Decimal
from typing import Any

import pytest

from hypothesis_reasoning.errors import InputReferenceError
from hypothesis_reasoning.models import (
    Critique,
    Hypothesis,
    TopicCandidate,
    TopicDecision,
)
from hypothesis_reasoning.skill_adapters.base import RunContext
from tests.factories import (
    hypothesis_payload,
    topic_candidate_payload,
    topic_decision_payload,
)


def run_context() -> RunContext:
    return RunContext(
        run_id="run-task7-pipeline",
        case_id="case-001",
        method_id="B3",
        requested_model="qwen-max",
        random_seed=41,
        temperature=0.2,
        prompt_version="pipeline-v1",
        budget_partition="development",
        estimated_cost_cny=Decimal("0.10"),
        experiment_batch="task7-tests",
    )


def selected_topic() -> TopicCandidate:
    return TopicCandidate.model_validate(topic_candidate_payload())


def selected_decision(**overrides: Any) -> TopicDecision:
    return TopicDecision.model_validate(topic_decision_payload(**overrides))


def abstained_decision() -> TopicDecision:
    return TopicDecision.model_validate(
        topic_decision_payload(
            selected_topic_id=None,
            ranked_topic_ids=[],
            scores=[],
            rejection_reasons=[],
            status="abstained",
        )
    )


def candidate(channel: str, **overrides: Any) -> Hypothesis:
    payload = hypothesis_payload(
        hypothesis_id=f"hypothesis-{channel}",
        statement=f"The {channel} channel proposes a distinct timing explanation.",
        mechanism=f"The {channel} mechanism changes the measured timing residual.",
    )
    payload.update(overrides)
    return Hypothesis.model_validate(payload)


class FakeGenerator:
    def __init__(
        self,
        name: str,
        output: list[Hypothesis] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.output = output if output is not None else [candidate(name)]
        self.error = error
        self.calls: list[str] = []

    def generate(
        self,
        bundle: Any,
        topic: TopicCandidate,
        context: RunContext,
    ) -> list[Hypothesis]:
        self.calls.append(topic.topic_id)
        if self.error is not None:
            raise self.error
        return list(self.output)


class FakeCritic:
    def __init__(self, severity: str | None = None) -> None:
        self.severity = severity
        self.calls: list[tuple[str, bool]] = []

    def review(
        self,
        hypothesis: Hypothesis,
        bundle: Any,
        context: RunContext,
        *,
        include_evidence_review: bool = True,
    ) -> list[Critique]:
        self.calls.append((hypothesis.hypothesis_id, include_evidence_review))
        if self.severity is None:
            return []
        return [
            Critique(
                critique_id=f"critique-{hypothesis.hypothesis_id}",
                hypothesis_id=hypothesis.hypothesis_id,
                issue_type="logical_gap",
                severity=self.severity,
                target_field="mechanism",
                evidence_ids=["evidence-001"],
                description="The mechanism skips a causal step.",
                actionable_revision="Add the missing causal step.",
                decision="reject" if self.severity == "blocking" else "revise",
            )
        ]


class FakeRefiner:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def revise(
        self,
        hypothesis: Hypothesis,
        critiques: list[Critique],
        bundle: Any,
        context: RunContext,
    ) -> Hypothesis:
        self.calls.append(hypothesis.hypothesis_id)
        return hypothesis.model_copy(
            update={
                "hypothesis_id": f"{hypothesis.hypothesis_id}-r1",
                "parent_id": hypothesis.hypothesis_id,
            }
        )


def dependencies(
    *,
    critic: FakeCritic | None = None,
    refiner: FakeRefiner | None = None,
    inductive: FakeGenerator | None = None,
    deductive: FakeGenerator | None = None,
) -> dict[str, Any]:
    return {
        "topics": [selected_topic()],
        "baseline_generator": FakeGenerator("baseline"),
        "skill_generator": FakeGenerator("skill"),
        "inductive_generator": inductive or FakeGenerator("inductive"),
        "deductive_generator": deductive or FakeGenerator("deductive"),
        "a1_inductive_generator": FakeGenerator(
            "a1-inductive",
            [
                candidate(
                    "a1-inductive",
                    evidence_ids=[],
                    observation_ids=[],
                    predictions=[],
                    falsification_criteria=[],
                    alternative_explanations=[],
                    unsupported_claims=["Intentionally unsupported for the ablation."],
                    confidence=0.95,
                )
            ],
        ),
        "a1_deductive_generator": FakeGenerator("a1-deductive"),
        "critic": critic or FakeCritic(),
        "refiner": refiner or FakeRefiner(),
        "a1_refiner": FakeRefiner(),
    }


@pytest.mark.parametrize(
    ("mode", "called", "critic_calls", "refiner_calls"),
    [
        ("B0", {"baseline"}, 0, 0),
        ("B1", {"skill"}, 0, 0),
        ("B2", {"inductive", "deductive"}, 0, 0),
        ("B3", {"inductive", "deductive"}, 2, 0),
        ("A1", {"a1-inductive", "a1-deductive"}, 2, 0),
        ("A2", {"inductive", "deductive"}, 0, 0),
        ("A3", {"inductive", "deductive"}, 2, 0),
    ],
)
def test_pipeline_maps_each_mode_explicitly(
    valid_bundle: Any,
    mode: str,
    called: set[str],
    critic_calls: int,
    refiner_calls: int,
) -> None:
    from hypothesis_reasoning.pipeline import ExperimentMode, Pipeline, PipelineStatus

    deps = dependencies()
    pipeline = Pipeline(**deps)

    result = pipeline.run(
        valid_bundle,
        selected_decision(),
        ExperimentMode(mode),
        run_context(),
    )

    generators = {
        value.name: value
        for key, value in deps.items()
        if key.endswith("generator")
    }
    assert {name for name, generator in generators.items() if generator.calls} == called
    assert len(deps["critic"].calls) == critic_calls
    assert len(deps["refiner"].calls) == refiner_calls
    assert result.status is PipelineStatus.COMPLETED
    if mode == "A1":
        assert all(not evidence_review for _, evidence_review in deps["critic"].calls)


def test_abstained_topic_returns_without_generator_calls(valid_bundle: Any) -> None:
    from hypothesis_reasoning.pipeline import ExperimentMode, Pipeline, PipelineStatus

    deps = dependencies()

    result = Pipeline(**deps).run(
        valid_bundle,
        abstained_decision(),
        ExperimentMode.B3,
        run_context(),
    )

    assert result.status is PipelineStatus.ABSTAINED
    assert result.hypotheses == []
    assert not any(
        value.calls for key, value in deps.items() if key.endswith("generator")
    )


def test_missing_selected_topic_is_refused_before_generation(valid_bundle: Any) -> None:
    from hypothesis_reasoning.pipeline import ExperimentMode, Pipeline

    deps = dependencies()
    decision = selected_decision(
        selected_topic_id="topic-missing",
        ranked_topic_ids=["topic-missing"],
        scores=[
            {
                "topic_id": "topic-missing",
                "evidence_sufficiency": 4,
                "scientific_value_proxy": 4,
                "testability": 4,
                "data_availability": 4,
                "validation_cost": 4,
                "novelty_proxy": 4,
                "uncertainty": 4,
                "reason": "A deliberately missing frozen topic.",
            }
        ],
        rejection_reasons=[],
    )

    with pytest.raises(InputReferenceError, match="topic-missing"):
        Pipeline(**deps).run(
            valid_bundle,
            decision,
            ExperimentMode.B3,
            run_context(),
        )
    assert not any(
        value.calls for key, value in deps.items() if key.endswith("generator")
    )


@pytest.mark.parametrize("mode", ["B2", "B3"])
def test_one_dual_generator_failure_degrades_and_continues(
    valid_bundle: Any,
    mode: str,
) -> None:
    from hypothesis_reasoning.pipeline import ExperimentMode, Pipeline, PipelineStatus

    critic = FakeCritic()
    deps = dependencies(
        critic=critic,
        inductive=FakeGenerator("inductive", error=RuntimeError("inductive failed")),
    )

    result = Pipeline(**deps).run(
        valid_bundle,
        selected_decision(),
        ExperimentMode(mode),
        run_context(),
    )

    assert result.status is PipelineStatus.COMPLETED
    assert result.degraded is True
    assert result.failed_stages == ["generation_inductive"]
    assert [item.hypothesis_id for item in result.hypotheses] == [
        "hypothesis-deductive"
    ]
    assert len(critic.calls) == (1 if mode == "B3" else 0)


@pytest.mark.parametrize("mode", ["B2", "B3"])
def test_both_dual_generators_fail_without_critique_calls(
    valid_bundle: Any,
    mode: str,
) -> None:
    from hypothesis_reasoning.pipeline import ExperimentMode, Pipeline, PipelineStatus

    critic = FakeCritic()
    deps = dependencies(
        critic=critic,
        inductive=FakeGenerator("inductive", error=RuntimeError("inductive failed")),
        deductive=FakeGenerator("deductive", error=RuntimeError("deductive failed")),
    )

    result = Pipeline(**deps).run(
        valid_bundle,
        selected_decision(),
        ExperimentMode(mode),
        run_context(),
    )

    assert result.status is PipelineStatus.FAILED
    assert result.degraded is True
    assert result.hypotheses == []
    assert result.failed_stages == [
        "generation_inductive",
        "generation_deductive",
    ]
    assert critic.calls == []


def test_b3_revises_each_hypothesis_no_more_than_once(valid_bundle: Any) -> None:
    from hypothesis_reasoning.pipeline import ExperimentMode, Pipeline

    critic = FakeCritic(severity="high")
    refiner = FakeRefiner()
    deps = dependencies(critic=critic, refiner=refiner)

    result = Pipeline(**deps).run(
        valid_bundle,
        selected_decision(),
        ExperimentMode.B3,
        run_context(),
    )

    counts = Counter(refiner.calls)
    assert counts == {"hypothesis-inductive": 1, "hypothesis-deductive": 1}
    assert result.revision_count == 2
    assert all(item.parent_id is not None for item in result.hypotheses)


def test_a3_runs_critique_but_never_refines(valid_bundle: Any) -> None:
    from hypothesis_reasoning.pipeline import ExperimentMode, Pipeline

    critic = FakeCritic(severity="high")
    refiner = FakeRefiner()
    deps = dependencies(critic=critic, refiner=refiner)

    result = Pipeline(**deps).run(
        valid_bundle,
        selected_decision(),
        ExperimentMode.A3,
        run_context(),
    )

    assert len(critic.calls) == 2
    assert refiner.calls == []
    assert result.revision_count == 0


def test_a2_skips_model_critique_but_keeps_gate_driven_refinement(
    valid_bundle: Any,
) -> None:
    from hypothesis_reasoning.pipeline import ExperimentMode, Pipeline

    critic = FakeCritic(severity="high")
    refiner = FakeRefiner()
    incomplete = FakeGenerator(
        "inductive",
        [candidate("inductive", falsification_criteria=[])],
    )
    deps = dependencies(critic=critic, refiner=refiner, inductive=incomplete)

    result = Pipeline(**deps).run(
        valid_bundle,
        selected_decision(),
        ExperimentMode.A2,
        run_context(),
    )

    assert critic.calls == []
    assert refiner.calls == ["hypothesis-inductive"]
    assert result.revision_count == 1


def test_a1_uses_relaxed_refiner_for_non_evidence_critique(valid_bundle: Any) -> None:
    from hypothesis_reasoning.pipeline import ExperimentMode, Pipeline

    critic = FakeCritic(severity="high")
    deps = dependencies(critic=critic)

    result = Pipeline(**deps).run(
        valid_bundle,
        selected_decision(),
        ExperimentMode.A1,
        run_context(),
    )

    assert deps["refiner"].calls == []
    assert deps["a1_refiner"].calls == [
        "hypothesis-a1-inductive",
        "hypothesis-a1-deductive",
    ]
    assert result.revision_count == 2


def test_blocking_gate_rejects_before_model_critique(valid_bundle: Any) -> None:
    from hypothesis_reasoning.pipeline import ExperimentMode, Pipeline, PipelineStatus

    critic = FakeCritic()
    unsafe = FakeGenerator(
        "inductive",
        [candidate("unsafe", evidence_ids=["invented-evidence"])],
    )
    deps = dependencies(critic=critic, inductive=unsafe)

    result = Pipeline(**deps).run(
        valid_bundle,
        selected_decision(),
        ExperimentMode.B3,
        run_context(),
    )

    assert result.status is PipelineStatus.COMPLETED
    assert [item.hypothesis_id for item in result.hypotheses] == [
        "hypothesis-deductive"
    ]
    assert len(critic.calls) == 1
    assert any(item.issue_type == "unknown_evidence_reference" for item in result.critiques)


def test_pipeline_rejects_more_than_one_configured_round() -> None:
    from hypothesis_reasoning.pipeline import Pipeline

    with pytest.raises(ValueError, match="exactly one"):
        Pipeline(**dependencies(), max_refinement_rounds=2)
