from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence
from uuid import uuid4

from agent_eval_lab.models import (
    ComparisonRanking,
    ComparisonResult,
    ComparisonScenario,
    ComparisonScenarioResult,
    ComparisonSummary,
    EvalCase,
    EvalConfig,
    PromptSpec,
    utc_timestamp,
)
from agent_eval_lab.prompt_registry import select_prompt_spec
from agent_eval_lab.providers import build_system_under_test
from agent_eval_lab.runner import run_evaluation


def load_comparison_scenarios(path: Path) -> List[ComparisonScenario]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError("Scenario registry must be a non-empty JSON list")

    scenarios = [ComparisonScenario(**item) for item in payload]
    validate_comparison_scenarios(scenarios)
    return scenarios


def validate_comparison_scenarios(scenarios: Sequence[ComparisonScenario]) -> None:
    if not scenarios:
        raise ValueError("Comparison requires at least one scenario")

    seen_labels: set[str] = set()

    for scenario in scenarios:
        if scenario.label in seen_labels:
            raise ValueError(f"Duplicate scenario label: {scenario.label}")
        seen_labels.add(scenario.label)

        if scenario.prompt_version is not None and scenario.prompt_id is None:
            raise ValueError("prompt_id is required when scenario specifies prompt_version")

        if scenario.provider == "static" and scenario.provider_response is None:
            raise ValueError("static scenarios require provider_response")


def run_comparison(
    *,
    dataset: Iterable[EvalCase],
    config: EvalConfig,
    scenarios: Sequence[ComparisonScenario | dict],
    prompt_specs: Sequence[PromptSpec] | None = None,
) -> ComparisonResult:
    normalized_scenarios = [
        scenario if isinstance(scenario, ComparisonScenario) else ComparisonScenario(**scenario)
        for scenario in scenarios
    ]
    validate_comparison_scenarios(normalized_scenarios)

    prompt_specs = list(prompt_specs or [])
    dataset = list(dataset)

    scenario_results: List[ComparisonScenarioResult] = []
    for scenario in normalized_scenarios:
        prompt_spec = None
        if scenario.prompt_id is not None:
            if not prompt_specs:
                raise ValueError("prompt-file is required when a comparison scenario selects a prompt")
            prompt_spec = select_prompt_spec(
                prompt_specs,
                prompt_id=scenario.prompt_id,
                version=scenario.prompt_version,
            )

        provider_config = {}
        if scenario.provider_response is not None:
            provider_config["response_text"] = scenario.provider_response
        if scenario.provider_model is not None:
            provider_config["model"] = scenario.provider_model

        system_under_test = build_system_under_test(scenario.provider, provider_config)
        eval_result = run_evaluation(
            dataset=dataset,
            config=config,
            system_under_test=system_under_test,
            prompt_spec=prompt_spec,
        )
        scenario_results.append(
            ComparisonScenarioResult(
                label=scenario.label,
                provider=scenario.provider,
                provider_model=scenario.provider_model,
                prompt_id=scenario.prompt_id,
                prompt_version=scenario.prompt_version,
                result=eval_result,
            )
        )

    baseline = scenario_results[0]
    baseline_pass_rate = baseline.result.summary.pass_rate
    rankings = [
        ComparisonRanking(
            label=item.label,
            provider=item.provider,
            provider_model=item.provider_model,
            prompt_id=item.prompt_id,
            prompt_version=item.prompt_version,
            pass_rate=item.result.summary.pass_rate,
            average_score=item.result.summary.average_score,
            delta_vs_baseline=item.result.summary.pass_rate - baseline_pass_rate,
        )
        for item in scenario_results
    ]
    rankings.sort(key=lambda item: (-item.pass_rate, -item.average_score, item.label))
    best = rankings[0]

    return ComparisonResult(
        comparison_id=str(uuid4()),
        config_name=config.name,
        created_at=utc_timestamp(),
        summary=ComparisonSummary(
            scenario_count=len(scenario_results),
            baseline_label=baseline.label,
            best_label=best.label,
            best_pass_rate=best.pass_rate,
        ),
        rankings=rankings,
        scenario_results=scenario_results,
    )
