import json
from pathlib import Path

from agent_eval_lab.models import EvalCase, EvalConfig, PromptSpec
from agent_eval_lab.comparison import load_comparison_scenarios, run_comparison


def test_load_comparison_scenarios_reads_valid_registry(tmp_path: Path):
    scenario_path = tmp_path / "scenarios.json"
    scenario_path.write_text(
        json.dumps(
            [
                {
                    "label": "echo-baseline",
                    "provider": "echo"
                },
                {
                    "label": "static-answer",
                    "provider": "static",
                    "provider_response": "fixed answer"
                }
            ]
        )
    )

    scenarios = load_comparison_scenarios(scenario_path)

    assert [scenario.label for scenario in scenarios] == ["echo-baseline", "static-answer"]
    assert scenarios[1].provider_response == "fixed answer"


def test_load_comparison_scenarios_rejects_duplicate_labels(tmp_path: Path):
    scenario_path = tmp_path / "scenarios.json"
    scenario_path.write_text(
        json.dumps(
            [
                {"label": "dup", "provider": "echo"},
                {"label": "dup", "provider": "static", "provider_response": "x"}
            ]
        )
    )

    try:
        load_comparison_scenarios(scenario_path)
    except ValueError as exc:
        assert "Duplicate scenario label" in str(exc)
    else:
        raise AssertionError("Expected duplicate scenario labels to be rejected")


def test_load_comparison_scenarios_rejects_prompt_version_without_prompt_id(tmp_path: Path):
    scenario_path = tmp_path / "scenarios.json"
    scenario_path.write_text(
        json.dumps(
            [
                {
                    "label": "bad-prompt",
                    "provider": "echo",
                    "prompt_version": "v2"
                }
            ]
        )
    )

    try:
        load_comparison_scenarios(scenario_path)
    except ValueError as exc:
        assert "prompt_id" in str(exc)
    else:
        raise AssertionError("Expected prompt_version without prompt_id to be rejected")


def test_load_comparison_scenarios_rejects_static_without_provider_response(tmp_path: Path):
    scenario_path = tmp_path / "scenarios.json"
    scenario_path.write_text(
        json.dumps(
            [
                {
                    "label": "bad-static",
                    "provider": "static"
                }
            ]
        )
    )

    try:
        load_comparison_scenarios(scenario_path)
    except ValueError as exc:
        assert "provider_response" in str(exc)
    else:
        raise AssertionError("Expected static scenario without provider_response to be rejected")


def test_run_comparison_ranks_scenarios_and_computes_baseline_deltas():
    dataset = [
        EvalCase(case_id="1", input_text="hello", expected_answer="hello"),
        EvalCase(case_id="2", input_text="agentic harness", expected_answer="agentic", validator_name="contains"),
    ]
    config = EvalConfig(name="compare-smoke")
    prompt_specs = [
        PromptSpec(
            prompt_id="prefix",
            version="v1",
            template="Prompted: {input_text}",
            description="prefix prompt",
        )
    ]
    scenarios = [
        {
            "label": "baseline-echo",
            "provider": "echo",
        },
        {
            "label": "prompted-echo",
            "provider": "echo",
            "prompt_id": "prefix",
            "prompt_version": "v1",
        },
        {
            "label": "static-bad",
            "provider": "static",
            "provider_response": "wrong",
        },
    ]

    result = run_comparison(
        dataset=dataset,
        config=config,
        scenarios=scenarios,
        prompt_specs=prompt_specs,
    )

    assert result.summary.baseline_label == "baseline-echo"
    assert result.summary.best_label == "baseline-echo"
    assert [item.label for item in result.rankings] == ["baseline-echo", "prompted-echo", "static-bad"]
    assert result.rankings[0].pass_rate == 1.0
    assert result.rankings[1].pass_rate == 0.5
    assert result.rankings[0].delta_vs_baseline == 0.0
    assert result.rankings[2].delta_vs_baseline == -1.0
    assert result.scenario_results[1].result.summary.prompt_id == "prefix"
    assert result.scenario_results[1].result.summary.prompt_version == "v1"
    assert result.scenario_results[1].result.case_results[0].prompt_text == "Prompted: hello"


def test_run_comparison_rejects_empty_scenario_sequence():
    dataset = [EvalCase(case_id="1", input_text="hello", expected_answer="hello")]
    config = EvalConfig(name="compare-smoke")

    try:
        run_comparison(dataset=dataset, config=config, scenarios=[])
    except ValueError as exc:
        assert "at least one scenario" in str(exc)
    else:
        raise AssertionError("Expected empty comparison scenarios to be rejected")


def test_run_comparison_supports_empty_string_static_response():
    dataset = [EvalCase(case_id="1", input_text="hello", expected_answer="")]
    config = EvalConfig(name="compare-smoke")

    result = run_comparison(
        dataset=dataset,
        config=config,
        scenarios=[{"label": "empty-static", "provider": "static", "provider_response": ""}],
    )

    assert result.summary.best_label == "empty-static"
    assert result.rankings[0].pass_rate == 1.0
    assert result.scenario_results[0].result.case_results[0].actual_answer == ""
    assert result.scenario_results[0].result.case_results[0].passed is True
