import json
from pathlib import Path

from agent_eval_lab.models import (
    ComparisonRanking,
    ComparisonResult,
    ComparisonScenarioResult,
    ComparisonSummary,
    EvalCase,
    EvalConfig,
    EvalRunResult,
    EvalSummary,
    PromptSpec,
)
from agent_eval_lab.runner import run_evaluation
from agent_eval_lab.report_io import write_comparison_report, write_run_report


def test_write_run_report_persists_json(tmp_path: Path):
    dataset = [
        EvalCase(case_id="1", input_text="hello", expected_answer="hello", tags=[])
    ]
    config = EvalConfig(name="smoke")

    result = run_evaluation(dataset=dataset, config=config, system_under_test=lambda prompt: prompt)
    output_path = write_run_report(result, output_dir=tmp_path)

    payload = json.loads(output_path.read_text())
    assert payload["summary"]["total_cases"] == 1
    assert payload["summary"]["passed_cases"] == 1
    assert payload["case_results"][0]["case_id"] == "1"


def test_write_run_report_persists_prompt_metadata(tmp_path: Path):
    dataset = [
        EvalCase(case_id="1", input_text="hello", expected_answer="Prompted: hello", tags=[])
    ]
    config = EvalConfig(name="prompted")
    prompt_spec = PromptSpec(
        prompt_id="baseline",
        version="v1",
        template="Prompted: {input_text}",
        description="prompted test",
    )

    result = run_evaluation(
        dataset=dataset,
        config=config,
        system_under_test=lambda prompt: prompt,
        prompt_spec=prompt_spec,
    )
    output_path = write_run_report(result, output_dir=tmp_path)

    payload = json.loads(output_path.read_text())
    assert payload["summary"]["prompt_id"] == "baseline"
    assert payload["summary"]["prompt_version"] == "v1"
    assert payload["case_results"][0]["prompt_text"] == "Prompted: hello"


def test_write_comparison_report_persists_rankings_and_baseline(tmp_path: Path):
    dataset = [EvalCase(case_id="1", input_text="hello", expected_answer="hello", tags=[])]
    config = EvalConfig(name="compare")
    eval_result = run_evaluation(dataset=dataset, config=config, system_under_test=lambda prompt: prompt)
    comparison_result = ComparisonResult(
        comparison_id="cmp-1",
        config_name="compare",
        created_at="2026-04-27T00:00:00+00:00",
        summary=ComparisonSummary(
            scenario_count=2,
            baseline_label="baseline",
            best_label="candidate",
            best_pass_rate=1.0,
        ),
        rankings=[
            ComparisonRanking(label="candidate", provider="echo", provider_model=None, prompt_id=None, prompt_version=None, pass_rate=1.0, average_score=1.0, delta_vs_baseline=0.5),
            ComparisonRanking(label="baseline", provider="static", provider_model=None, prompt_id=None, prompt_version=None, pass_rate=0.5, average_score=0.5, delta_vs_baseline=0.0),
        ],
        scenario_results=[
            ComparisonScenarioResult(label="baseline", provider="static", provider_model=None, prompt_id=None, prompt_version=None, result=eval_result),
            ComparisonScenarioResult(label="candidate", provider="echo", provider_model=None, prompt_id=None, prompt_version=None, result=eval_result),
        ],
    )

    output_path = write_comparison_report(comparison_result, output_dir=tmp_path)

    payload = json.loads(output_path.read_text())
    assert payload["summary"]["baseline_label"] == "baseline"
    assert payload["summary"]["best_label"] == "candidate"
    assert payload["rankings"][0]["label"] == "candidate"
    assert payload["rankings"][0]["delta_vs_baseline"] == 0.5
    assert payload["scenario_results"][0]["result"]["summary"]["total_cases"] == 1


def test_write_comparison_report_persists_static_provider_response(tmp_path: Path):
    dataset = [EvalCase(case_id="1", input_text="hello", expected_answer="hello", tags=[])]
    config = EvalConfig(name="compare")
    eval_result = run_evaluation(dataset=dataset, config=config, system_under_test=lambda prompt: prompt)
    comparison_result = ComparisonResult(
        comparison_id="cmp-static",
        config_name="compare",
        created_at="2026-04-27T00:00:00+00:00",
        summary=ComparisonSummary(
            scenario_count=1,
            baseline_label="static-case",
            best_label="static-case",
            best_pass_rate=1.0,
        ),
        rankings=[
            ComparisonRanking(
                label="static-case",
                provider="static",
                provider_model=None,
                prompt_id=None,
                prompt_version=None,
                pass_rate=1.0,
                average_score=1.0,
                delta_vs_baseline=0.0,
                provider_response="",
            )
        ],
        scenario_results=[
            ComparisonScenarioResult(
                label="static-case",
                provider="static",
                provider_model=None,
                prompt_id=None,
                prompt_version=None,
                result=eval_result,
                provider_response="",
            )
        ],
    )

    output_path = write_comparison_report(comparison_result, output_dir=tmp_path)

    payload = json.loads(output_path.read_text())
    assert payload["rankings"][0]["provider_response"] == ""
    assert payload["scenario_results"][0]["provider_response"] == ""


def test_comparison_dataclasses_preserve_legacy_positional_order():
    eval_result = EvalRunResult(
        run_id="run-1",
        config_name="compare",
        created_at="2026-04-27T00:00:00+00:00",
        summary=EvalSummary(
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            pass_rate=1.0,
            average_score=1.0,
        ),
        case_results=[],
    )

    scenario_result = ComparisonScenarioResult(
        "legacy",
        "echo",
        None,
        "prompt-a",
        "v1",
        eval_result,
    )
    ranking = ComparisonRanking(
        "legacy",
        "echo",
        None,
        "prompt-a",
        "v1",
        1.0,
        1.0,
        0.0,
    )

    assert scenario_result.prompt_id == "prompt-a"
    assert scenario_result.prompt_version == "v1"
    assert scenario_result.result is eval_result
    assert scenario_result.provider_response is None
    assert ranking.prompt_id == "prompt-a"
    assert ranking.prompt_version == "v1"
    assert ranking.pass_rate == 1.0
    assert ranking.average_score == 1.0
    assert ranking.delta_vs_baseline == 0.0
    assert ranking.provider_response is None
