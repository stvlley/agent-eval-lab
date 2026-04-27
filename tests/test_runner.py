from agent_eval_lab.models import EvalCase, EvalConfig
from agent_eval_lab.runner import run_evaluation


def test_run_evaluation_records_pass_fail_and_summary():
    dataset = [
        EvalCase(
            case_id="sum-1",
            input_text="2 + 2",
            expected_answer="4",
            tags=["math"],
        ),
        EvalCase(
            case_id="capital-1",
            input_text="Capital of France",
            expected_answer="Paris",
            tags=["geography"],
        ),
    ]

    config = EvalConfig(name="baseline")

    def system_under_test(prompt: str) -> str:
        outputs = {
            "2 + 2": "4",
            "Capital of France": "Lyon",
        }
        return outputs[prompt]

    result = run_evaluation(dataset=dataset, config=config, system_under_test=system_under_test)

    assert result.summary.total_cases == 2
    assert result.summary.passed_cases == 1
    assert result.summary.failed_cases == 1
    assert result.summary.pass_rate == 0.5
    assert result.summary.average_score == 0.5
    assert result.summary.failure_categories["wrong_answer"] == 1
    assert result.case_results[0].passed is True
    assert result.case_results[1].passed is False
    assert result.case_results[1].failure_reason == "exact_match_failed"
    assert result.case_results[1].failure_category == "wrong_answer"


def test_run_evaluation_uses_case_level_validator_settings():
    dataset = [
        EvalCase(
            case_id="contains-1",
            input_text="summary",
            expected_answer="agentic",
            validator_name="contains",
        )
    ]

    config = EvalConfig(name="contains-suite")
    result = run_evaluation(dataset=dataset, config=config, system_under_test=lambda _: "agentic harness building")

    assert result.summary.passed_cases == 1
    assert result.case_results[0].passed is True
