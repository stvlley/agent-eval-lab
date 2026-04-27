from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable, List
from uuid import uuid4

from agent_eval_lab.models import CaseResult, EvalCase, EvalConfig, EvalRunResult, EvalSummary, PromptSpec, utc_timestamp
from agent_eval_lab.validators import validate_output


SystemUnderTest = Callable[[str], str]


def run_evaluation(
    dataset: Iterable[EvalCase],
    config: EvalConfig,
    system_under_test: SystemUnderTest,
    prompt_spec: PromptSpec | None = None,
) -> EvalRunResult:
    case_results: List[CaseResult] = []

    for case in dataset:
        prompt_text = prompt_spec.render(case) if prompt_spec else case.input_text
        actual_answer = system_under_test(prompt_text)
        outcome = validate_output(case, actual_answer)
        case_results.append(
            CaseResult(
                case_id=case.case_id,
                input_text=case.input_text,
                expected_answer=case.expected_answer,
                actual_answer=actual_answer,
                passed=outcome.passed,
                score=outcome.score,
                failure_reason=outcome.failure_reason,
                failure_category=outcome.failure_category,
                tags=list(case.tags),
                prompt_text=prompt_text,
            )
        )

    total_cases = len(case_results)
    passed_cases = sum(1 for item in case_results if item.passed)
    failed_cases = total_cases - passed_cases
    average_score = sum(item.score for item in case_results) / total_cases if total_cases else 0.0
    pass_rate = passed_cases / total_cases if total_cases else 0.0
    failure_categories = dict(
        sorted(
            Counter(item.failure_category for item in case_results if item.failure_category).items()
        )
    )

    return EvalRunResult(
        run_id=str(uuid4()),
        config_name=config.name,
        created_at=utc_timestamp(),
        summary=EvalSummary(
            total_cases=total_cases,
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            pass_rate=pass_rate,
            average_score=average_score,
            failure_categories=failure_categories,
            prompt_id=prompt_spec.prompt_id if prompt_spec else None,
            prompt_version=prompt_spec.version if prompt_spec else None,
        ),
        case_results=case_results,
    )
