from __future__ import annotations

import json
import re
from typing import Iterable

from agent_eval_lab.models import EvalCase, ValidationOutcome


def exact_match(expected: str, actual: str) -> bool:
    return expected.strip() == actual.strip()


def contains_match(expected: str, actual: str) -> bool:
    return expected.strip().lower() in actual.strip().lower()


def regex_match(pattern: str, actual: str) -> bool:
    return re.search(pattern, actual) is not None


def json_has_required_keys(actual: str, required_keys: Iterable[str]) -> bool:
    payload = json.loads(actual)
    return all(key in payload for key in required_keys)


def validate_output(case: EvalCase, actual: str) -> ValidationOutcome:
    validator_name = case.validator_name or "exact_match"

    if validator_name == "exact_match":
        passed = exact_match(case.expected_answer, actual)
        return ValidationOutcome(
            passed=passed,
            score=1.0 if passed else 0.0,
            failure_reason=None if passed else "exact_match_failed",
            failure_category=None if passed else "wrong_answer",
        )

    if validator_name == "contains":
        passed = contains_match(case.expected_answer, actual)
        return ValidationOutcome(
            passed=passed,
            score=1.0 if passed else 0.0,
            failure_reason=None if passed else "contains_match_failed",
            failure_category=None if passed else "wrong_answer",
        )

    if validator_name == "regex":
        passed = regex_match(case.expected_answer, actual)
        return ValidationOutcome(
            passed=passed,
            score=1.0 if passed else 0.0,
            failure_reason=None if passed else "regex_match_failed",
            failure_category=None if passed else "format_error",
        )

    if validator_name == "json_keys":
        required_keys = case.validator_config.get("required_keys", [])
        try:
            passed = json_has_required_keys(actual, required_keys)
        except json.JSONDecodeError:
            return ValidationOutcome(
                passed=False,
                score=0.0,
                failure_reason="json_parse_failed",
                failure_category="format_error",
            )

        return ValidationOutcome(
            passed=passed,
            score=1.0 if passed else 0.0,
            failure_reason=None if passed else "json_missing_keys",
            failure_category=None if passed else "format_error",
        )

    raise ValueError(f"Unsupported validator: {validator_name}")
