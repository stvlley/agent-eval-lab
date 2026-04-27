from agent_eval_lab.models import EvalCase
from agent_eval_lab.validators import validate_output


def test_contains_validator_passes_when_expected_substring_present():
    case = EvalCase(
        case_id="contains-1",
        input_text="greet",
        expected_answer="hello",
        validator_name="contains",
    )

    result = validate_output(case, "say hello to the user")

    assert result.passed is True
    assert result.failure_reason is None
    assert result.failure_category is None


def test_regex_validator_fails_with_format_error_category():
    case = EvalCase(
        case_id="regex-1",
        input_text="date",
        expected_answer=r"\d{4}-\d{2}-\d{2}",
        validator_name="regex",
    )

    result = validate_output(case, "April 27, 2026")

    assert result.passed is False
    assert result.failure_reason == "regex_match_failed"
    assert result.failure_category == "format_error"


def test_json_keys_validator_fails_when_required_keys_missing():
    case = EvalCase(
        case_id="json-1",
        input_text="structured",
        expected_answer="",
        validator_name="json_keys",
        validator_config={"required_keys": ["answer", "confidence"]},
    )

    result = validate_output(case, '{"answer": "ok"}')

    assert result.passed is False
    assert result.failure_reason == "json_missing_keys"
    assert result.failure_category == "format_error"
