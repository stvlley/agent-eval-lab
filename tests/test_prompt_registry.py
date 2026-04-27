import json
from pathlib import Path

from agent_eval_lab.models import EvalCase, PromptSpec
from agent_eval_lab.prompt_registry import load_prompt_specs, select_prompt_spec


def test_load_prompt_specs_reads_json_prompt_file(tmp_path: Path):
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(
        json.dumps(
            [
                {
                    "prompt_id": "baseline",
                    "version": "v1",
                    "template": "Answer briefly: {input_text}",
                    "description": "Default prompt",
                }
            ]
        )
    )

    specs = load_prompt_specs(prompt_path)

    assert len(specs) == 1
    assert specs[0].prompt_id == "baseline"
    assert specs[0].version == "v1"


def test_select_prompt_spec_prefers_requested_id():
    specs = [
        PromptSpec(prompt_id="baseline", version="v1", template="Base {input_text}", description="base"),
        PromptSpec(prompt_id="reasoning", version="v2", template="Reason {input_text}", description="reasoning"),
    ]

    selected = select_prompt_spec(specs, prompt_id="reasoning")

    assert selected.prompt_id == "reasoning"
    assert selected.version == "v2"


def test_select_prompt_spec_supports_prompt_version():
    specs = [
        PromptSpec(prompt_id="baseline", version="v1", template="Base one {input_text}", description="base1"),
        PromptSpec(prompt_id="baseline", version="v2", template="Base two {input_text}", description="base2"),
    ]

    selected = select_prompt_spec(specs, prompt_id="baseline", version="v2")

    assert selected.prompt_id == "baseline"
    assert selected.version == "v2"


def test_select_prompt_spec_requires_version_when_prompt_id_is_ambiguous():
    specs = [
        PromptSpec(prompt_id="baseline", version="v1", template="Base one {input_text}", description="base1"),
        PromptSpec(prompt_id="baseline", version="v2", template="Base two {input_text}", description="base2"),
    ]

    try:
        select_prompt_spec(specs, prompt_id="baseline")
    except ValueError as exc:
        assert "version" in str(exc)
    else:
        raise AssertionError("Expected ValueError for ambiguous prompt_id without version")


def test_select_prompt_spec_returns_single_prompt_family_without_prompt_id():
    specs = [
        PromptSpec(prompt_id="baseline", version="v1", template="Base one {input_text}", description="base1"),
        PromptSpec(prompt_id="baseline", version="v2", template="Base two {input_text}", description="base2"),
    ]

    selected = select_prompt_spec(specs)

    assert selected.prompt_id == "baseline"
    assert selected.version == "v1"


def test_load_prompt_specs_rejects_duplicate_prompt_id_and_version(tmp_path: Path):
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(
        json.dumps(
            [
                {
                    "prompt_id": "baseline",
                    "version": "v1",
                    "template": "One {input_text}",
                    "description": "first",
                },
                {
                    "prompt_id": "baseline",
                    "version": "v1",
                    "template": "Two {input_text}",
                    "description": "duplicate",
                },
            ]
        )
    )

    try:
        load_prompt_specs(prompt_path)
    except ValueError as exc:
        assert "Duplicate" in str(exc)
    else:
        raise AssertionError("Expected ValueError for duplicate prompt id/version")


def test_load_prompt_specs_rejects_disallowed_placeholder(tmp_path: Path):
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(
        json.dumps(
            [
                {
                    "prompt_id": "leaky",
                    "version": "v1",
                    "template": "Answer is {expected_answer}",
                    "description": "leaks labels",
                }
            ]
        )
    )

    try:
        load_prompt_specs(prompt_path)
    except ValueError as exc:
        assert "expected_answer" in str(exc)
    else:
        raise AssertionError("Expected ValueError for disallowed placeholder")


def test_load_prompt_specs_rejects_nested_disallowed_placeholder(tmp_path: Path):
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(
        json.dumps(
            [
                {
                    "prompt_id": "nested-leak",
                    "version": "v1",
                    "template": "{input_text:{expected_answer}}",
                    "description": "nested label leak",
                }
            ]
        )
    )

    try:
        load_prompt_specs(prompt_path)
    except ValueError as exc:
        assert "expected_answer" in str(exc)
    else:
        raise AssertionError("Expected ValueError for nested disallowed placeholder")


def test_load_prompt_specs_rejects_invalid_format_spec_for_input_text(tmp_path: Path):
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(
        json.dumps(
            [
                {
                    "prompt_id": "bad-format",
                    "version": "v1",
                    "template": "{input_text:.2f}",
                    "description": "invalid string format",
                }
            ]
        )
    )

    try:
        load_prompt_specs(prompt_path)
    except ValueError as exc:
        assert "Invalid format spec" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid format spec")


def test_select_prompt_spec_rejects_empty_registry():
    try:
        select_prompt_spec([], prompt_id=None)
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for empty prompt registry")


def test_prompt_spec_renders_eval_case_input():
    case = EvalCase(case_id="1", input_text="What is 2+2?", expected_answer="4")
    spec = PromptSpec(
        prompt_id="baseline",
        version="v1",
        template="Answer this exactly: {input_text}",
        description="default",
    )

    assert spec.render(case) == "Answer this exactly: What is 2+2?"
