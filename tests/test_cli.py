import json
from pathlib import Path
from unittest.mock import patch

from agent_eval_lab.cli import main


def test_cli_run_writes_report(tmp_path: Path, capsys):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "echo-1",
                    "input_text": "hello",
                    "expected_answer": "hello",
                    "tags": ["smoke"],
                }
            ]
        )
    )

    exit_code = main([
        "run",
        "--dataset",
        str(dataset_path),
        "--output-dir",
        str(tmp_path / "runs"),
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "pass_rate=1.00" in captured.out
    assert "failure_categories={}" in captured.out
    assert len(list((tmp_path / "runs").glob("*.json"))) == 1


def test_cli_run_accepts_static_provider(tmp_path: Path, capsys):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "static-1",
                    "input_text": "question",
                    "expected_answer": "fixed answer",
                    "validator_name": "exact_match"
                }
            ]
        )
    )

    exit_code = main([
        "run",
        "--dataset",
        str(dataset_path),
        "--output-dir",
        str(tmp_path / "runs"),
        "--provider",
        "static",
        "--provider-response",
        "fixed answer",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "pass_rate=1.00" in captured.out


def test_cli_run_accepts_openai_provider_with_stubbed_client(tmp_path: Path, capsys):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "openai-1",
                    "input_text": "question",
                    "expected_answer": "stub answer",
                    "validator_name": "exact_match"
                }
            ]
        )
    )

    def fake_openai_call(*, api_key: str, model: str, prompt: str) -> str:
        assert api_key == "test-key"
        assert model == "gpt-4.1-mini"
        assert prompt == "question"
        return "stub answer"

    with patch("agent_eval_lab.cli.resolve_provider_client", return_value=fake_openai_call):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
            exit_code = main([
                "run",
                "--dataset",
                str(dataset_path),
                "--output-dir",
                str(tmp_path / "runs"),
                "--provider",
                "openai",
                "--provider-model",
                "gpt-4.1-mini",
            ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "provider=openai" in captured.out
    assert "pass_rate=1.00" in captured.out


def test_cli_run_uses_prompt_file_and_prompt_id(tmp_path: Path, capsys):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "prompt-1",
                    "input_text": "hello",
                    "expected_answer": "Prompted: hello"
                }
            ]
        )
    )
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(
        json.dumps(
            [
                {
                    "prompt_id": "baseline",
                    "version": "v1",
                    "template": "Prompted: {input_text}",
                    "description": "baseline prompt"
                }
            ]
        )
    )

    exit_code = main([
        "run",
        "--dataset",
        str(dataset_path),
        "--output-dir",
        str(tmp_path / "runs"),
        "--prompt-file",
        str(prompt_path),
        "--prompt-id",
        "baseline",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "prompt_id=baseline" in captured.out
    assert "prompt_version=v1" in captured.out


def test_cli_run_rejects_prompt_id_without_prompt_file(tmp_path: Path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "prompt-1",
                    "input_text": "hello",
                    "expected_answer": "hello"
                }
            ]
        )
    )

    try:
        main([
            "run",
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(tmp_path / "runs"),
            "--prompt-id",
            "baseline",
        ])
    except ValueError as exc:
        assert "prompt-file" in str(exc)
    else:
        raise AssertionError("Expected ValueError when prompt-id is provided without prompt-file")


def test_cli_run_supports_prompt_version_selection(tmp_path: Path, capsys):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "prompt-1",
                    "input_text": "hello",
                    "expected_answer": "Prompt v2: hello"
                }
            ]
        )
    )
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(
        json.dumps(
            [
                {
                    "prompt_id": "baseline",
                    "version": "v1",
                    "template": "Prompt v1: {input_text}",
                    "description": "baseline v1"
                },
                {
                    "prompt_id": "baseline",
                    "version": "v2",
                    "template": "Prompt v2: {input_text}",
                    "description": "baseline v2"
                }
            ]
        )
    )

    exit_code = main([
        "run",
        "--dataset",
        str(dataset_path),
        "--output-dir",
        str(tmp_path / "runs"),
        "--prompt-file",
        str(prompt_path),
        "--prompt-id",
        "baseline",
        "--prompt-version",
        "v2",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "prompt_id=baseline" in captured.out
    assert "prompt_version=v2" in captured.out


def test_cli_run_rejects_multi_prompt_registry_without_prompt_id(tmp_path: Path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "prompt-1",
                    "input_text": "hello",
                    "expected_answer": "hello"
                }
            ]
        )
    )
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(
        json.dumps(
            [
                {
                    "prompt_id": "baseline",
                    "version": "v1",
                    "template": "{input_text}",
                    "description": "baseline"
                },
                {
                    "prompt_id": "prefix",
                    "version": "v1",
                    "template": "Prompted: {input_text}",
                    "description": "prefix"
                }
            ]
        )
    )

    try:
        main([
            "run",
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(tmp_path / "runs"),
            "--prompt-file",
            str(prompt_path),
        ])
    except ValueError as exc:
        assert "prompt_id" in str(exc)
    else:
        raise AssertionError("Expected ValueError for multi-prompt registry without prompt-id")


def test_cli_validates_prompt_arguments_before_provider_env(tmp_path: Path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "prompt-1",
                    "input_text": "hello",
                    "expected_answer": "hello"
                }
            ]
        )
    )

    try:
        with patch.dict("os.environ", {}, clear=True):
            main([
                "run",
                "--dataset",
                str(dataset_path),
                "--output-dir",
                str(tmp_path / "runs"),
                "--provider",
                "openai",
                "--prompt-id",
                "baseline",
            ])
    except ValueError as exc:
        assert "prompt-file" in str(exc)
    else:
        raise AssertionError("Expected prompt validation to run before provider env checks")
