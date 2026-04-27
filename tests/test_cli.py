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
