import json
from pathlib import Path

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
