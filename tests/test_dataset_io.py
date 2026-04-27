import json
from pathlib import Path

from agent_eval_lab.dataset_io import load_eval_cases


def test_load_eval_cases_reads_json_dataset(tmp_path: Path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "hello-1",
                    "input_text": "hello",
                    "expected_answer": "world",
                    "tags": ["smoke"],
                }
            ]
        )
    )

    cases = load_eval_cases(dataset_path)

    assert len(cases) == 1
    assert cases[0].case_id == "hello-1"
    assert cases[0].expected_answer == "world"
    assert cases[0].tags == ["smoke"]
