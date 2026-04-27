import json
from pathlib import Path

from agent_eval_lab.models import EvalCase, EvalConfig, PromptSpec
from agent_eval_lab.runner import run_evaluation
from agent_eval_lab.report_io import write_run_report


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
