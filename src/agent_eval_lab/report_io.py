from __future__ import annotations

import json
from pathlib import Path

from agent_eval_lab.models import ComparisonResult, EvalRunResult


def write_run_report(result: EvalRunResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{result.run_id}.json"
    output_path.write_text(json.dumps(result.to_dict(), indent=2))
    return output_path


def write_comparison_report(result: ComparisonResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"comparison-{result.comparison_id}.json"
    output_path.write_text(json.dumps(result.to_dict(), indent=2))
    return output_path
