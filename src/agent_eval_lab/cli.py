from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from agent_eval_lab.dataset_io import load_eval_cases
from agent_eval_lab.models import EvalConfig
from agent_eval_lab.providers import build_system_under_test
from agent_eval_lab.report_io import write_run_report
from agent_eval_lab.runner import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-eval-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a baseline evaluation")
    run_parser.add_argument("--dataset", required=True, help="Path to JSON dataset")
    run_parser.add_argument("--output-dir", default="runs", help="Directory for JSON reports")
    run_parser.add_argument("--config-name", default="baseline", help="Logical name for this run")
    run_parser.add_argument("--provider", default="echo", help="Provider adapter to use (echo, static)")
    run_parser.add_argument(
        "--provider-response",
        default=None,
        help="Fixed response text for the static provider",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        dataset = load_eval_cases(Path(args.dataset))
        config = EvalConfig(name=args.config_name)
        provider_config = {}
        if args.provider_response is not None:
            provider_config["response_text"] = args.provider_response
        system_under_test = build_system_under_test(args.provider, provider_config)
        result = run_evaluation(dataset=dataset, config=config, system_under_test=system_under_test)
        output_path = write_run_report(result, output_dir=Path(args.output_dir))
        print(
            f"run_id={result.run_id} provider={args.provider} total={result.summary.total_cases} "
            f"passed={result.summary.passed_cases} failed={result.summary.failed_cases} "
            f"pass_rate={result.summary.pass_rate:.2f} "
            f"failure_categories={result.summary.failure_categories} report={output_path}"
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
