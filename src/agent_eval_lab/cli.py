from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from agent_eval_lab.dataset_io import load_eval_cases
from agent_eval_lab.models import EvalConfig
from agent_eval_lab.prompt_registry import load_prompt_specs, select_prompt_spec
from agent_eval_lab.providers import build_system_under_test, resolve_provider_client
from agent_eval_lab.report_io import write_run_report
from agent_eval_lab.runner import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-eval-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a baseline evaluation")
    run_parser.add_argument("--dataset", required=True, help="Path to JSON dataset")
    run_parser.add_argument("--output-dir", default="runs", help="Directory for JSON reports")
    run_parser.add_argument("--config-name", default="baseline", help="Logical name for this run")
    run_parser.add_argument(
        "--provider",
        default="echo",
        help="Provider adapter to use (echo, static, openai, anthropic)",
    )
    run_parser.add_argument(
        "--provider-response",
        default=None,
        help="Fixed response text for the static provider",
    )
    run_parser.add_argument(
        "--provider-model",
        default=None,
        help="Model name for real providers like openai or anthropic",
    )
    run_parser.add_argument(
        "--prompt-file",
        default=None,
        help="Path to JSON prompt registry file",
    )
    run_parser.add_argument(
        "--prompt-id",
        default=None,
        help="Prompt ID to select from the prompt registry file",
    )
    run_parser.add_argument(
        "--prompt-version",
        default=None,
        help="Prompt version to select from the prompt registry file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        dataset = load_eval_cases(Path(args.dataset))
        config = EvalConfig(name=args.config_name)

        if args.prompt_id is not None and args.prompt_file is None:
            raise ValueError("--prompt-id requires --prompt-file")
        if args.prompt_version is not None and args.prompt_file is None:
            raise ValueError("--prompt-version requires --prompt-file")

        prompt_spec = None
        if args.prompt_file is not None:
            prompt_specs = load_prompt_specs(Path(args.prompt_file))
            prompt_spec = select_prompt_spec(
                prompt_specs,
                prompt_id=args.prompt_id,
                version=args.prompt_version,
            )

        provider_config = {}
        if args.provider_response is not None:
            provider_config["response_text"] = args.provider_response
        if args.provider_model is not None:
            provider_config["model"] = args.provider_model
        resolved_client = resolve_provider_client(args.provider)
        if resolved_client is not None:
            provider_config["client"] = resolved_client
        system_under_test = build_system_under_test(args.provider, provider_config)

        result = run_evaluation(
            dataset=dataset,
            config=config,
            system_under_test=system_under_test,
            prompt_spec=prompt_spec,
        )
        output_path = write_run_report(result, output_dir=Path(args.output_dir))
        print(
            f"run_id={result.run_id} provider={args.provider} total={result.summary.total_cases} "
            f"passed={result.summary.passed_cases} failed={result.summary.failed_cases} "
            f"pass_rate={result.summary.pass_rate:.2f} "
            f"prompt_id={result.summary.prompt_id} prompt_version={result.summary.prompt_version} "
            f"failure_categories={result.summary.failure_categories} report={output_path}"
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
