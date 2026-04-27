# Comparison Runner Implementation Plan

> **For Hermes:** implement with strict TDD. Write failing tests first, run them, then add the minimum code to pass before refactoring.

**Goal:** Add a real comparison runner that can evaluate multiple explicit scenarios and produce ranked, baseline-aware reports for prompt, provider, and model experiments.

**Architecture:** Use an explicit scenario registry file instead of fragile flag cross-products. Each scenario declares provider/model/prompt selections. The comparison runner will execute each scenario through the existing `run_evaluation()` path, then derive ranking and deltas relative to the first scenario as the baseline.

**Tech Stack:** Python dataclasses, existing CLI, existing report writer pattern, pytest.

---

## Task 1: Add failing tests for scenario loading and validation
- Create scenario registry tests covering:
  - valid scenario loading
  - duplicate labels rejected
  - `prompt_version` without `prompt_id` rejected
  - `static` without `provider_response` rejected

## Task 2: Add failing tests for comparison execution
- Create comparison-runner tests covering:
  - multiple scenario execution
  - baseline selection from first scenario
  - ranking order by pass rate / average score
  - prompt metadata preserved inside nested eval runs
  - deltas computed against baseline

## Task 3: Add failing tests for comparison report persistence
- Create report I/O tests covering:
  - comparison JSON written successfully
  - ranking and baseline metadata persisted

## Task 4: Add failing tests for CLI compare flow
- Create CLI tests covering:
  - `compare` command runs from dataset + scenario file
  - prompt file integration works
  - output summary includes best scenario and report path
  - invalid scenario file fails before provider setup when possible

## Task 5: Implement comparison models and loader
- Add comparison dataclasses to `src/agent_eval_lab/models.py`
- Add `src/agent_eval_lab/comparison.py` for scenario loading, validation, and execution

## Task 6: Implement comparison report writer and CLI integration
- Extend `report_io.py` with comparison report writing
- Extend `cli.py` with `compare` subcommand
- Add example scenario file under `examples/comparisons/`

## Task 7: Verify and harden
- Run focused tests, then full suite
- Run a real CLI compare command against examples
- Inspect output JSON manually
- Review README and update usage docs
- Commit and push only after the repo is clean and green
