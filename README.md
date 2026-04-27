# agent-eval-lab

Local-first starter repo for building agentic harnesses and robust eval pipelines.

## What this starter includes
- typed task/result models
- multiple deterministic validators
- failure taxonomy in run summaries
- traceable JSON run output
- CLI for running a baseline eval suite
- tests for core evaluation behavior

## Supported validators
- `exact_match`
- `contains`
- `regex`
- `json_keys`

Each eval case can choose its own validator and optional validator config.

## Quick start
```bash
cd agent-eval-lab
python3 -m pytest -q
PYTHONPATH=src python3 -m agent_eval_lab.cli run --dataset examples/datasets/basic_tasks.json
```

## Repo shape
- `src/agent_eval_lab/` core package
- `tests/` unit tests
- `examples/` sample dataset and prompt config
- `runs/` output directory for eval runs

## Next upgrades
- provider adapters
- agent traces
- LLM judge integration
- SQLite/Postgres run store
- HTML report generation
- CI regression gates
