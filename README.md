# agent-eval-lab

Local-first starter repo for building agentic harnesses and robust eval pipelines.

## What this starter includes
- typed task/result models
- provider adapter layer
- prompt/version registry
- multiple deterministic validators
- failure taxonomy in run summaries
- traceable JSON run output
- CLI for running eval suites
- tests for core evaluation behavior

## Supported providers
- `echo` — returns the input text
- `static` — returns a fixed configured response
- `openai` — calls the OpenAI Responses API, gated by `OPENAI_API_KEY`
- `anthropic` — calls the Anthropic Messages API, gated by `ANTHROPIC_API_KEY`

The real providers are env-var gated so the repo still works locally without paid API access.

## Prompt registry
Prompt specs are stored in a JSON file and let you version prompt templates independently from datasets.

Each prompt spec includes:
- `prompt_id`
- `version`
- `template`
- `description`

Templates can reference:
- `{input_text}`
- `{case_id}`

`{expected_answer}` is intentionally disallowed so prompt experiments do not leak labels into model inputs or stored run artifacts.

Example prompt file:
```json
[
  {
    "prompt_id": "baseline",
    "version": "v1",
    "template": "{input_text}",
    "description": "Direct baseline"
  },
  {
    "prompt_id": "prefix",
    "version": "v2",
    "template": "Prompted v2: {input_text}",
    "description": "Prefixed variant"
  }
]
```

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
PYTHONPATH=src python3 -m agent_eval_lab.cli run --dataset examples/datasets/basic_tasks.json --provider static --provider-response "hello"
PYTHONPATH=src python3 -m agent_eval_lab.cli run --dataset examples/datasets/basic_tasks.json --prompt-file examples/prompts/basic_prompts.json --prompt-id baseline
PYTHONPATH=src python3 -m agent_eval_lab.cli run --dataset examples/datasets/basic_tasks.json --prompt-file examples/prompts/basic_prompts.json --prompt-id prefix --prompt-version v2
PYTHONPATH=src python3 -m agent_eval_lab.cli compare --dataset examples/datasets/basic_tasks.json --scenario-file examples/comparisons/basic_compare.json --prompt-file examples/prompts/basic_prompts.json
```

## Comparison runner
Comparison runs are driven by an explicit JSON scenario file so you can compare prompts, providers, and models without hidden flag cross-products.

Each scenario supports:
- `label`
- `provider`
- `provider_model` (optional)
- `provider_response` (required for `static`)
- `prompt_id` (optional)
- `prompt_version` (optional, requires `prompt_id`)

Example scenario file:
```json
[
  {
    "label": "echo-baseline",
    "provider": "echo"
  },
  {
    "label": "prefix-prompt",
    "provider": "echo",
    "prompt_id": "prefix",
    "prompt_version": "v1"
  },
  {
    "label": "bad-static",
    "provider": "static",
    "provider_response": "wrong"
  }
]
```

The first scenario is treated as the baseline. Comparison reports include:
- per-scenario nested eval results
- ranked leaderboard by pass rate, then average score
- `delta_vs_baseline` for each scenario
- best scenario metadata in the top-level summary

Run it:
```bash
PYTHONPATH=src python3 -m agent_eval_lab.cli compare \
  --dataset examples/datasets/basic_tasks.json \
  --scenario-file examples/comparisons/basic_compare.json \
  --prompt-file examples/prompts/basic_prompts.json
```

Use this pattern for:
- prompt A vs prompt B
- model A vs model B
- provider A vs provider B
- baseline vs candidate regression checks
## Real provider usage
```bash
export OPENAI_API_KEY=...
PYTHONPATH=src python3 -m agent_eval_lab.cli run \
  --dataset examples/datasets/basic_tasks.json \
  --provider openai \
  --provider-model gpt-4.1-mini

export ANTHROPIC_API_KEY=...
PYTHONPATH=src python3 -m agent_eval_lab.cli run \
  --dataset examples/datasets/basic_tasks.json \
  --provider anthropic \
  --provider-model claude-3-5-haiku-latest
```

## Optional install extras
```bash
pip install -e .[providers]
```

## Repo shape
- `src/agent_eval_lab/` core package
- `tests/` unit tests
- `examples/` sample dataset and prompt config
- `runs/` output directory for eval runs

## Next upgrades
- LiteLLM adapter
- agent traces
- LLM judge integration
- SQLite/Postgres run store
- HTML report generation
- CI regression gates
