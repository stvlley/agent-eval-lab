from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Iterable, List

from agent_eval_lab.models import EvalCase, PromptSpec

ALLOWED_PLACEHOLDERS = {"input_text", "case_id"}


def load_prompt_specs(path: Path) -> List[PromptSpec]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError("Prompt registry must be a non-empty JSON list")

    specs = [PromptSpec(**item) for item in payload]
    validate_prompt_specs(specs)
    return specs


def validate_prompt_specs(specs: Iterable[PromptSpec]) -> None:
    seen: set[tuple[str, str]] = set()

    for spec in specs:
        key = (spec.prompt_id, spec.version)
        if key in seen:
            raise ValueError(f"Duplicate prompt spec for prompt_id={spec.prompt_id} version={spec.version}")
        seen.add(key)

        field_names = extract_template_fields(spec.template)
        disallowed = sorted(field_names - ALLOWED_PLACEHOLDERS)
        if disallowed:
            raise ValueError(
                "Prompt template contains disallowed placeholders: " + ", ".join(disallowed)
            )

        validate_prompt_template_renderability(spec)


def validate_prompt_template_renderability(spec: PromptSpec) -> None:
    sample_case = EvalCase(case_id="sample-case", input_text="sample input", expected_answer="unused")
    try:
        spec.render(sample_case)
    except Exception as exc:  # format errors vary by template shape
        raise ValueError(
            f"Invalid format spec or template rendering failure for prompt_id={spec.prompt_id} "
            f"version={spec.version}: {exc}"
        ) from exc


def extract_template_fields(template: str) -> set[str]:
    formatter = string.Formatter()
    field_names: set[str] = set()

    for _, field_name, format_spec, _ in formatter.parse(template):
        if field_name is not None:
            field_names.add(field_name)
        if format_spec and "{" in format_spec:
            field_names.update(extract_template_fields(format_spec))

    return field_names


def select_prompt_spec(
    specs: Iterable[PromptSpec],
    prompt_id: str | None = None,
    version: str | None = None,
) -> PromptSpec:
    specs = list(specs)
    if not specs:
        raise ValueError("Prompt registry is empty")

    if prompt_id is None:
        if version is not None:
            raise ValueError("prompt_id is required when selecting by version")
        unique_prompt_ids = {spec.prompt_id for spec in specs}
        if len(unique_prompt_ids) > 1:
            raise ValueError("prompt_id is required when prompt registry contains multiple prompt IDs")
        return specs[0]

    matches = [spec for spec in specs if spec.prompt_id == prompt_id]
    if not matches:
        raise ValueError(f"Prompt ID not found: {prompt_id}")

    if version is None:
        if len(matches) > 1:
            raise ValueError(f"Prompt ID '{prompt_id}' is ambiguous; specify a version")
        return matches[0]

    version_matches = [spec for spec in matches if spec.version == version]
    if not version_matches:
        raise ValueError(f"Prompt version not found for prompt_id={prompt_id}: {version}")
    return version_matches[0]
