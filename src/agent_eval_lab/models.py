from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


@dataclass
class EvalCase:
    case_id: str
    input_text: str
    expected_answer: str
    tags: List[str] = field(default_factory=list)
    validator_name: str = "exact_match"
    validator_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptSpec:
    prompt_id: str
    version: str
    template: str
    description: str

    def render(self, case: EvalCase) -> str:
        return self.template.format(
            input_text=case.input_text,
            case_id=case.case_id,
        )


@dataclass
class EvalConfig:
    name: str
    validator: str = "exact_match"
    rubric: str = "baseline"


@dataclass
class ValidationOutcome:
    passed: bool
    score: float
    failure_reason: str | None
    failure_category: str | None


@dataclass
class CaseResult:
    case_id: str
    input_text: str
    expected_answer: str
    actual_answer: str
    passed: bool
    score: float
    failure_reason: str | None
    failure_category: str | None
    tags: List[str] = field(default_factory=list)
    prompt_text: str | None = None


@dataclass
class EvalSummary:
    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    average_score: float
    failure_categories: Dict[str, int] = field(default_factory=dict)
    prompt_id: str | None = None
    prompt_version: str | None = None


@dataclass
class EvalRunResult:
    run_id: str
    config_name: str
    created_at: str
    summary: EvalSummary
    case_results: List[CaseResult]

    def to_dict(self) -> dict:
        return asdict(self)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
