from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


@dataclass
class ComparisonScenario:
    label: str
    provider: str
    provider_model: str | None = None
    provider_response: str | None = None
    prompt_id: str | None = None
    prompt_version: str | None = None


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


@dataclass(init=False)
class ComparisonScenarioResult:
    label: str
    provider: str
    provider_model: str | None
    prompt_id: str | None = None
    prompt_version: str | None = None
    result: EvalRunResult | None = None
    provider_response: str | None = None

    def __init__(
        self,
        label: str,
        provider: str,
        provider_model: str | None,
        prompt_id: str | None = None,
        prompt_version: str | None = None,
        result: EvalRunResult | None = None,
        provider_response: str | None = None,
    ) -> None:
        self.label = label
        self.provider = provider
        self.provider_model = provider_model
        self.prompt_id = prompt_id
        self.prompt_version = prompt_version
        self.result = result
        self.provider_response = provider_response


@dataclass(init=False)
class ComparisonRanking:
    label: str
    provider: str
    provider_model: str | None
    prompt_id: str | None = None
    prompt_version: str | None = None
    pass_rate: float = 0.0
    average_score: float = 0.0
    delta_vs_baseline: float = 0.0
    provider_response: str | None = None

    def __init__(
        self,
        label: str,
        provider: str,
        provider_model: str | None,
        prompt_id: str | None = None,
        prompt_version: str | None = None,
        pass_rate: float = 0.0,
        average_score: float = 0.0,
        delta_vs_baseline: float = 0.0,
        provider_response: str | None = None,
    ) -> None:
        self.label = label
        self.provider = provider
        self.provider_model = provider_model
        self.prompt_id = prompt_id
        self.prompt_version = prompt_version
        self.pass_rate = pass_rate
        self.average_score = average_score
        self.delta_vs_baseline = delta_vs_baseline
        self.provider_response = provider_response


@dataclass
class ComparisonSummary:
    scenario_count: int
    baseline_label: str
    best_label: str
    best_pass_rate: float


@dataclass
class ComparisonResult:
    comparison_id: str
    config_name: str
    created_at: str
    summary: ComparisonSummary
    rankings: List[ComparisonRanking]
    scenario_results: List[ComparisonScenarioResult]

    def to_dict(self) -> dict:
        return asdict(self)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
