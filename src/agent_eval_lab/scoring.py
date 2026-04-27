from agent_eval_lab.models import ValidationOutcome


def score_validation(outcome: ValidationOutcome) -> float:
    return outcome.score
