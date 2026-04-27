from __future__ import annotations

import json
from pathlib import Path
from typing import List

from agent_eval_lab.models import EvalCase


def load_eval_cases(path: Path) -> List[EvalCase]:
    payload = json.loads(path.read_text())
    return [EvalCase(**item) for item in payload]
