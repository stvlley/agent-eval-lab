from __future__ import annotations

from typing import Any, Callable, Dict


SystemUnderTest = Callable[[str], str]


def build_system_under_test(
    provider_name: str,
    provider_config: Dict[str, Any] | None = None,
) -> SystemUnderTest:
    config = provider_config or {}

    if provider_name == "echo":
        return lambda prompt: prompt

    if provider_name == "static":
        response_text = config.get("response_text")
        if not response_text:
            raise ValueError("static provider requires provider_config['response_text']")
        return lambda prompt: response_text

    raise ValueError(f"Unsupported provider: {provider_name}")
