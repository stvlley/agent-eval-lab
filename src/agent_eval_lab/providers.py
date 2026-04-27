from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Callable, Dict


SystemUnderTest = Callable[[str], str]
ProviderClient = Callable[..., str]


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

    if provider_name == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("openai provider requires OPENAI_API_KEY")
        model = config.get("model", "gpt-4.1-mini")
        client = config.get("client", call_openai_responses_api)
        return lambda prompt: client(api_key=api_key, model=model, prompt=prompt)

    if provider_name == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("anthropic provider requires ANTHROPIC_API_KEY")
        model = config.get("model", "claude-3-5-haiku-latest")
        client = config.get("client", call_anthropic_messages_api)
        return lambda prompt: client(api_key=api_key, model=model, prompt=prompt)

    raise ValueError(f"Unsupported provider: {provider_name}")


def resolve_provider_client(provider_name: str) -> ProviderClient | None:
    if provider_name == "openai":
        return call_openai_responses_api
    if provider_name == "anthropic":
        return call_anthropic_messages_api
    return None


def call_openai_responses_api(*, api_key: str, model: str, prompt: str) -> str:
    body = json.dumps(
        {
            "model": model,
            "input": prompt,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["output"][0]["content"][0]["text"]


def call_anthropic_messages_api(*, api_key: str, model: str, prompt: str) -> str:
    body = json.dumps(
        {
            "model": model,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["content"][0]["text"]
