import os
from unittest.mock import patch

from agent_eval_lab.providers import build_system_under_test


def test_echo_provider_returns_input_text():
    system = build_system_under_test(provider_name="echo")

    assert system("hello") == "hello"


def test_static_provider_returns_fixed_response():
    system = build_system_under_test(provider_name="static", provider_config={"response_text": "always this"})

    assert system("ignored") == "always this"


def test_missing_static_response_raises_value_error():
    try:
        build_system_under_test(provider_name="static", provider_config={})
    except ValueError as exc:
        assert "response_text" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing response_text")


def test_static_provider_allows_empty_string_response():
    system = build_system_under_test(provider_name="static", provider_config={"response_text": ""})

    assert system("ignored") == ""


def test_openai_provider_requires_api_key():
    with patch.dict(os.environ, {}, clear=True):
        try:
            build_system_under_test(provider_name="openai", provider_config={})
        except ValueError as exc:
            assert "OPENAI_API_KEY" in str(exc)
        else:
            raise AssertionError("Expected ValueError for missing OPENAI_API_KEY")


def test_anthropic_provider_requires_api_key():
    with patch.dict(os.environ, {}, clear=True):
        try:
            build_system_under_test(provider_name="anthropic", provider_config={})
        except ValueError as exc:
            assert "ANTHROPIC_API_KEY" in str(exc)
        else:
            raise AssertionError("Expected ValueError for missing ANTHROPIC_API_KEY")


def test_openai_provider_returns_stubbed_response():
    def fake_call(**kwargs):
        assert kwargs["api_key"] == "openai-key"
        assert kwargs["model"] == "gpt-4.1-mini"
        assert kwargs["prompt"] == "hello"
        return "openai says hi"

    with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True):
        system = build_system_under_test(
            provider_name="openai",
            provider_config={"model": "gpt-4.1-mini", "client": fake_call},
        )

        assert system("hello") == "openai says hi"


def test_anthropic_provider_returns_stubbed_response():
    def fake_call(**kwargs):
        assert kwargs["api_key"] == "anthropic-key"
        assert kwargs["model"] == "claude-3-5-haiku-latest"
        assert kwargs["prompt"] == "hello"
        return "anthropic says hi"

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "anthropic-key"}, clear=True):
        system = build_system_under_test(
            provider_name="anthropic",
            provider_config={"model": "claude-3-5-haiku-latest", "client": fake_call},
        )

        assert system("hello") == "anthropic says hi"
