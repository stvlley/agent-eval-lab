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
