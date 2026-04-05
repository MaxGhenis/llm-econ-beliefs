import pytest

from llm_econ_beliefs import build_claude_command, build_openai_chat_payload


def test_build_claude_command_includes_model_and_schema():
    command = build_claude_command("hello", model_name="sonnet")

    assert command[0] == "claude"
    assert "--model" in command
    assert "sonnet" in command
    assert "--json-schema" in command
    assert command[-1] == "hello"


def test_build_openai_chat_payload_includes_n_and_schema():
    payload = build_openai_chat_payload(
        "hello",
        model_name="gpt-5.4-mini",
        n=4,
        temperature=0.8,
    )

    assert payload["model"] == "gpt-5.4-mini"
    assert payload["n"] == 4
    assert payload["temperature"] == 0.8
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["messages"][-1]["content"] == "hello"


def test_build_openai_chat_payload_rejects_n_above_api_limit():
    with pytest.raises(ValueError, match="n must be <= 8"):
        build_openai_chat_payload(
            "hello",
            model_name="gpt-5.4-mini",
            n=9,
        )
