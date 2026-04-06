import pytest

from llm_econ_beliefs import (
    build_claude_command,
    build_litellm_belief_tool,
    build_openai_chat_payload,
    build_openai_response_payload,
    resolve_litellm_model_name,
)
from llm_econ_beliefs.providers import (
    run_litellm_prompt_logged,
)


def test_build_claude_command_includes_model_and_schema():
    command = build_claude_command("hello", model_name="sonnet")

    assert command[0].endswith("claude")
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
    assert payload["messages"][0]["content"] == (
        "Follow the user's instructions exactly and return only the final answer."
    )


def test_build_openai_chat_payload_rejects_n_above_api_limit():
    with pytest.raises(ValueError, match="n must be <= 8"):
        build_openai_chat_payload(
            "hello",
            model_name="gpt-5.4-mini",
            n=9,
        )


def test_build_openai_response_payload_with_full_tools():
    payload = build_openai_response_payload(
        "hello",
        model_name="gpt-5.4-mini",
        tool_regime="full",
    )

    assert payload["model"] == "gpt-5.4-mini"
    assert payload["tool_choice"] == "auto"
    assert payload["tools"][0]["type"] == "web_search"
    assert payload["tools"][1]["type"] == "code_interpreter"
    assert payload["include"] == ["web_search_call.action.sources"]


def test_resolve_litellm_model_name_supports_policybench_aliases():
    assert resolve_litellm_model_name("claude-haiku-4.5") == "claude-haiku-4-5-20251001"
    assert resolve_litellm_model_name("gemini-3-flash-preview") == "gemini/gemini-3-flash-preview"
    assert resolve_litellm_model_name("custom-model") == "custom-model"


def test_build_litellm_belief_tool_uses_schema():
    tool = build_litellm_belief_tool()

    assert tool["type"] == "function"
    assert tool["function"]["name"] == "submit_belief"
    assert tool["function"]["parameters"]["required"] == [
        "interpretation",
        "point_estimate",
        "quantiles",
        "citations",
        "reasoning_summary",
    ]


class _FakeLiteLLM:
    def __init__(self, response):
        self._response = response

    def completion(self, **kwargs):
        self.kwargs = kwargs
        return self._response

    @staticmethod
    def completion_cost(*, completion_response):
        return 0.123


class _FakeLiteLLMCostFromTicks(_FakeLiteLLM):
    @staticmethod
    def completion_cost(*, completion_response):
        raise RuntimeError("no direct cost")


class _FakeFunction:
    def __init__(self, arguments):
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, arguments):
        self.function = _FakeFunction(arguments)


class _FakeMessage:
    def __init__(self, *, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeUsage:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return dict(self._payload)


class _FakeResponse:
    def __init__(self, *, message, usage, request_id="resp_1"):
        self.id = request_id
        self.choices = [_FakeChoice(message)]
        self.usage = _FakeUsage(usage)


def test_run_litellm_prompt_logged_reads_json_object(monkeypatch):
    response = _FakeResponse(
        message=_FakeMessage(content='{"point_estimate": 0.5}'),
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )
    fake_litellm = _FakeLiteLLM(response)
    monkeypatch.setattr(
        "llm_econ_beliefs.providers._import_litellm",
        lambda: fake_litellm,
    )

    result = run_litellm_prompt_logged("hello", model_name="gemini-3-flash-preview")

    assert result.outputs == ['{"point_estimate": 0.5}']
    assert result.request_id == "resp_1"
    assert result.usage["litellm_cost_usd"] == 0.123
    assert fake_litellm.kwargs["response_format"] == {"type": "json_object"}


def test_run_litellm_prompt_logged_reads_function_call(monkeypatch):
    response = _FakeResponse(
        message=_FakeMessage(
            tool_calls=[_FakeToolCall('{"point_estimate": 0.5, "quantiles": {}}')]
        ),
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )
    fake_litellm = _FakeLiteLLM(response)
    monkeypatch.setattr(
        "llm_econ_beliefs.providers._import_litellm",
        lambda: fake_litellm,
    )

    result = run_litellm_prompt_logged("hello", model_name="claude-haiku-4.5")

    assert result.outputs == ['{"point_estimate": 0.5, "quantiles": {}}']
    assert result.request_id == "resp_1"
    assert fake_litellm.kwargs["tool_choice"]["function"]["name"] == "submit_belief"


def test_run_litellm_prompt_logged_falls_back_to_cost_ticks(monkeypatch):
    response = _FakeResponse(
        message=_FakeMessage(content='{"point_estimate": 0.5}'),
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "cost_in_usd_ticks": 2164000,
        },
    )
    fake_litellm = _FakeLiteLLMCostFromTicks(response)
    monkeypatch.setattr(
        "llm_econ_beliefs.providers._import_litellm",
        lambda: fake_litellm,
    )

    result = run_litellm_prompt_logged("hello", model_name="gemini-3-flash-preview")

    assert result.usage["litellm_cost_usd"] == pytest.approx(0.0002164)
