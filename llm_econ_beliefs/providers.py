"""Provider adapters for running prompts against locally available CLIs."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import shutil
from urllib import error, request
from typing import Any

from .models import ProviderBatchResult


OPENAI_CHAT_COMPLETIONS_MAX_N = 8

POLICYBENCH_LITELLM_MODEL_ALIASES: dict[str, str] = {
    "claude-opus-4.7": "claude-opus-4-7",
    "claude-sonnet-4.6": "claude-sonnet-4-6",
    "claude-haiku-4.5": "claude-haiku-4-5-20251001",
    "grok-4.20": "xai/grok-4.20-reasoning",
    "grok-4.1-fast": "xai/grok-4-1-fast-non-reasoning",
    "gemini-3.1-pro-preview": "gemini/gemini-3.1-pro-preview",
    "gemini-3-flash-preview": "gemini/gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview": "gemini/gemini-3.1-flash-lite-preview",
}


DEFAULT_BELIEF_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "interpretation": {"type": "string"},
        "point_estimate": {"type": "number"},
        "quantiles": {
            "type": "object",
            "properties": {
                "p05": {"type": "number"},
                "p25": {"type": "number"},
                "p50": {"type": "number"},
                "p75": {"type": "number"},
                "p95": {"type": "number"},
            },
            "required": ["p05", "p25", "p50", "p75", "p95"],
            "additionalProperties": False,
        },
        "citations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reasoning_summary": {"type": "string"},
    },
    "required": [
        "interpretation",
        "point_estimate",
        "quantiles",
        "citations",
        "reasoning_summary",
    ],
    "additionalProperties": False,
}


def resolve_litellm_model_name(model_name: str) -> str:
    """Resolve a PolicyBench-style alias to the underlying LiteLLM model name."""
    return POLICYBENCH_LITELLM_MODEL_ALIASES.get(model_name, model_name)


def build_litellm_belief_tool(
    json_schema: dict[str, Any] | None = DEFAULT_BELIEF_JSON_SCHEMA,
) -> dict[str, Any]:
    """Build a function-tool schema for providers that support forced structured output."""
    if json_schema is None:
        raise ValueError("LiteLLM structured tool output requires a JSON schema")

    return {
        "type": "function",
        "function": {
            "name": "submit_belief",
            "description": "Return the elicited belief as structured JSON.",
            "parameters": json_schema,
        },
    }


def _import_litellm() -> Any:
    try:
        return importlib.import_module("litellm")
    except ImportError as exc:
        raise RuntimeError(
            "litellm is not installed in this Python environment. "
            "Run the LiteLLM-backed experiments from the PolicyBench virtualenv."
        ) from exc


def _litellm_output_mode(model_name: str) -> str:
    resolved = resolve_litellm_model_name(model_name)
    if resolved.startswith("gemini/"):
        return "json_object"
    if resolved.startswith("claude") or resolved.startswith("xai/grok"):
        return "function_call"
    raise ValueError(f"Unsupported LiteLLM model: {model_name}")


def run_litellm_prompt_logged(
    prompt: str,
    *,
    model_name: str,
    json_schema: dict[str, Any] | None = DEFAULT_BELIEF_JSON_SCHEMA,
    temperature: float = 1.0,
    max_completion_tokens: int = 1200,
    timeout_seconds: int = 180,
) -> ProviderBatchResult:
    """Run one prompt through LiteLLM and return one structured output."""
    litellm = _import_litellm()
    resolved_model_name = resolve_litellm_model_name(model_name)
    output_mode = _litellm_output_mode(model_name)

    request_kwargs: dict[str, Any] = {
        "model": resolved_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_completion_tokens,
        "temperature": temperature,
        "timeout": timeout_seconds,
    }
    if output_mode == "json_object":
        request_kwargs["response_format"] = {"type": "json_object"}
    elif output_mode == "function_call":
        request_kwargs["tools"] = [build_litellm_belief_tool(json_schema)]
        request_kwargs["tool_choice"] = {
            "type": "function",
            "function": {"name": "submit_belief"},
        }
    else:
        raise ValueError(f"Unsupported LiteLLM output mode: {output_mode}")

    response = litellm.completion(**request_kwargs)
    message = response.choices[0].message
    if output_mode == "json_object":
        content = _stringify_litellm_message_content(getattr(message, "content", None))
        if not content:
            raise RuntimeError("LiteLLM response contained no JSON content")
        outputs = [content]
    else:
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            raise RuntimeError("LiteLLM response contained no structured tool call")
        arguments = getattr(tool_calls[0].function, "arguments", None)
        if isinstance(arguments, str):
            outputs = [arguments]
        else:
            outputs = [json.dumps(_to_jsonable(arguments))]

    usage = _to_jsonable(getattr(response, "usage", None)) or {}
    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        cost = None
    if cost is None:
        cost_ticks = usage.get("cost_in_usd_ticks")
        if isinstance(cost_ticks, (int, float)):
            cost = cost_ticks / 10_000_000_000
    if cost is not None:
        usage["litellm_cost_usd"] = float(cost)

    request_id = getattr(response, "id", None)
    return ProviderBatchResult(
        outputs=outputs,
        request_id=request_id if isinstance(request_id, str) else None,
        usage=usage,
    )


def build_claude_command(
    prompt: str,
    *,
    model_name: str = "sonnet",
    json_schema: dict[str, Any] | None = DEFAULT_BELIEF_JSON_SCHEMA,
) -> list[str]:
    """Build a non-interactive Claude CLI invocation."""
    command = [
        resolve_claude_executable(),
        "-p",
        "--output-format",
        "text",
        "--model",
        model_name,
    ]
    if json_schema is not None:
        command.extend(["--json-schema", json.dumps(json_schema)])
    command.append(prompt)
    return command


def resolve_claude_executable() -> str:
    """Resolve the Claude CLI executable, including common Homebrew locations."""
    discovered = shutil.which("claude")
    if discovered:
        return discovered

    for candidate in ("/opt/homebrew/bin/claude", "/usr/local/bin/claude"):
        if os.path.exists(candidate):
            return candidate

    return "claude"


def run_claude_prompt(
    prompt: str,
    *,
    model_name: str = "sonnet",
    json_schema: dict[str, Any] | None = DEFAULT_BELIEF_JSON_SCHEMA,
    cwd: str | None = None,
    timeout_seconds: int = 180,
) -> str:
    """Run one prompt through the locally authenticated Claude CLI."""
    command = build_claude_command(
        prompt,
        model_name=model_name,
        json_schema=json_schema,
    )
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(stderr or f"Claude CLI exited with code {completed.returncode}")
    return completed.stdout.strip()


def build_openai_chat_payload(
    prompt: str,
    *,
    model_name: str = "gpt-5.4-mini",
    json_schema: dict[str, Any] | None = DEFAULT_BELIEF_JSON_SCHEMA,
    n: int = 1,
    temperature: float = 1.0,
    max_completion_tokens: int = 1200,
) -> dict[str, Any]:
    """Build a Chat Completions payload with optional structured outputs."""
    if n <= 0:
        raise ValueError("n must be positive")
    if n > OPENAI_CHAT_COMPLETIONS_MAX_N:
        raise ValueError(
            f"OpenAI Chat Completions n must be <= {OPENAI_CHAT_COMPLETIONS_MAX_N}, got {n}"
        )

    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "Follow the user's instructions exactly and return only the final answer.",
            },
            {"role": "user", "content": prompt},
        ],
        "n": n,
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
    }
    if json_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "belief_elicitation",
                "strict": True,
                "schema": json_schema,
            },
        }
    return payload


def build_openai_response_payload(
    prompt: str,
    *,
    model_name: str = "gpt-5.4-mini",
    json_schema: dict[str, Any] | None = DEFAULT_BELIEF_JSON_SCHEMA,
    tool_regime: str = "none",
    max_output_tokens: int = 1200,
) -> dict[str, Any]:
    """Build a Responses API payload with optional tool access."""
    payload: dict[str, Any] = {
        "model": model_name,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
    }

    if json_schema is not None:
        payload["text"] = {
            "format": {
                "type": "json_schema",
                "name": "belief_elicitation",
                "strict": True,
                "schema": json_schema,
            }
        }

    if tool_regime == "none":
        payload["tool_choice"] = "none"
    elif tool_regime == "full":
        payload["tools"] = [
            {"type": "web_search"},
            {"type": "code_interpreter", "container": {"type": "auto"}},
        ]
        payload["tool_choice"] = "auto"
        payload["include"] = ["web_search_call.action.sources"]
    else:
        raise ValueError(f"Unsupported tool_regime: {tool_regime}")

    return payload


def run_openai_prompt_batch(
    prompt: str,
    *,
    model_name: str = "gpt-5.4-mini",
    json_schema: dict[str, Any] | None = DEFAULT_BELIEF_JSON_SCHEMA,
    n: int = 1,
    temperature: float = 1.0,
    max_completion_tokens: int = 1200,
    timeout_seconds: int = 180,
) -> list[str]:
    """Run one prompt through OpenAI and return only the sampled outputs."""
    return run_openai_prompt_batch_logged(
        prompt,
        model_name=model_name,
        json_schema=json_schema,
        n=n,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        timeout_seconds=timeout_seconds,
    ).outputs


def run_openai_prompt_batch_logged(
    prompt: str,
    *,
    model_name: str = "gpt-5.4-mini",
    json_schema: dict[str, Any] | None = DEFAULT_BELIEF_JSON_SCHEMA,
    n: int = 1,
    temperature: float = 1.0,
    max_completion_tokens: int = 1200,
    timeout_seconds: int = 180,
) -> ProviderBatchResult:
    """Run one prompt through the OpenAI Chat Completions API and return all choices."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    payload = build_openai_chat_payload(
        prompt,
        model_name=model_name,
        json_schema=json_schema,
        n=n,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error {exc.code}: {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenAI API request failed: {exc.reason}") from exc

    parsed = json.loads(response_body)
    choices = parsed.get("choices", [])
    outputs = []
    for choice in choices:
        message = choice.get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected OpenAI response content: {content!r}")
        outputs.append(content.strip())

    if len(outputs) != n:
        raise RuntimeError(f"Expected {n} choices, got {len(outputs)}")
    return ProviderBatchResult(
        outputs=outputs,
        request_id=parsed.get("id"),
        usage=parsed.get("usage", {}) or {},
    )


def run_openai_response_logged(
    prompt: str,
    *,
    model_name: str = "gpt-5.4-mini",
    json_schema: dict[str, Any] | None = DEFAULT_BELIEF_JSON_SCHEMA,
    tool_regime: str = "none",
    max_output_tokens: int = 1200,
    timeout_seconds: int = 180,
) -> ProviderBatchResult:
    """Run one prompt through the OpenAI Responses API."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    payload = build_openai_response_payload(
        prompt,
        model_name=model_name,
        json_schema=json_schema,
        tool_regime=tool_regime,
        max_output_tokens=max_output_tokens,
    )
    parsed = _post_openai_json(
        "https://api.openai.com/v1/responses",
        payload,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )

    output = parsed.get("output", []) or []
    text_outputs: list[str] = []
    tool_trace: list[dict[str, Any]] = []
    tool_sources: list[str] = []

    for item in output:
        item_type = item.get("type")
        if item_type == "message":
            text = _extract_response_message_text(item)
            if text:
                text_outputs.append(text)
        elif isinstance(item_type, str) and item_type.endswith("_call"):
            tool_trace.append(item)
            if item_type == "web_search_call":
                action = item.get("action", {}) or {}
                sources = action.get("sources", []) or []
                for source in sources:
                    url = source.get("url")
                    if isinstance(url, str) and url:
                        tool_sources.append(url)
        else:
            tool_trace.append(item)

    if not text_outputs:
        raise RuntimeError("Responses API returned no message output")

    return ProviderBatchResult(
        outputs=[text_outputs[-1]],
        request_id=parsed.get("id"),
        usage=parsed.get("usage", {}) or {},
        tool_trace=tool_trace,
        tool_sources=tool_sources,
    )


def _post_openai_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error {exc.code}: {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenAI API request failed: {exc.reason}") from exc

    return json.loads(response_body)


def _extract_response_message_text(message_item: dict[str, Any]) -> str:
    content = message_item.get("content", []) or []
    chunks: list[str] = []
    for part in content:
        if part.get("type") == "output_text":
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def _stringify_litellm_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
            else:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunk for chunk in chunks if chunk).strip()
    return ""


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _to_jsonable(value.dict())
    if hasattr(value, "__dict__"):
        return _to_jsonable(vars(value))
    return str(value)
