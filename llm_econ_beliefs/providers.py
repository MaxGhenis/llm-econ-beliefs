"""Provider adapters for running prompts against locally available CLIs."""

from __future__ import annotations

import json
import os
import subprocess
import shutil
from urllib import error, request
from typing import Any

from .models import ProviderBatchResult


OPENAI_CHAT_COMPLETIONS_MAX_N = 8


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
