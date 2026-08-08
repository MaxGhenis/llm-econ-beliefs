Note: Per-model generation-harness configuration. The prompt text and repeated-run design are identical across models; the structured-output mechanism, completion budget, sampling regime, and reasoning configuration follow each provider's API surface and are therefore confounded with model identity. Completion budgets are truncation guards: reasoning tokens count against them on models that reason, so budgets were raised where required to avoid truncation. Identifiers marked alias float with provider updates; dated snapshots are pinned.

| Model | Provider path | Output mechanism | Completion budget | Sampling | Reasoning config | API identifier | Identifier type |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-5.5 | OpenAI Chat Completions | strict JSON schema | 1200 (8000 for the 40 re-elicited runs) | temperature 1.0, batched n <= 8 | provider default effort | gpt-5.5 | alias |
| GPT-5.6 Sol | OpenAI Chat Completions | strict JSON schema | 8000 | temperature 1.0, batched n <= 8 | provider default effort | gpt-5.6-sol | alias |
| GPT-5.6 Luna | OpenAI Chat Completions | strict JSON schema | 8000 | temperature 1.0, batched n <= 8 | provider default effort | gpt-5.6-luna | alias |
| GPT-5.6 Terra | OpenAI Chat Completions | strict JSON schema | 8000 | temperature 1.0, batched n <= 8 | provider default effort | gpt-5.6-terra | alias |
| GPT-5.4 | OpenAI Chat Completions | strict JSON schema | 1200 | temperature 1.0, batched n <= 8 | provider default effort | gpt-5.4 | alias |
| GPT-5.4 mini | OpenAI Chat Completions | strict JSON schema | 1200 | temperature 1.0, batched n <= 8 | provider default effort | gpt-5.4-mini | alias |
| GPT-5.4 nano | OpenAI Chat Completions | strict JSON schema | 1200 | temperature 1.0, batched n <= 8 | provider default effort | gpt-5.4-nano | alias |
| Claude Fable 5 | native Anthropic API | strict JSON schema | 32000 | none accepted (provider default) | always-on reasoning | claude-fable-5 | alias |
| Claude Opus 4.8 | native Anthropic API | strict JSON schema | 32000 | none accepted (provider default) | off (provider default) | claude-opus-4-8 | alias |
| Claude Sonnet 5 | native Anthropic API | strict JSON schema | 32000 | none accepted (provider default) | adaptive (provider default) | claude-sonnet-5 | alias |
| Claude Opus 5 | native Anthropic API | strict JSON schema | 32000 | none accepted (provider default) | adaptive, on by default (provider default) | claude-opus-5 | alias |
| Claude Opus 4.7 | LiteLLM | forced function call | 1200 | temperature 1.0 | off (provider default) | claude-opus-4-7 | alias |
| Claude Sonnet 4.6 | LiteLLM | forced function call | 1200 | temperature 1.0 | off (provider default) | claude-sonnet-4-6 | alias |
| Claude Haiku 4.5 | LiteLLM | forced function call | 1200 | temperature 1.0 | off (provider default) | claude-haiku-4-5-20251001 | dated snapshot |
| Gemini 3.1 Pro | LiteLLM | forced JSON object | 1200 | temperature 1.0 | provider default thinking | gemini-3.1-pro-preview | preview alias |
| Gemini 3.5 Flash | LiteLLM | forced JSON object | 4000 | temperature 1.0 | provider default thinking | gemini-3.5-flash | alias |
| Gemini 3.6 Flash | LiteLLM | forced JSON object | 8000 | temperature 1.0 | provider default thinking | gemini-3.6-flash | alias |
| Gemini 3 Flash | LiteLLM | forced JSON object | 1200 | temperature 1.0 | provider default thinking | gemini-3-flash-preview | preview alias |
| Gemini 3.1 Flash-Lite | LiteLLM | forced JSON object | 1200 | temperature 1.0 | provider default thinking | gemini-3.1-flash-lite-preview | preview alias |
| Grok 4.20 | LiteLLM | forced function call | 1200 | temperature 1.0 | reasoning variant | xai/grok-4.20-reasoning | alias |
| Grok 4.3 | LiteLLM | forced function call | 4000 | temperature 1.0 | provider default | xai/grok-4.3 | alias |
| Grok 4.5 | LiteLLM | forced function call | 8000 | temperature 1.0 | provider default | xai/grok-4.5 | alias |
| DeepSeek V4 Pro | LiteLLM via OpenRouter | forced JSON object (schema validated locally) | 8000 | temperature 1.0 | provider default | openrouter/deepseek/deepseek-v4-pro | alias |
| Qwen 3.7 Max | LiteLLM via OpenRouter | forced JSON object (schema validated locally) | 8000 | temperature 1.0 | provider default | openrouter/qwen/qwen3.7-max | alias |
| Kimi K2.6 | LiteLLM via OpenRouter | forced JSON object (schema validated locally) | 8000 | temperature 1.0 | provider default | openrouter/moonshotai/kimi-k2.6 | alias |
| Kimi K3 | LiteLLM via OpenRouter | forced JSON object (schema validated locally) | 8000 | temperature 1.0 | provider default | openrouter/moonshotai/kimi-k3 | alias |
| GLM-5.2 | LiteLLM via OpenRouter | forced JSON object (schema validated locally) | 16000 | temperature 1.0 | provider default | openrouter/z-ai/glm-5.2 | alias |
| MiniMax M3 | LiteLLM via OpenRouter | forced JSON object (schema validated locally) | 8000 | temperature 1.0 | provider default | openrouter/minimax/minimax-m3 | alias |
| Grok 4.1 Fast | LiteLLM | forced function call | 1200 | temperature 1.0 | non-reasoning variant | xai/grok-4-1-fast-non-reasoning | alias |
| Qwen 3.8 Max | LiteLLM via OpenRouter | forced JSON object (schema validated locally) | 32000 | temperature 1.0 | provider default | openrouter/qwen/qwen3.8-max | alias |
| Inkling | LiteLLM via OpenRouter | forced JSON object (schema validated locally) | 24000 | temperature 1.0 | provider default | openrouter/thinkingmachines/inkling | alias |
