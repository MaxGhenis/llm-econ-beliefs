Note: Toy public-finance mapping from each model's pooled ETI distribution to an optimal top marginal tax rate under the Saez top-bracket formula tau* = (1 - g_bar) / (1 - g_bar + a e), with a Pareto parameter a = 1.621 estimated from the weighted top 1% tax-unit AGI tail in the retained PolicyEngine calibration (threshold $725,533, tail mean $1,894,129). The welfare weight g_bar = a / (a + gamma) = 0.618 is the average marginal utility of top-bracket earners under CRRA (gamma = 1) utility and a Pareto(a) income tail, normalized to the marginal utility of the earner at the top-bracket threshold. It is a threshold-normalized weight, not the population-normalized utilitarian weight, which would drive g_bar toward zero; the Revenue-max column reports that g_bar -> 0 (Diamond-Saez revenue-maximizing) benchmark tau* = 1 / (1 + a e) at the ETI median. ETI is truncated below at zero for this policy mapping. Cached replay uses the frozen calibration artifact; its historical model version, dataset build and simulation period are unknown (results/calibration-provenance.json).

| Model | ETI median [90%] | Top rate median [90%] | Revenue-max median | Top-rate 90% width (pp) |
| --- | --- | --- | --- | --- |
| Claude Opus 5 | 0.337 [0.086, 0.961] | 41.2% [19.7%, 73.2%] | 64.7% | 53.5 |
| Gemini 3.1 Pro | 0.351 [0.103, 0.868] | 40.2% [21.3%, 69.5%] | 63.8% | 48.2 |
| Gemini 3.5 Flash | 0.357 [0.102, 0.832] | 39.8% [22.0%, 69.8%] | 63.4% | 47.7 |
| GPT-5.5 | 0.369 [0.122, 0.994] | 39.0% [19.1%, 65.9%] | 62.6% | 46.8 |
| Kimi K3 | 0.371 [0.081, 0.992] | 38.8% [19.2%, 74.4%] | 62.4% | 55.2 |
| Qwen 3.8 Max | 0.372 [0.105, 1.131] | 38.7% [17.2%, 69.2%] | 62.4% | 51.9 |
| GPT-5.6 Luna | 0.377 [0.068, 1.202] | 38.4% [16.4%, 77.5%] | 62.0% | 61.1 |
| Claude Opus 4.8 | 0.383 [0.105, 0.993] | 38.0% [19.2%, 69.1%] | 61.7% | 49.9 |
| Gemini 3.1 Flash-Lite | 0.389 [0.101, 0.951] | 37.7% [19.8%, 69.9%] | 61.3% | 50.1 |
| Claude Opus 4.7 | 0.400 [0.123, 0.993] | 37.0% [19.2%, 65.7%] | 60.7% | 46.5 |
| Gemini 3 Flash | 0.400 [0.131, 1.040] | 37.0% [18.5%, 64.3%] | 60.7% | 45.8 |
| Grok 4.1 Fast | 0.400 [0.109, 1.191] | 37.0% [16.5%, 68.4%] | 60.7% | 51.9 |
| Grok 4.5 | 0.410 [0.105, 1.128] | 36.5% [17.3%, 69.3%] | 60.1% | 52 |
| GPT-5.4 | 0.420 [0.150, 0.996] | 35.9% [19.1%, 61.1%] | 59.5% | 42 |
| Gemini 3.6 Flash | 0.423 [0.112, 0.897] | 35.8% [20.8%, 67.7%] | 59.3% | 46.9 |
| GPT-5.4 mini | 0.437 [0.116, 1.377] | 35.0% [14.6%, 67.1%] | 58.6% | 52.5 |
| GPT-5.6 Sol | 0.431 [0.150, 1.165] | 35.3% [16.8%, 61.1%] | 58.9% | 44.3 |
| Claude Fable 5 | 0.437 [0.151, 1.076] | 35.0% [17.9%, 60.9%] | 58.6% | 42.9 |
| Grok 4.3 | 0.439 [0.133, 1.091] | 34.9% [17.8%, 64.0%] | 58.4% | 46.2 |
| MiniMax M3 | 0.438 [0.122, 1.330] | 34.9% [15.0%, 65.9%] | 58.5% | 50.9 |
| Inkling | 0.443 [0.151, 0.883] | 34.7% [21.0%, 60.9%] | 58.2% | 39.9 |
| Claude Sonnet 5 | 0.471 [0.145, 1.146] | 33.3% [17.0%, 61.9%] | 56.7% | 44.9 |
| GPT-5.6 Terra | 0.492 [0.166, 1.182] | 32.4% [16.6%, 58.7%] | 55.6% | 42.1 |
| DeepSeek V4 Pro | 0.479 [0.073, 1.808] | 32.9% [11.5%, 76.3%] | 56.3% | 64.7 |
| GLM-5.2 | 0.495 [0.156, 1.226] | 32.2% [16.1%, 60.1%] | 55.5% | 44 |
| Claude Sonnet 4.6 | 0.500 [0.138, 1.282] | 32.0% [15.5%, 63.1%] | 55.2% | 47.6 |
| Grok 4.20 | 0.500 [0.174, 1.243] | 32.0% [15.9%, 57.5%] | 55.2% | 41.6 |
| Kimi K2.6 | 0.499 [0.146, 1.241] | 32.1% [15.9%, 61.7%] | 55.3% | 45.8 |
| Claude Haiku 4.5 | 0.502 [0.145, 1.299] | 31.9% [15.3%, 61.9%] | 55.1% | 46.6 |
| GPT-5.4 nano | 0.546 [0.151, 1.461] | 30.1% [13.9%, 60.9%] | 53.1% | 47 |
| Qwen 3.7 Max | 0.555 [0.161, 1.415] | 29.8% [14.3%, 59.4%] | 52.6% | 45.1 |
