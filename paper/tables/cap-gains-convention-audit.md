Note: Both capital-gains-realizations conventions elicited independently under prompt v4. Under the identity epsilon_taxrate = -(tau / (1 - tau)) * epsilon_netoftax, the observed ratio epsilon_taxrate / epsilon_netoftax pins down an implied tau; a model that answers consistently across conventions implies a positive epsilon_netoftax, a negative epsilon_taxrate, and an implied tau in the plausible U.S. top-bracket long-term-capital-gains range. The last column flags each cell as 'in band' (implied tau between 0.15 and 0.30), 'plausible sign, outside band' (implied tau in (0, 1) but outside that window), or 'out of band' (implied tau non-positive or > 1).

| Model | Provider | epsilon w.r.t. tax rate | epsilon w.r.t. net-of-tax rate | Implied tau | Consistency | In plausible tau band [0.15, 0.30] |
| --- | --- | --- | --- | --- | --- | --- |
| Claude Haiku 4.5 | Anthropic | -0.783 | 0.893 | 0.467 | as expected (tax<0, net>0) | plausible sign, outside band |
| Claude Opus 4.7 | Anthropic | -0.7 | 0.7 | 0.5 | as expected (tax<0, net>0) | plausible sign, outside band |
| Claude Sonnet 4.6 | Anthropic | -0.7 | 4.6 | 0.132 | as expected (tax<0, net>0) | plausible sign, outside band |
| GPT-5.4 | OpenAI | -0.787 | 0.913 | 0.463 | as expected (tax<0, net>0) | plausible sign, outside band |
| GPT-5.4 mini | OpenAI | -0.93 | 1.04 | 0.472 | as expected (tax<0, net>0) | plausible sign, outside band |
| Gemini 3 Flash | Google | -0.793 | 0.807 | 0.496 | as expected (tax<0, net>0) | plausible sign, outside band |
| Gemini 3.1 Flash-Lite | Google | -0.747 | 0.88 | 0.459 | as expected (tax<0, net>0) | plausible sign, outside band |
| Gemini 3.1 Pro | Google | -0.67 | 0.373 | 0.642 | as expected (tax<0, net>0) | plausible sign, outside band |
| Grok 4.1 Fast | xAI | -0.767 | 1.527 | 0.334 | as expected (tax<0, net>0) | plausible sign, outside band |
| Grok 4.20 | xAI | -0.553 | 0.67 | 0.452 | as expected (tax<0, net>0) | plausible sign, outside band |
| GPT-5.4 nano | OpenAI | -0.373 | -0.167 | 1.806 | both negative | out of band |
