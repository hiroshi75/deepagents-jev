# Claude Opus 4.8: context selection cost experiment

Measured 22 September 2026. Same original corpus, questions, middleware, and explicit
5-minute prompt-cache configuration as the GLM comparison; only the answer/summary
model changes to `claude-opus-4-8`. No fallback model or modified task was used.
Four cases (60k/120k approximate history targets × seeds 17/29), three turns each.

Costs below use provider-reported usage and official rates: $5/M ordinary input,
$6.25/M five-minute cache writes, $0.50/M cache reads, $25/M output.
JEV costs $0.042/M input. Thinking output is included once. These are estimates,
not invoice measurements. Provider caches were not forcibly cleared; two successful
compatibility probes preceded the main run and may have warmed common prefixes.

|Method|LLM USD|JEV USD|Total USD|Exact fields|Refusals|Wall seconds|
|---|---:|---:|---:|---:|---:|---:|
|summarization|$2.907931|$0.000000|$2.907931|48/48|0|112.1|
|jev|$0.076336|$0.053796|$0.130132|48/48|0|109.8|
|full|$3.878537|$0.000000|$3.878537|48/48|0|30.0|

JEV total reduction versus summarization: 95.5%.

This is sparse fact retrieval, not autonomous coding. The 16k compaction threshold is intentionally early; default 85%-window compaction would not trigger here. Archive recovery tools were disabled. Do not extrapolate three-turn savings to much longer sessions.

Reproduce with:
```sh
uv run --no-sync python -m benchmarks.anthropic_compare --model claude-opus-4-8 --output results/opus48_context_comparison/new-run
```

Sources: https://platform.claude.com/docs/en/models/opus-4-8/overview ; https://platform.claude.com/docs/en/about-claude/pricing ; https://docs.typesafe.ai/models
