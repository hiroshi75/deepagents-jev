# Final executable artifact experiment

A separate follow-up asks each model to generate a Python request-admission
function from the same historical ledgers. We execute the final code against
32 independent-oracle checks per artifact, including boundary values, invalid
inputs, rejection priority, exact metadata, and output types. Every test must
pass for an artifact to count as correct. These are fresh API calls with their
own costs, not a relabeling of the JSON-answer experiment above.

|Model|Method|Total USD|Passing artifacts|Functional tests|USD / passing artifact|
|---|---|---:|---:|---:|---:|
|GLM-5.3|Summarization|$0.508680|12/12|384/384|$0.042390|
|GLM-5.3|JEV selection|$0.082033|12/12|384/384|$0.006836|
|GLM-5.3|Full context|$0.603380|12/12|384/384|$0.050282|
|Claude Opus 4.8|Summarization|$3.037862|12/12|384/384|$0.253155|
|Claude Opus 4.8|JEV selection|$0.250455|12/12|384/384|$0.020871|
|Claude Opus 4.8|Full context|$3.975434|12/12|384/384|$0.331286|

JEV cost **83.9% less for GLM** and **91.8% less for Opus 4.8** than summarization
on this executable-policy task. All three methods passed all artifacts: there
was no observed loss of final-output correctness on these cases. This is a small,
constrained programming task, not evidence of equal quality on arbitrary tasks.
Tests within an artifact are correlated, and the four histories share a task
schema; they are not independent real-world software projects.

Replay all 72 generated artifacts and 2,304 functional checks without API calls:

```sh
uv run --no-sync python -m benchmarks.end_to_end_report
```

The saved code, expected/actual test outputs, raw API responses, and costs are
included in `experiments/recorded/end-to-end/` and
[final-artifacts.json](final-artifacts.json).

