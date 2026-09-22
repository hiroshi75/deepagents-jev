# deepagents-jev

**Choose relevant original context instead of generating a summary.** A small,
reversible context-selection middleware for [LangChain Deep Agents](https://github.com/langchain-ai/deepagents), powered by [TypeSafe JEV](https://docs.typesafe.ai/models).

JEV scores older conversation fragments against the current task. The middleware
sends selected original messages to your chat model while keeping the complete
history in the checkpoint. A fragment omitted now can return for a later question.
It replaces Deep Agents' built-in summarization slot; no fork or monkey patch is required.

[Read the experiment report](https://context-selection-field-notes.ayukawa-hiroshi.chatgpt.site)
· [Configuration](docs/configuration.md)
· [Reproduce the experiments](experiments/README.md)

## Install

Python 3.11+ is required. Compatibility is currently pinned and tested against
Deep Agents **0.7.16**. The package has not been published to PyPI; install from Git:

```sh
pip install 'git+https://github.com/hiroshi75/deepagents-jev.git'
```

For development and the included experiment tools:

```sh
git clone git@github.com:hiroshi75/deepagents-jev.git
cd deepagents-jev
uv sync --locked --group dev
```

The lockfile uses public PyPI packages. No local Deep Agents checkout is needed.

## Add it to an agent

Set `JEV_API_KEY` in your environment, or copy `.jev.example` to `.jev` and fill
in your key. Configure your chat model's provider credentials separately.
JEV selects context; your ordinary chat model still generates answers and calls tools.

```python
from deepagents import create_deep_agent
from deepagents_jev import JevContextMiddleware, SelectionConfig
from langgraph.checkpoint.memory import InMemorySaver

# Use the LangChain chat model already configured in your application.
def create_agent(model):
    context = JevContextMiddleware(
        config=SelectionConfig(
            max_tokens=16_000,
            trigger_tokens=4_000,
            keep_recent=6,
            relevance_threshold=0.2,
        ),
    )
    return create_deep_agent(
        model=model,
        middleware=[context],
        checkpointer=InMemorySaver(),
    )

# agent = create_agent(your_chat_model)
# config = {"configurable": {"thread_id": "my-project"}}
# agent.invoke({"messages": [("user", "Investigate the authentication code.")]}, config)
# Reuse the thread_id for subsequent questions; ainvoke is also supported.
```

A complete model-selectable example is included:

```sh
uv run --no-sync python examples/agent.py \
  --model anthropic:claude-opus-4-8 --task 'Explain the package structure.'
```

This example makes paid JEV and chat-model calls and runs the normal Deep Agents
tool loop. Install any additional LangChain provider integration your model needs.
To exercise selection and fragment recovery with JEV alone:

```sh
uv run --no-sync python examples/live_selection.py
```

## What stays intact

- Full checkpoint history; only the next model request is filtered.
- All user/system messages and the recent-message tail.
- Tool calls paired with their results, in original order.
- Opaque media, signed reasoning, incomplete tool transactions, and oversized groups.

The budget covers history messages, not system prompts, tool schemas, or output.
If protected history alone is too large, the middleware raises `ContextBudgetExceeded`.
Scoring failures raise `JevError` by default. The optional `fail_open` setting allows
full-history fallback, which can exceed your budget. See [configuration](docs/configuration.md).

Eligible context text is sent to the TypeSafe API. The standard Deep Agents
summarizer also archives originals; this project's difference is selection of
original text for the next request, rather than generation of a replacement summary.

## Measured cost **and** correctness

Four cases: two long-history sizes × two seeds, with three questions per case.
Each answer must recover four exact values from synthetic release-ledger records
buried in historical source-file reads. Costs include observed caching and JEV fees.

|Model|Method|Total USD|Correct fields|Fully correct answers|
|---|---|---:|---:|---:|
|GLM-5.3|Summarization|$0.484313|48/48|12/12|
|GLM-5.3|JEV selection|$0.059991|48/48|12/12|
|GLM-5.3|Full context|$0.590291|48/48|12/12|
|Claude Opus 4.8|Summarization|$2.907931|48/48|12/12|
|Claude Opus 4.8|JEV selection|$0.130132|48/48|12/12|
|Claude Opus 4.8|Full context|$3.878537|48/48|12/12|

JEV cost **87.6% less for GLM** and **95.5% less for Opus 4.8** than summarization
across this three-turn workload, at the same measured exact-answer accuracy.
Summarization was cheaper on the subsequent two turns after its initial cost.
This is a sparse fact-retrieval benchmark, not a general coding-quality evaluation.
The 16k summary trigger was deliberately earlier than the default context-window
threshold, and tools could not reopen archived history. No invoice comparison,
long-session savings guarantee, or statistical equivalence claim is implied.

Original datasets, questions, answers, raw API usage, and checksum manifests are
included. Regenerate costs and independently grade answers **without API calls**:

```sh
uv run --no-sync python -m benchmarks.report --verify
uv run --no-sync python -m benchmarks.quality
```

For live reruns, see [experiments/README.md](experiments/README.md).

### Final executable output

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
uv run --no-sync python -m benchmarks.end_to_end_report --verify
```

The saved code, expected/actual test outputs, raw API responses, and costs are
included in `experiments/recorded/end-to-end/` and
[final-artifacts.json](experiments/summary/final-artifacts.json).

## Repository layout

```text
src/deepagents_jev/       Installable middleware and JEV client
examples/                Minimal agent and live selection examples
tests/                   Offline client, middleware, graph, and evaluation tests
benchmarks/              Live runners, cost aggregation, final-output evaluators
docs/                    Configuration and design constraints
experiments/recorded/    Immutable GLM and Opus 4.8 datasets and API records
experiments/summary/     Published cost and correctness results
experiments/SHA256SUMS   Checksums of recorded evidence
.github/workflows/      Offline CI (no API keys)
```

## Development

```sh
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests examples benchmarks
uv run --no-sync ty check src
uv build
```

The distribution contains the middleware, not the large recorded experiment data.
Clone this repository for the full experiment bundle. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Inspiration and license

Inspired by [@CompleteSkeptic's post](https://x.com/CompleteSkeptic/status/2101894250401271876)
and the memo [Why yet another agent?](https://docs.google.com/document/d/1G61uUB0FifUnmmrPzFQojZ3KpczYKmXGpgEXDJ2l_Zg/edit).
This repository tests an independent implementation of query-aware selection.

MIT. Dataset source excerpts retain LangChain's MIT notice; see
[NOTICE](NOTICE) and [LICENSES/DeepAgents-MIT.txt](LICENSES/DeepAgents-MIT.txt).
