# Reproducing the experiment

The publication includes **GLM-5.3 and Claude Opus 4.8**. No model fallback is used.
There are two distinct reproduction tasks: recomputing existing observations
(no API calls), and rerunning inference (paid APIs; results can change).

## 1. Verify existing evidence without credentials

From the repository root:

```sh
uv sync --locked --group dev
uv run --no-sync python -m benchmarks.report --verify
uv run --no-sync python -m benchmarks.quality
```

The first command checks every recorded file against `SHA256SUMS`, recomputes
costs from raw provider usage, and compares aggregates with the published values.
The second independently grades saved answers against the original synthetic
records, and emits question/expected/actual evidence plus cost per correct answer.
Generated reports go to `experiments/generated/`; committed evidence is untouched.
Neither command requires credentials or sends a network request.

## 2. Run new API measurements

Set `JEV_API_KEY`, `ZAI_API_KEY`, and `ANTHROPIC_API_KEY` as environment variables,
or copy the corresponding root `.jev.example`, `.zai.example`, and
`.anthropic.example` files to their names without `.example` and fill in values.
Never commit the filled files.

Check the fixtures with **no keys and no API calls**:

```sh
uv run --no-sync python -m benchmarks.zai_compare \
  --dry-run --output experiments/runs/dataset-check
```

Run against the exact included histories (these incur provider charges):

```sh
uv run --no-sync python -m benchmarks.zai_compare \
  --output experiments/runs/glm-new
uv run --no-sync python -m benchmarks.anthropic_compare \
  --output experiments/runs/opus48-new
uv run --no-sync python -m benchmarks.report \
  --glm-run experiments/runs/glm-new \
  --opus-run experiments/runs/opus48-new \
  --output experiments/generated/new
uv run --no-sync python -m benchmarks.quality \
  --glm-run experiments/runs/glm-new \
  --opus-run experiments/runs/opus48-new \
  --output experiments/generated/new/quality.json
```

Choose fresh run directories. Runners reject overwriting the committed recordings.
Use `--conditions jev` for a smaller smoke run; the complete report requires all
three conditions and all four cases. Pricing constants are the 22 September 2026
list rates; check current provider rates before a new run. Changing them means a
new cost basis, not a revision to the original measured token counts.

## Workload and evaluation

Histories are deterministic shuffled, non-repeated chunks (roughly 6,500 bytes)
from Deep Agents Python source at commit `11c7669f5dddd472bdbf20c97bf4507f365e002b`.
They are formatted as assistant `read_file` calls and tool results. Three synthetic
service ledgers occur near 12%, 48%, and 78% of the chunk sequence. Seeds are 17
and 29. Approximate targets are 60k and 120k tokens; provider tokenization differs.
Six tail checkpoints make ledgers old enough to be selected or summarized.

Questions successively request `lease_seconds`, `max_payload_bytes`, `release_tag`,
and `owner_team` for orion-auth, lyra-search, and vega-export. Expected values are
not in the questions. Each model/method has 12 answers and 48 graded fields.

The correctness evaluator parses JSON, permitting a surrounding Markdown code
fence, and compares values and types against the independent dataset record.
A fully correct answer also has exactly the four required keys. It reports
parse failures, missing/extra keys, nulls, wrong values, field accuracy, and
complete-answer accuracy. No LLM judge is used. These metrics validate exact
retrieval, not semantic reasoning, code execution, or production task quality.

## Fixed settings

- Real Deep Agents graph and in-memory checkpointer; three continuing turns.
- Same initial history and questions for every strategy.
- Summarization: same answer model, standard prompt, 16k trigger, keep six messages,
  `trim_tokens_to_summarize=None`; the summary is reused rather than regenerated each turn.
- JEV: `jev-latest`, 16k message budget, 4k trigger, keep six, threshold 0.2.
- Full: no compaction or selection. Tool definitions remain; `tool_choice=none`.
- No extra retrieval or archive recovery, edits, or deployments during the benchmark.
- Answer output limit 2,048; summary limit 8,192; reasoning effort low.
- GLM temperature 0; Opus provider default. Strategy order reverses between seeds.
- GLM automatic cache; Opus explicit ephemeral five-minute caching on system,
  final tool schema, and request. Caches were not force-cleared. Two Opus probes
  preceded the main run and may have warmed common prefixes.

These 1M-window models would not hit the default 85% compaction threshold at the
tested sizes. The earlier trigger tests cost control, not untouched defaults.
JEV pays for repeated relevance selection; long-horizon economics can differ.

## Recorded files and provenance

`recorded/zai_context_comparison/run1/tokens*/dataset.json` contains original
histories and expected ledger records. Both model runs use these exact datasets.
Each case's `manifest.json` records settings and a canonical-JSON dataset SHA-256.
The SHA256SUMS file separately checks the actual bytes of every recorded artifact.

Each method directory contains `call_*.json` with request bodies and raw responses
(no auth headers), `events.jsonl` with usage/timing, and `result.json` with answers
and selections. Backend archive duplicates and intermediate progress snapshots
are omitted: the original input is already present in dataset.json.
Historical relative `source_dataset` labels are preserved as provenance; resolve
`results/zai_context_comparison/run1/…` under `experiments/recorded/` here.
Examples such as `/home/user/project` inside source excerpts belong to upstream
source text, not private experiment files.

The live generator helper uses the installed Deep Agents source; exact reruns use
the included dataset fixtures. Provider aliases, server behavior, nondeterminism,
and cache warmth can change results. The lockfile fixes client packages; it cannot
freeze remote models. The original run used Deep Agents from the pinned upstream
checkout; this release is tested using the published 0.7.16 package.

## Rates and references

USD per million tokens: GLM input 1.40 / cached input 0.26 / output 4.40;
Opus 4.8 input 5 / 5m cache write 6.25 / cache read 0.50 / output 25;
JEV input 0.042, output free. Thinking output is included once.

- https://docs.z.ai/guides/overview/pricing
- https://platform.claude.com/docs/en/about-claude/pricing
- https://docs.typesafe.ai/models
- https://context-selection-field-notes.ayukawa-hiroshi.chatgpt.site

Figures are usage-based estimates, not invoice or account-balance measurements.
They exclude taxes, contract discounts, diagnostic calls, and report production.

## 3. Evaluate the final executable artifact

The follow-up benchmark asks the final chat model to implement
`evaluate_request(payload_bytes, age_seconds)` from the historical ledger and
explicit decision rules. Its generated Python source is then executed against
**32 tests per artifact**: 12 explicit boundaries/priority cases and 20 deterministic
pseudo-random inputs. The expected outputs are derived independently from the
original ledger, not from the model's explanation. Each model/method generates
12 artifacts. A passing artifact must pass every test, including returned values,
types, metadata, and rejection-priority behavior.

```sh
# Paid API run; keeps the same long histories and three context strategies.
uv run --no-sync python -m benchmarks.end_to_end \
  --output experiments/runs/new-final-artifacts
# Re-execute saved code and rebuild cost/success metrics, without API calls.
uv run --no-sync python -m benchmarks.end_to_end_report --verify
```

For a newly generated run, add `--runs experiments/runs/new-final-artifacts`
to the report command and omit `--verify` (which compares the committed run). The committed results live in `recorded/end-to-end/`.
Failed attempts remain in total cost and the artifact-success denominator.
Cost per passing artifact divides the total spend, including failed attempts,
by the number of fully passing artifacts; if none pass, that metric is null.

The final output is executable text, not a live deployment or external tool action.
The worker accepts only a small side-effect-free Python subset, disables builtins
except annotation types, rejects imports/calls/loops/attribute access, applies a
CPU limit, and runs without provider credentials. It requires Linux or macOS.
This validates the stated policy task, not arbitrary generated programs or general
coding ability. The 32 tests on one artifact are correlated, not 32 independent
model trials. See `benchmarks/code_quality.py` and `code_worker.py` for the contract.

Publication metadata note: the follow-up runner initially inherited GLM pricing
and temperature labels in its Opus manifests. Those manifest labels were corrected
to match the actual request bodies and pricing function before publication. Raw
requests, responses, generated artifacts, and measured costs were unchanged.

## 4. Separate startup and cached continuation costs

```sh
uv run --no-sync python -m benchmarks.warm_cache_report
uv run --no-sync python -m benchmarks.readme_chart
uv run --no-sync python -m benchmarks.warm_cache_chart
```

`warm-cache.json` splits the same executable-artifact experiment into turn 1,
turns 2–3, and all three turns. No new API calls are made. Costs are recalculated
from raw provider responses, including JEV scoring; the continuation slice excludes
all first-turn costs for every method. The cache-hit fraction uses answer-model
cache-read tokens divided by all answer-model input tokens (including cache writes).
Correctness counts use the saved functional grades, independently re-executed by
`end_to_end_report --verify`. The charts read the committed summary JSON files.

The full-history continuation cache-hit fractions are 99.48% (GLM) and 99.56% (Opus).
JEV costs 68.0% and 70.9% less than cached full context in that slice, respectively.
Reused summaries are cheaper than JEV on these turns. This is a two-turn continuation
slice, not a separately pre-warmed run or an estimate of steady-state/long-session costs.
