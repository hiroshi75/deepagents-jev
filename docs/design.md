# Design choices

A middleware is sufficient: Deep Agents 0.7.16 supports replacing its built-in
summarization middleware by name. The adapter reports `SummarizationMiddleware`
as its name and uses model-call wrapping, leaving persisted graph state intact.
Forking the agent runtime would add upgrade work without improving this mechanism.

At each model call the adapter groups original messages, pins protected groups,
extracts the current query, batches JEV relevance requests, then selects groups
within the message budget and restores their original order. The model sees the
original selected bytes, not a generated paraphrase. A later task can select
previously omitted groups from the complete checkpoint history.

The original motivation is the linked memo's query-dependent meta-attention idea.
This implementation does not implement all proposals in that memo: model routing,
dynamic tool selection, within-output line selection, or a measured KV-cache
optimizer are outside its scope.

The complete implementation is in `src/deepagents_jev/`. Benchmarks are deliberately
separate: the published wheel does not import the experiment runners or ship raw
request histories. Tests use fake chat models and mock JEV transport; live examples
and benchmark entry points are explicit opt-ins to paid API calls.
