# Configuration and behavior

`JevContextMiddleware` replaces the built-in `SummarizationMiddleware` by using
Deep Agents' supported same-name middleware replacement. It changes only the
messages passed to the model; it does not emit summaries, delete messages, or
replace checkpoint history. Start a new thread; this cannot recover original
text that was already lost in an older compacted checkpoint.

|SelectionConfig field|Default|Meaning|
|---|---:|---|
|max_tokens|16000|History-message budget; reserve system/tool/output headroom separately|
|trigger_tokens|4000|Skip JEV below this approximate message-token count|
|keep_recent|6|At least this many trailing messages remain pinned|
|relevance_threshold|0.2|Discard groups below this relevance score|
|min_savings_ratio|0.15|Preserve a fitting prefix if the reduction is too small|
|query_bytes|6000|Reject an oversized current query rather than silently truncating|
|fragment_bytes|12000|Pin larger groups rather than splitting tool transactions|
|fail_open|false|Raise on scoring errors; true explicitly permits full-history fallback|

All user/system messages are pinned. Tool calls and their contiguous results
form atomic groups; opaque media, signed reasoning, incomplete tool transactions,
and oversized groups are protected. If pinned messages exceed the budget,
`ContextBudgetExceeded` is raised. A model-specific `token_counter` may replace
the default LangChain approximate counter.

JEV receives query text and eligible original fragments through the official
TypeSafe API. The bounded score cache contains hashes and scores. Selection
reports contain counts, opaque fragment IDs, scores, and a reason, not raw text
or keys. This is input-cost management, not a reduction in stored history.

Use environment variable `JEV_API_KEY` (or `TYPESAFE_API_KEY`) or an explicit
`.jev` dotenv file. `JevClient(key_file=...)` does not search parent directories
or perform variable interpolation. Model-provider credentials are separate.
JevClient defaults: `jev-latest`, 15-second timeout, two retries, 2,048 cached
scores, batches of at most 24 groups / 24,000 bytes. Benchmarks use a 30-second
timeout. Scores are query-specific; changing the question can incur new JEV calls.

## Subagents and async calls

The default general-purpose and forked subagents inherit the replacement.
Set `middleware: [context]` explicitly on custom declarative subagents.
Compiled or remote subagents need it in their own graph. Do not also install
another summarizer or a `compact_conversation` tool. Both `invoke` and `ainvoke`
are supported. Ordinary large-tool-output file offloading still applies.

## Scope

Selection is by message/tool-result group, not by arbitrary lines inside a tool
result. The prefix-saving heuristic is not a measured KV-cache cost optimizer.
Relevance is model judgment and may omit useful evidence; the original remains
available for later selection. No claim of general coding-quality improvement
is made by the included sparse-retrieval benchmark.
