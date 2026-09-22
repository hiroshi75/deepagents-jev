# Contributing

Use Python 3.11 or later and uv. Run `uv sync --locked --group dev`, then:

```sh
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests examples benchmarks
uv run --no-sync ty check src
uv run --no-sync python -m benchmarks.report --verify
uv build
```

Tests and report verification are offline and require no keys. Live examples and
benchmark commands call paid APIs. Never add credentials or private conversations.
Keep `experiments/recorded/` immutable; write new runs under `experiments/runs/`.
The PyPI dependency is deliberately pinned to the tested Deep Agents release.
When upgrading it, verify middleware replacement, checkpoint preservation, and
subagent behavior before changing the pin.
