"""Run the recorded fixtures with cache-enabled Claude Opus 4.8 calls."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx
from anthropic.types import Message
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import messages_from_dict
from pydantic import Field, SecretStr

from benchmarks.zai_compare import (
    DATASETS,
    ROOT,
    Journal,
    read_key,
    run_condition,
    validate_output,
    write_json,
)


class MeteredOpus(ChatAnthropic):
    """Preserve native Anthropic conversion and Deep Agents' prompt-cache middleware."""

    journal: Any = Field(exclude=True, repr=False)
    stage: str = "answer"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        if payload.get("tools"):
            payload["tool_choice"] = {"type": "none"}
        replayed = self.journal.replay_response("anthropic", self.stage, payload)
        if replayed is not None:
            return self._format_output(Message.model_validate(replayed), **kwargs)
        start = time.perf_counter()
        with httpx.Client(timeout=httpx.Timeout(300, connect=30)) as client:
            response = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.anthropic_api_key.get_secret_value(),
                    "anthropic-version": "2023-06-01",
                },
                json=payload,
            )
        body = response.json()
        self.journal.record(
            "anthropic",
            self.stage,
            payload,
            body,
            time.perf_counter() - start,
            response.status_code,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Anthropic HTTP {response.status_code}; see raw record")
        if body["stop_reason"] not in {"end_turn", "refusal"}:
            raise RuntimeError(f"Incomplete Claude response: {body['stop_reason']}")
        return self._format_output(Message.model_validate(body), **kwargs)


def model_factory(journal: Journal, key: str, model: str = "claude-opus-4-8") -> MeteredOpus:
    return MeteredOpus(
        model=model,
        api_key=SecretStr(key),
        journal=journal,
        max_tokens=2048,
        max_retries=0,
        timeout=300,
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
        profile={"max_input_tokens": 1_000_000},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["claude-opus-4-8"], default="claude-opus-4-8")
    parser.add_argument("--datasets", type=Path, default=DATASETS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--conditions", nargs="+", default=["summarization", "jev", "full"])
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / "experiments/runs/claude-opus-4.8"
    validate_output(args.output)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output already contains data; select a fresh run directory")
    if not list(args.datasets.glob("tokens*/dataset.json")):
        parser.error("No dataset.json fixtures found")
    key = read_key("ANTHROPIC_API_KEY", ".anthropic")
    outputs = []
    for source in sorted(args.datasets.glob("tokens*/dataset.json")):
        dataset = json.loads(source.read_text())
        manifest = json.loads(source.with_name("manifest.json").read_text())
        base = args.output / source.parent.name
        write_json(
            base / "manifest.json",
            {
                **manifest,
                "model": args.model,
                "temperature": "provider default",
                "prices_usd_per_million": {
                    "input": 5,
                    "output": 25,
                    "cache_write_5m": 6.25,
                    "cache_write_1h": 10,
                    "cache_read": 0.5,
                    "jev_input": 0.042,
                },
                "source_dataset": source.parent.name + "/dataset.json",
                "prompt_caching": "Deep Agents native AnthropicPromptCachingMiddleware (5m)",
            },
        )
        for condition in manifest["condition_order"]:
            if condition not in args.conditions:
                continue
            result = run_condition(
                base,
                condition,
                messages_from_dict(dataset["messages"]),
                dataset["records"],
                key,
                model_factory=lambda journal, key: model_factory(journal, key, args.model),
                resume=False,
            )
            outputs.append({"case": base.name, **result})
            write_json(args.output / "results.json", outputs)


if __name__ == "__main__":
    main()
