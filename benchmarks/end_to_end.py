"""Live final-artifact comparison: generate executable policies, then run 32 tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_core.messages import messages_from_dict

from benchmarks.anthropic_compare import model_factory
from benchmarks.code_quality import grade, question
from benchmarks.zai_compare import (
    DATASETS,
    ROOT,
    read_key,
    run_condition,
    validate_output,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/runs/end-to-end")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["glm-5.3", "claude-opus-4-8"],
        default=["glm-5.3", "claude-opus-4-8"],
    )
    args = parser.parse_args()
    validate_output(args.output)
    for model in args.models:
        destination = args.output / model
        if destination.exists() and any(destination.iterdir()):
            parser.error(f"Output for {model} already contains data; select a fresh directory")
    for model in args.models:
        key = (
            read_key("ZAI_API_KEY", ".zai")
            if model == "glm-5.3"
            else read_key("ANTHROPIC_API_KEY", ".anthropic")
        )
        for source in sorted(DATASETS.glob("tokens*/dataset.json")):
            dataset = json.loads(source.read_text())
            original = json.loads(source.with_name("manifest.json").read_text())
            base = args.output / model / source.parent.name
            write_json(
                base / "manifest.json",
                {
                    **original,
                    "model": model,
                    "task": "executable request-admission policy",
                    "tests_per_artifact": 32,
                    "source_dataset": source.parent.name + "/dataset.json",
                    "temperature": 0 if model == "glm-5.3" else "provider default",
                    "prices_usd_per_million": original["prices_usd_per_million"]
                    if model == "glm-5.3"
                    else {
                        "input": 5,
                        "output": 25,
                        "cache_write_5m": 6.25,
                        "cache_write_1h": 10,
                        "cache_read": 0.5,
                        "jev_input": 0.042,
                    },
                    "prompt_caching": "automatic"
                    if model == "glm-5.3"
                    else "Deep Agents native AnthropicPromptCachingMiddleware (5m)",
                },
            )
            for condition in original["condition_order"]:
                run_condition(
                    base,
                    condition,
                    messages_from_dict(dataset["messages"]),
                    dataset["records"],
                    key,
                    model_factory=model_factory if model == "claude-opus-4-8" else None,
                    question_builder=question,
                    answer_grader=grade,
                )


if __name__ == "__main__":
    main()
