"""Build a cost/quality report exclusively from persisted provider usage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from benchmarks.zai_compare import ROOT, response_cost, write_json


def summarize(root: Path) -> dict:
    conditions: dict[str, list] = defaultdict(list)
    for path in sorted(root.glob("tokens*/*/result.json")):
        result = json.loads(path.read_text())
        result["case"] = path.parent.parent.name
        result["refusals"] = 0
        result["empty_answers"] = sum(not turn["answer"] for turn in result["turns"])
        for event in result["events"]:
            raw = json.loads((path.parent / event["raw_file"]).read_text())
            event["nominal_usage_usd"] = event["estimated_usd"]
            event["estimated_usd"], event["uncached_usd"] = response_cost(
                event["provider"], raw["response"]
            )
            event["stop_reason"] = raw["response"].get("stop_reason")
            event["stop_details"] = raw["response"].get("stop_details")
            if event["provider"] == "anthropic" and event["stage"] == "answer":
                result["refusals"] += raw["response"].get("stop_reason") == "refusal"
        for turn in result["turns"]:
            events = [e for e in result["events"] if e["turn"] == turn["turn"]]
            for key in ("estimated_usd", "uncached_usd"):
                turn[key] = sum(e[key] for e in events)
        for key in ("estimated_usd", "uncached_usd"):
            result[key] = sum(e[key] for e in result["events"])
        conditions[result["condition"]].append(result)
    aggregate = {}
    for condition, results in conditions.items():
        events = [event for r in results for event in r["events"]]
        glm = [e for e in events if e["provider"] == "zai"]
        opus = [e for e in events if e["provider"] == "anthropic"]
        jev = [e for e in events if e["provider"] == "jev"]
        aggregate[condition] = {
            "cases": len(results),
            "refusals": sum(r["refusals"] for r in results),
            "empty_answers": sum(r["empty_answers"] for r in results),
            "turns": sum(len(r["turns"]) for r in results),
            "estimated_usd": sum(r["estimated_usd"] for r in results),
            "uncached_usd": sum(r["uncached_usd"] for r in results),
            "llm_usd": sum(e["estimated_usd"] for e in glm + opus),
            "summary_usd": sum(e["estimated_usd"] for e in events if e["stage"] == "summary"),
            "jev_usd": sum(e["estimated_usd"] for e in jev),
            "llm_input_tokens": sum(e["usage"]["prompt_tokens"] for e in glm)
            + sum(
                e["usage"]["input_tokens"]
                + e["usage"].get("cache_creation_input_tokens", 0)
                + e["usage"].get("cache_read_input_tokens", 0)
                for e in opus
            ),
            "llm_cache_read_tokens": sum(
                e["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0) for e in glm
            )
            + sum(e["usage"].get("cache_read_input_tokens", 0) for e in opus),
            "llm_cache_write_tokens": sum(
                e["usage"].get("cache_creation_input_tokens", 0) for e in opus
            ),
            "llm_output_tokens": sum(e["usage"]["completion_tokens"] for e in glm)
            + sum(e["usage"]["output_tokens"] for e in opus),
            "jev_input_tokens": sum(e["usage"]["input_tokens"] for e in jev),
            "jev_requests": len(jev),
            "llm_requests": len(glm) + len(opus),
            "total_seconds": sum(r["elapsed_seconds"] for r in results),
            "correct_fields": sum(r["correct_fields"] for r in results),
            "total_fields": sum(r["total_fields"] for r in results),
            "first_turn_usd": sum(r["turns"][0]["estimated_usd"] for r in results),
            "continuation_usd": sum(t["estimated_usd"] for r in results for t in r["turns"][1:]),
        }
    return {"aggregate": aggregate, "cases": dict(conditions)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--glm-run", type=Path, default=ROOT / "experiments/recorded/zai_context_comparison/run1"
    )
    parser.add_argument(
        "--opus-run",
        type=Path,
        default=ROOT / "experiments/recorded/opus48_context_comparison/run1",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/generated")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify committed files and compare rebuilt aggregates with the publication",
    )
    args = parser.parse_args()
    from benchmarks.zai_compare import validate_output

    validate_output(args.output)
    if args.verify:
        for line in (ROOT / "experiments/SHA256SUMS").read_text().splitlines():
            digest, name = line.split("  ", 1)
            path = ROOT / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Checksum mismatch: {name}")
    data = {"glm-5.3": summarize(args.glm_run), "claude-opus-4-8": summarize(args.opus_run)}
    if any(
        len(model["cases"].get(c, [])) != 4
        for model in data.values()
        for c in ["full", "jev", "summarization"]
    ):
        raise ValueError("Expected four cases for each model and method")
    if args.verify:
        reference = json.loads((ROOT / "experiments/summary/results.json").read_text())
        for model, d in data.items():
            for method, aggregate in d["aggregate"].items():
                for key, expected in reference[model]["aggregate"][method].items():
                    if not math.isclose(aggregate[key], expected, rel_tol=1e-10, abs_tol=1e-12):
                        raise ValueError(f"Aggregate mismatch: {model}/{method}/{key}")
    write_json(args.output / "results.json", data)
    lines = [
        "# Recorded context-selection experiment",
        "",
        "USD: provider-reported usage × list rates, including cache and JEV fees.",
        "",
        "|Model|Method|LLM|JEV|Total|Exact fields|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model, d in data.items():
        for method in ["summarization", "jev", "full"]:
            a = d["aggregate"][method]
            lines.append(
                f"|{model}|{method}|${a['llm_usd']:.6f}|${a['jev_usd']:.6f}|${a['estimated_usd']:.6f}|{a['correct_fields']}/{a['total_fields']}|"
            )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if args.verify:
        print("All recorded file checksums and published aggregates verified. No API calls made.")


if __name__ == "__main__":
    main()
