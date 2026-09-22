"""Split the recorded code benchmark into initial and cached continuation costs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from benchmarks.report import summarize
from benchmarks.zai_compare import ROOT, validate_output, write_json


def build_report(runs: Path) -> dict:
    report = {
        "task": "Executable request-admission policy generation",
        "source": "Same recorded final-artifact experiment; no new API calls",
        "scopes": {"initial": [1], "continuation": [2, 3], "total": [1, 2, 3]},
        "models": {},
    }
    for model in ["glm-5.3", "claude-opus-4-8"]:
        data = summarize(runs / model)
        scopes = {}
        for scope, turn_ids in report["scopes"].items():
            methods = {}
            for method in ["full", "jev", "summarization"]:
                cases = data["cases"][method]
                if len(cases) != 4:
                    raise ValueError("Expected four histories per method")
                turns = [t for c in cases for t in c["turns"] if t["turn"] in turn_ids]
                if len(turns) != 4 * len(turn_ids):
                    raise ValueError("Missing turns")
                events = [e for c in cases for e in c["events"] if e["turn"] in turn_ids]
                inputs = reads = writes = 0
                for event in events:
                    if event["stage"] != "answer":
                        continue
                    usage = event["usage"]
                    if event["provider"] == "zai":
                        inputs += usage["prompt_tokens"]
                        reads += usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                    elif event["provider"] == "anthropic":
                        cached = usage.get("cache_read_input_tokens", 0)
                        created = usage.get("cache_creation_input_tokens", 0)
                        inputs += usage["input_tokens"] + cached + created
                        reads += cached
                        writes += created
                methods[method] = {
                    "total_usd": sum(e["estimated_usd"] for e in events),
                    "jev_usd": sum(e["estimated_usd"] for e in events if e["provider"] == "jev"),
                    "artifacts": len(turns),
                    "passed_artifacts": sum(t["grade"]["artifact_passed"] for t in turns),
                    "passed_tests": sum(t["grade"]["passed_tests"] for t in turns),
                    "total_tests": sum(t["grade"]["total_tests"] for t in turns),
                    "answer_input_tokens": inputs,
                    "answer_cache_read_tokens": reads,
                    "answer_cache_write_tokens": writes,
                    "answer_cache_hit_fraction": reads / inputs if inputs else None,
                }
            scopes[scope] = methods
        for method in ["full", "jev", "summarization"]:
            assert math.isclose(
                scopes["total"][method]["total_usd"],
                scopes["initial"][method]["total_usd"]
                + scopes["continuation"][method]["total_usd"],
                rel_tol=1e-12,
            )
        report["models"][model] = scopes
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "experiments/recorded/end-to-end")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "experiments/generated/warm-cache.json"
    )
    args = parser.parse_args()
    validate_output(args.output)
    report = build_report(args.runs)
    write_json(args.output, report)
    print(json.dumps({m: d["continuation"] for m, d in report["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
