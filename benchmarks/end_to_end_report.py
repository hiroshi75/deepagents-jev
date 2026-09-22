"""Re-execute saved final artifacts and account for successful and failed attempts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from benchmarks.code_quality import grade, question
from benchmarks.report import summarize
from benchmarks.zai_compare import DATASETS, ROOT, validate_output, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "experiments/recorded/end-to-end")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "experiments/generated/final-artifacts.json"
    )
    parser.add_argument("--verify", action="store_true", help="Compare with published evidence")
    args = parser.parse_args()
    validate_output(args.output)
    report = {
        "task": "Generate a Python request-admission policy from historical ledger specifications",
        "tests_per_artifact": 32,
        "models": {},
    }
    for model in ["glm-5.3", "claude-opus-4-8"]:
        data = summarize(args.runs / model)
        if any(len(data["cases"].get(c, [])) != 4 for c in ["summarization", "jev", "full"]):
            raise ValueError("Incomplete final-artifact experiment")
        artifacts = []
        methods = {}
        for method, cases in data["cases"].items():
            outcomes = []
            for case in cases:
                source = json.loads((DATASETS / case["case"] / "dataset.json").read_text())
                records = {r["service"]: r for r in source["records"]}
                for turn in case["turns"]:
                    record = records[turn["service"]]
                    result = grade(turn["answer"], record)
                    if result["passed_tests"] != turn["grade"]["passed_tests"]:
                        raise ValueError("Re-execution does not match the recorded grade")
                    outcomes.append(result)
                    artifacts.append(
                        {
                            "case": case["case"],
                            "method": method,
                            "turn": turn["turn"],
                            "service": turn["service"],
                            "question": question(record),
                            "code": turn["answer"],
                            **result,
                        }
                    )
            cost = data["aggregate"][method]
            successes = sum(r["artifact_passed"] for r in outcomes)
            methods[method] = {
                "artifacts": len(outcomes),
                "passed_artifacts": successes,
                "passed_tests": sum(r["passed_tests"] for r in outcomes),
                "total_tests": sum(r["total_tests"] for r in outcomes),
                "invalid_artifacts": sum(not r["valid_code"] for r in outcomes),
                "total_usd": cost["estimated_usd"],
                "llm_usd": cost["llm_usd"],
                "jev_usd": cost["jev_usd"],
                "refusals": cost["refusals"],
                "elapsed_seconds": cost["total_seconds"],
                "cost_per_passing_artifact_usd": cost["estimated_usd"] / successes
                if successes
                else None,
            }
        report["models"][model] = {"methods": methods, "artifacts": artifacts}
    if args.verify:
        reference = json.loads((ROOT / "experiments/summary/final-artifacts.json").read_text())
        # Python 3.12 changed float summation. Preserve exact artifact checks while
        # allowing sub-cent rounding differences in usage-based cost aggregation.
        for model, actual in report["models"].items():
            expected = reference["models"][model]
            if actual["artifacts"] != expected["artifacts"]:
                raise ValueError(f"Final-artifact execution differs from publication: {model}")
            for method, metrics in actual["methods"].items():
                for key, value in metrics.items():
                    published = expected["methods"][method][key]
                    equal = (
                        math.isclose(value, published, rel_tol=1e-12, abs_tol=1e-12)
                        if isinstance(value, float) and isinstance(published, float)
                        else value == published
                    )
                    if not equal:
                        raise ValueError(f"Published metric differs: {model}/{method}/{key}")
    write_json(args.output, report)
    print(json.dumps({m: d["methods"] for m, d in report["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
