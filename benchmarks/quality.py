"""Independently check final JSON answers against the original dataset records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def evaluate(answer: str, expected: dict) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer.strip())
    try:
        actual = json.loads(text)
    except ValueError:
        actual = None
    parsed = isinstance(actual, dict)
    correct = {
        k: parsed and type(actual.get(k)) is type(v) and actual.get(k) == v
        for k, v in expected.items()
    }
    schema = parsed and set(actual) == set(expected)
    return {
        "expected": expected,
        "actual": actual,
        "parsed": parsed,
        "exact_schema": schema,
        "correct_fields": sum(correct.values()),
        "field_results": correct,
        "fully_correct": schema and all(correct.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    recorded = ROOT / "experiments/recorded"
    parser.add_argument("--glm-run", type=Path, default=recorded / "zai_context_comparison/run1")
    parser.add_argument(
        "--opus-run", type=Path, default=recorded / "opus48_context_comparison/run1"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/generated/quality.json")
    args = parser.parse_args()
    from benchmarks.zai_compare import DATASETS, question, validate_output

    validate_output(args.output)
    result = {}
    for model, run in [("glm-5.3", args.glm_run), ("claude-opus-4-8", args.opus_run)]:
        methods = {}
        answers = []
        for path in sorted(run.glob("tokens*/*/result.json")):
            case = path.parent.parent.name
            data = json.loads(path.read_text())
            dataset = json.loads((DATASETS / case / "dataset.json").read_text())
            records = {r["service"]: r for r in dataset["records"]}
            method = data["condition"]
            a = methods.setdefault(
                method,
                {
                    "answers": 0,
                    "correct_answers": 0,
                    "fields": 0,
                    "correct_fields": 0,
                    "parse_failures": 0,
                    "schema_failures": 0,
                    "cost_usd": 0,
                },
            )
            a["cost_usd"] += data["estimated_usd"]
            for turn in data["turns"]:
                record = records[turn["service"]]
                expected = {k: v for k, v in record.items() if k != "service"}
                grade = evaluate(turn["answer"], expected)
                answers.append(
                    {
                        "case": case,
                        "method": method,
                        "turn": turn["turn"],
                        "question": question(record),
                        "answer": turn["answer"],
                        **grade,
                    }
                )
                a["answers"] += 1
                a["correct_answers"] += grade["fully_correct"]
                a["fields"] += len(expected)
                a["correct_fields"] += grade["correct_fields"]
                a["parse_failures"] += not grade["parsed"]
                a["schema_failures"] += not grade["exact_schema"]
        for a in methods.values():
            a["cost_per_correct_answer_usd"] = (
                a["cost_usd"] / a["correct_answers"] if a["correct_answers"] else None
            )
        result[model] = {"methods": methods, "answers": answers}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({m: d["methods"] for m, d in result.items()}, indent=2))


if __name__ == "__main__":
    main()
