"""Grade final Python artifacts by executing functional tests in a restricted worker."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path


def question(record: dict) -> str:
    return (
        f"Implement a request-admission policy for service {record['service']} using its "
        "release ledger in the supplied history. Return only Python source defining "
        "evaluate_request(payload_bytes, age_seconds). Both arguments are integers. "
        "Return exactly four keys: accepted (bool), reason (str), owner_team, release_tag. "
        "Use the ledger's exact owner_team and release_tag on every result. "
        "Rules, in priority order: if either argument is negative, reject with reason 'invalid'; "
        "if payload_bytes exceeds max_payload_bytes, reject with 'too_large'; "
        "if age_seconds is greater than or equal to lease_seconds, reject with 'expired'; "
        "otherwise accept with 'ok'. Equal payload size is allowed. "
        "Use ordinary assignments, comparisons, if/elif/else, and dictionary returns. "
        "Do not use imports, loops, helper functions, function calls, or external state. "
        "No tools or explanation. The code must be executable and correct at boundary values."
    )


def cases(record: dict) -> list[dict]:
    limit, lease = record["max_payload_bytes"], record["lease_seconds"]
    inputs = [
        (0, 0),
        (limit, lease - 1),
        (limit + 1, 0),
        (limit, lease),
        (-1, 0),
        (0, -1),
        (limit + 1, lease),
        (limit + 1, -1),
        (-1, lease),
        (limit, lease + 1),
        (limit // 2, lease // 2),
        (1, lease - 1),
    ]
    rng = random.Random(limit + lease)
    inputs.extend((rng.randrange(-5, limit * 2), rng.randrange(-5, lease * 2)) for _ in range(20))
    result = []
    for payload, age in inputs:
        reason = (
            "invalid"
            if min(payload, age) < 0
            else "too_large"
            if payload > limit
            else "expired"
            if age >= lease
            else "ok"
        )
        result.append(
            {
                "input": [payload, age],
                "expected": {
                    "accepted": reason == "ok",
                    "reason": reason,
                    "owner_team": record["owner_team"],
                    "release_tag": record["release_tag"],
                },
            }
        )
    return result


def grade(answer: str, record: dict) -> dict:
    tests = cases(record)
    try:
        completed = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).with_name("code_worker.py"))],
            input=json.dumps({"code": answer, "tests": tests}),
            text=True,
            capture_output=True,
            timeout=5,
            env={"PATH": "/usr/bin:/bin", "PYTHONIOENCODING": "utf-8"},
        )
        result = (
            json.loads(completed.stdout)
            if completed.returncode == 0
            else {
                "valid_code": False,
                "error": "Worker terminated",
                "tests": [],
            }
        )
    except (subprocess.TimeoutExpired, ValueError):
        result = {"valid_code": False, "error": "Worker timeout or invalid output", "tests": []}
    passed = sum(test["passed"] for test in result["tests"])
    return {
        **result,
        "correct_fields": passed,
        "total_fields": len(tests),
        "passed_tests": passed,
        "total_tests": len(tests),
        "artifact_passed": passed == len(tests),
    }
