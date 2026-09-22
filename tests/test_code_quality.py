from benchmarks.code_quality import grade

RECORD = {
    "service": "example",
    "lease_seconds": 100,
    "max_payload_bytes": 500,
    "owner_team": "team-a",
    "release_tag": "release-b",
}


def policy(lease_comparison=">=", payload_comparison=">"):
    return f"""def evaluate_request(payload_bytes, age_seconds):
    if payload_bytes < 0 or age_seconds < 0:
        reason = "invalid"
    elif payload_bytes {payload_comparison} 500:
        reason = "too_large"
    elif age_seconds {lease_comparison} 100:
        reason = "expired"
    else:
        reason = "ok"
    return {{"accepted": reason == "ok", "reason": reason,
             "owner_team": "team-a", "release_tag": "release-b"}}
"""


def test_correct_artifact_passes_all_functional_tests():
    assert grade(policy(), RECORD)["artifact_passed"] is True


def test_correct_constants_with_wrong_boundary_logic_fail():
    result = grade(policy(lease_comparison=">", payload_comparison=">="), RECORD)
    assert result["valid_code"] is True
    assert result["artifact_passed"] is False
    assert 0 < result["passed_tests"] < 32


def test_unsafe_or_nonexecutable_artifact_fails():
    assert grade("import os\nos.system('echo unsafe')", RECORD)["valid_code"] is False
    assert grade("Here are the correct ledger values", RECORD)["artifact_passed"] is False
