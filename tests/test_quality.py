from benchmarks.quality import evaluate


def test_final_answer_must_have_correct_values_types_and_schema():
    expected = {"value": 1}
    assert evaluate('```json\n{"value": 1}\n```', expected)["fully_correct"]
    assert not evaluate('{"value": true}', expected)["fully_correct"]
    assert not evaluate('{"value": 1, "extra": 2}', expected)["fully_correct"]
    assert not evaluate('{"value": null}', expected)["fully_correct"]
    assert not evaluate("not JSON", expected)["fully_correct"]
