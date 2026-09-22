from benchmarks.zai_compare import response_cost


def test_preoutput_refusal_is_not_billed():
    body = {
        "stop_reason": "refusal",
        "content": [],
        "usage": {
            "input_tokens": 50,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 200,
            "output_tokens": 0,
        },
    }
    assert response_cost("anthropic", body) == (0, 0)


def test_midoutput_refusal_retains_billing():
    body = {
        "stop_reason": "refusal",
        "content": [],
        "usage": {"input_tokens": 50, "output_tokens": 10},
    }
    assert response_cost("anthropic", body) == (0.0005, 0.0005)


def test_successful_cache_usage_is_billed():
    body = {
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 50,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 200,
            "output_tokens": 10,
        },
    }
    assert response_cost("anthropic", body) == (0.0018, 0.002)
