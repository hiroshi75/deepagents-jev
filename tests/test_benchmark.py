import json
from unittest.mock import patch

import httpx
import pytest
from langchain_core.messages import HumanMessage

from benchmarks.anthropic_compare import model_factory
from benchmarks.zai_compare import Journal, corpus, cost, grade, question


def test_zai_cache_and_reasoning_are_accounted_once():
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "completion_tokens_details": {"reasoning_tokens": 50},
        "prompt_tokens_details": {"cached_tokens": 600},
    }
    actual, uncached = cost("zai", usage)
    assert actual == pytest.approx((400 * 1.4 + 600 * 0.26 + 100 * 4.4) / 1e6)
    assert uncached == pytest.approx((1000 * 1.4 + 100 * 4.4) / 1e6)


def test_anthropic_usage_categories_are_disjoint():
    usage = {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_creation_input_tokens": 300,
        "cache_read_input_tokens": 800,
        "cache_creation": {"ephemeral_1h_input_tokens": 200, "ephemeral_5m_input_tokens": 100},
    }
    actual, uncached = cost("anthropic", usage)
    assert actual == pytest.approx((100 * 5 + 100 * 6.25 + 200 * 10 + 800 * 0.5 + 20 * 25) / 1e6)
    assert uncached == pytest.approx((1200 * 5 + 20 * 25) / 1e6)


def test_jev_output_is_not_charged():
    assert cost("jev", {"input_tokens": 1_000_000, "output_tokens": 50_000}) == (0.042, 0.042)


def test_dataset_questions_do_not_reveal_answers_and_grade_is_exact():
    messages, records = corpus(10_000, 17)
    for record in records:
        for key, value in record.items():
            if key != "service":
                assert str(value) not in question(record)
        expected = {k: v for k, v in record.items() if k != "service"}
        assert grade(json.dumps(expected), record)["correct_fields"] == 4
        assert grade('{"lease_seconds": null}', record)["correct_fields"] == 0
        assert any(record["release_tag"] in m.text for m in messages)


def test_native_opus_adapter_preserves_cache_and_thinking(tmp_path):
    journal = Journal(tmp_path, "test")
    model = model_factory(journal, "fake-secret")
    body = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-8",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 3},
    }
    response = httpx.Response(200, json=body)
    with patch("httpx.Client.post", return_value=response) as post:
        result = model.invoke([HumanMessage("test")], cache_control={"type": "ephemeral"})
    assert result.text == "ok"
    request = post.call_args.kwargs["json"]
    assert request["cache_control"]["type"] == "ephemeral"
    assert request["thinking"]["type"] == "adaptive"
    assert request["output_config"]["effort"] == "low"
    raw = next(tmp_path.glob("call_*.json")).read_text()
    assert "fake-secret" not in raw
    assert journal.events[0]["provider"] == "anthropic"


def test_resume_replays_without_double_accounting_and_checks_input(tmp_path):
    original = Journal(tmp_path, "summary")
    request = {"messages": ["Read /conversation_history/session_aabb.md"]}
    response = {"model": "test", "usage": {"input_tokens": 100, "output_tokens": 3}}
    original.record("anthropic", "summary", request, response, 1.25)
    resumed = Journal(tmp_path, "summary", resume=True)
    assert (
        resumed.replay_response(
            "anthropic", "summary", {"messages": ["Read /conversation_history/session_ccdd.md"]}
        )
        == response
    )
    assert len(resumed.events) == 1 and resumed.replayed_seconds == 1.25
    mismatched = Journal(tmp_path, "summary", resume=True)
    with pytest.raises(RuntimeError, match="differs"):
        mismatched.replay_response("anthropic", "summary", {"messages": ["changed task"]})
