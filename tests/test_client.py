import json

import httpx
import pytest

from deepagents_jev import Fragment, JevClient, JevError, load_api_key


def transport_for(handler):
    return httpx.MockTransport(handler)


def answer(request, value=0.9):
    body = json.loads(request.content)
    return httpx.Response(
        200, json={"answers": {key: {"type": "noul", "noul": value} for key in body["questions"]}}
    )


def test_key_file_is_not_executed_or_interpolated(tmp_path, monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    key_file = tmp_path / ".jev"
    key_file.write_text('export JEV_API_KEY="literal-${HOME}-$(touch bad)"\n')
    assert load_api_key(key_file) == "literal-${HOME}-$(touch bad)"
    monkeypatch.setenv("JEV_API_KEY", "environment")
    assert load_api_key(key_file) == "environment"


def test_missing_credentials(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="JEV_API_KEY"):
        load_api_key(None)


def test_wire_contract_cache_and_query_invalidation():
    requests = []

    def handler(request):
        requests.append(request)
        body = json.loads(request.content)
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        assert request.headers["Authorization"] == "Bearer test-key"
        for key, question in body["questions"].items():
            assert key in question["instructions"]  # IDs alone are not visible to JEV.
            assert question["type"] == "noul"
        return answer(request)

    client = JevClient("test-key", transport=transport_for(handler))
    fragments = [Fragment("a", "content A"), Fragment("b", "content B")]
    assert client.score("auth", fragments) == {"a": 0.9, "b": 0.9}
    client.score("auth", fragments)
    assert len(requests) == 1
    client.score("CSS", fragments)
    assert len(requests) == 2
    client.score("auth", [Fragment("a", "changed content")])
    assert len(requests) == 3
    assert "test-key" not in repr(client)


def test_batch_limits_and_cache_eviction():
    sizes = []

    def handler(request):
        body = json.loads(request.content)
        sizes.append(len(body["questions"]))
        return answer(request)

    client = JevClient("key", transport=transport_for(handler), batch_size=2, cache_size=1)
    fragments = [Fragment(str(i), str(i) + "x" * 300) for i in range(5)]
    assert len(client.score("query", fragments)) == 5
    assert sizes == [2, 2, 1]
    client.score("query", [fragments[0]])
    assert len(sizes) == 4


@pytest.mark.parametrize("value", [None, True, "0.9", -0.1, 1.1])
def test_invalid_answers_fail_closed(value):
    client = JevClient("key", transport=transport_for(lambda r: answer(r, value)))
    with pytest.raises(JevError, match="invalid"):
        client.score("q", [Fragment("x", "text")])


@pytest.mark.parametrize("body", [{}, {"answers": {}}, {"answers": {"x": {"noul": 0.9}}}])
def test_incomplete_responses(body):
    client = JevClient("key", transport=transport_for(lambda r: httpx.Response(200, json=body)))
    with pytest.raises(JevError):
        client.score("q", [Fragment("x", "text")])


def test_errors_never_contain_server_body_or_secret():
    client = JevClient(
        "secret",
        transport=transport_for(lambda r: httpx.Response(401, text="secret request echoed here")),
    )
    with pytest.raises(JevError, match="HTTP 401") as caught:
        client.score("private text", [Fragment("x", "private fragment")])
    assert "secret" not in str(caught.value)
    assert "private" not in str(caught.value)


async def test_async_retry_and_cache():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return answer(request)

    client = JevClient("key", transport=transport_for(handler))
    assert await client.ascore("q", [Fragment("x", "text")]) == {"x": 0.9}
    assert client.score("q", [Fragment("x", "text")]) == {"x": 0.9}
    assert calls == 2


def test_timeout_is_sanitized():
    def handler(request):
        raise httpx.ReadTimeout("secret", request=request)

    client = JevClient("secret", transport=transport_for(handler))
    with pytest.raises(JevError, match="transport failure"):
        client.score("q", [Fragment("x", "text")])


def test_oversized_request_never_sent():
    client = JevClient("key", transport=transport_for(lambda r: pytest.fail("Unexpected HTTP")))
    with pytest.raises(JevError, match="byte limit"):
        client.score("q", [Fragment("x", "日" * 24000)])
