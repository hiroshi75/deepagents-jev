from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deepagents_jev import (
    ContextBudgetExceeded,
    JevContextMiddleware,
    JevError,
    SelectionConfig,
)


class Scorer:
    def __init__(self, score=0.01):
        self.value = score
        self.queries = []
        self.fragments = []

    def score(self, query, fragments):
        self.queries.append(query)
        self.fragments.extend(fragments)
        return {f.id: self.value for f in fragments}

    async def ascore(self, query, fragments):
        return self.score(query, fragments)


def configured(scorer=None, **kwargs):
    return JevContextMiddleware(
        client=scorer or Scorer(),
        config=SelectionConfig(trigger_tokens=0, keep_recent=1, min_savings_ratio=0, **kwargs),
    )


def exchange():
    return [
        HumanMessage("Work on this repository. Never publish it."),
        AIMessage(
            "Checking auth",
            tool_calls=[
                {"id": "read1", "name": "read_file", "args": {"file_path": "auth.py"}},
                {"id": "read2", "name": "read_file", "args": {"file_path": "test_auth.py"}},
            ],
        ),
        ToolMessage("Token expiry is unchecked.", tool_call_id="read1"),
        ToolMessage("There are no expiration tests.", tool_call_id="read2"),
        AIMessage("CSS background is blue."),
        HumanMessage("Fix token expiration."),
    ]


def test_removal_is_request_only_and_tool_exchange_is_atomic():
    original = exchange()
    snapshot = deepcopy(original)
    result = configured().select(original)
    assert result == [original[0], original[-1]]
    assert original == snapshot
    assert result[0] is original[0]
    assert result[-1] is original[-1]


def test_task_change_restores_original_evidence():
    class TopicScorer(Scorer):
        def score(self, query, fragments):
            topic = "CSS" if "Now fix CSS" in query else "Token"
            return {f.id: 0.99 if topic in f.text else 0.01 for f in fragments}

    original = exchange()
    middleware = configured(TopicScorer())
    first = middleware.select(original)
    assert original[1] in first and original[4] not in first
    next_history = [*original, AIMessage("Fixed expiration."), HumanMessage("Now fix CSS.")]
    second = middleware.select(next_history)
    assert original[4] in second and original[1] not in second
    assert second[0] == original[0]


async def test_sync_async_parity():
    middleware = configured()
    original = exchange()
    assert await middleware.aselect(original) == middleware.select(original)


def test_recent_boundary_keeps_entire_multitool_exchange():
    original = exchange()[:4]
    assert configured().select(original) == original


@pytest.mark.parametrize(
    "tail",
    [
        [AIMessage("pending", tool_calls=[{"id": "p", "name": "f", "args": {}}])],
        [ToolMessage("orphan", tool_call_id="missing")],
        [AIMessage(content=[{"type": "image_url", "image_url": {"url": "private"}}])],
        [AIMessage(content=[{"type": "thinking", "thinking": "signed", "signature": "sig"}])],
        [SystemMessage("Always obey this.")],
    ],
)
def test_opaque_incomplete_and_system_context_is_pinned(tail):
    scorer = Scorer()
    original = [HumanMessage("Task"), *tail, HumanMessage("Next")]
    assert configured(scorer).select(original) == original
    assert scorer.fragments == []


def test_token_budget_packs_by_relevance_and_keeps_order():
    class Ranked(Scorer):
        def score(self, query, fragments):
            return {f.id: 0.99 if "important" in f.text else 0.8 for f in fragments}

    history = [
        HumanMessage("task"),
        AIMessage("unrelated"),
        AIMessage("important"),
        HumanMessage("next"),
    ]
    middleware = JevContextMiddleware(
        Ranked(),
        token_counter=len,
        config=SelectionConfig(max_tokens=3, trigger_tokens=0, keep_recent=1),
    )
    assert middleware.select(history) == [history[0], history[2], history[3]]


def test_protected_budget_overflow_does_not_drop_user_instructions():
    middleware = JevContextMiddleware(
        Scorer(),
        token_counter=len,
        config=SelectionConfig(max_tokens=1, trigger_tokens=0, keep_recent=1),
    )
    with pytest.raises(ContextBudgetExceeded):
        middleware.select([HumanMessage("constraint"), HumanMessage("latest")])


def test_below_trigger_avoids_jev_and_reports():
    reports = []
    scorer = Scorer()
    middleware = JevContextMiddleware(scorer, on_selection=reports.append)
    history = exchange()
    assert middleware.select(history) == history
    assert scorer.queries == []
    assert reports[0].reason == "below_trigger"


def test_small_savings_preserve_prompt_prefix():
    reports = []
    middleware = JevContextMiddleware(
        Scorer(),
        token_counter=len,
        on_selection=reports.append,
        config=SelectionConfig(trigger_tokens=0, keep_recent=1, min_savings_ratio=0.9),
    )
    history = exchange()
    assert middleware.select(history) == history
    assert reports[-1].reason == "preserve_prefix"


@pytest.mark.parametrize("fail_open", [False, True])
async def test_service_failure_policy(fail_open):
    class Broken(Scorer):
        def score(self, query, fragments):
            raise JevError("unavailable")

    middleware = configured(Broken(), fail_open=fail_open)
    history = exchange()
    if fail_open:
        assert middleware.select(history) == history
        assert await middleware.aselect(history) == history
    else:
        with pytest.raises(JevError):
            middleware.select(history)
        with pytest.raises(JevError):
            await middleware.aselect(history)


def test_empty_history():
    assert configured().select([]) == []


def test_tool_artifacts_and_usage_metadata_not_sent_to_jev():
    scorer = Scorer()
    history = exchange()
    history[2] = history[2].model_copy(update={"artifact": "PRIVATE_ARTIFACT"})
    history[1] = history[1].model_copy(update={"response_metadata": {"private": "PRIVATE_META"}})
    configured(scorer).select(history)
    payload = str(scorer.fragments)
    assert "PRIVATE_ARTIFACT" not in payload and "PRIVATE_META" not in payload


def test_follow_up_query_includes_earlier_user_task():
    scorer = Scorer()
    middleware = configured(scorer)
    middleware.select(
        [
            HumanMessage("Investigate expired JWT acceptance."),
            AIMessage("Some old evidence."),
            HumanMessage("Please implement the fix."),
        ]
    )
    assert "expired JWT" in scorer.queries[0]
    assert "Please implement the fix" in scorer.queries[0]


def test_new_task_does_not_use_stale_assistant_step():
    scorer = Scorer()
    configured(scorer).select(
        [
            HumanMessage("Old task."),
            AIMessage("OLD_ASSISTANT_STEP"),
            HumanMessage("Unrelated new task."),
        ]
    )
    assert "OLD_ASSISTANT_STEP" not in scorer.queries[0]


def test_no_candidates_does_not_require_a_small_query():
    middleware = configured(query_bytes=1)
    assert middleware.select([HumanMessage("A long protected request")])


async def test_async_cancellation_propagates():
    import asyncio

    class Cancelled(Scorer):
        async def ascore(self, query, fragments):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await configured(Cancelled(), fail_open=True).aselect(exchange())
