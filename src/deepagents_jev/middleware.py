"""Non-destructive, query-aware selection of original conversation fragments."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from deepagents_jev.client import Fragment, JevClient, JevError

TokenCounter = Callable[[Sequence[AnyMessage]], int]


class RelevanceScorer(Protocol):
    """Injectable synchronous/asynchronous relevance provider."""

    def score(self, query: str, fragments: Sequence[Fragment]) -> dict[str, float]: ...

    async def ascore(self, query: str, fragments: Sequence[Fragment]) -> dict[str, float]: ...


class ContextBudgetExceeded(RuntimeError):
    """Protected context alone exceeds the configured message budget."""


@dataclass(frozen=True)
class SelectionConfig:
    """Selection limits for history messages (system prompt/tools need extra headroom).

    All user/system messages and at least `keep_recent` trailing messages are pinned.
    Selection starts at `trigger_tokens`; query/fragment size guards protect JEV.
    A Noul threshold of 0.2 discards only relatively clear negative judgments.
    """

    max_tokens: int = 16_000
    trigger_tokens: int = 4_000
    keep_recent: int = 6
    relevance_threshold: float = 0.2
    min_savings_ratio: float = 0.15
    query_bytes: int = 6_000
    fragment_bytes: int = 12_000
    fail_open: bool = False

    def __post_init__(self) -> None:
        if self.max_tokens <= 0 or not 0 <= self.trigger_tokens <= self.max_tokens:
            raise ValueError("Require 0 <= trigger_tokens <= max_tokens and max_tokens > 0.")
        if self.keep_recent < 1 or self.query_bytes < 1 or self.fragment_bytes < 1:
            raise ValueError("keep_recent and byte limits must be positive.")
        if not 0 <= self.relevance_threshold <= 1 or not 0 <= self.min_savings_ratio <= 1:
            raise ValueError("Threshold and savings ratio must be in [0, 1].")


@dataclass(frozen=True)
class SelectionReport:
    """Payload-free diagnostics; no credentials or conversation text."""

    before_tokens: int
    after_tokens: int
    kept_fragments: tuple[str, ...]
    omitted_fragments: tuple[str, ...]
    scores: tuple[tuple[str, float], ...]
    reason: str


@dataclass(frozen=True)
class _Group:
    id: str
    indices: tuple[int, ...]
    text: str
    pinned: bool


@dataclass(frozen=True)
class _Plan:
    messages: list[AnyMessage]
    groups: list[_Group]
    query: str
    before: int
    bypass: str | None = None


def _count(messages: Sequence[AnyMessage]) -> int:
    return int(count_tokens_approximately(list(messages)))


def _text(message: AnyMessage) -> str:
    """Render only model-visible text and calls; never send artifacts or media."""
    result = f"{message.type}: {message.text}"
    if isinstance(message, AIMessage) and message.tool_calls:
        result += "\ntool_calls: " + json.dumps(message.tool_calls, ensure_ascii=False)
    if isinstance(message, ToolMessage):
        result = f"tool {message.name or ''} call_id={message.tool_call_id}: {message.text}"
    return result


def _opaque(message: AnyMessage) -> bool:
    # Signed reasoning and non-text blocks are kept intact and out of the evaluator.
    return not isinstance(message.content, str) and any(
        not isinstance(block, str) and block.get("type") != "text" for block in message.content
    )


def _group_indices(messages: list[AnyMessage]) -> list[tuple[int, ...]]:
    """Treat one assistant call and all contiguous tool results as one unit."""
    groups: list[tuple[int, ...]] = []
    i = 0
    while i < len(messages):
        end = i + 1
        if isinstance(messages[i], AIMessage):
            while end < len(messages) and isinstance(messages[end], ToolMessage):
                end += 1
        groups.append(tuple(range(i, end)))
        i = end
    return groups


def _incomplete(messages: list[AnyMessage]) -> bool:
    first = messages[0]
    if isinstance(first, ToolMessage):
        return True
    if not isinstance(first, AIMessage):
        return False
    expected = [call["id"] for call in first.tool_calls]
    actual = [m.tool_call_id for m in messages[1:] if isinstance(m, ToolMessage)]
    return (
        bool(first.invalid_tool_calls)
        or len(expected) != len(set(expected))
        or set(expected) != set(actual)
        or len(expected) != len(actual)
    )


class JevContextMiddleware(AgentMiddleware):
    """Replace Deep Agents compaction with reversible relevance selection.

    This middleware deliberately takes the existing `SummarizationMiddleware`
    slot. It never emits RemoveMessage, summaries, or state updates. Only the
    request passed to the model is changed; checkpoints keep every original.

    Pass the same middleware explicitly to custom declarative subagents. Deep
    Agents inherits this replacement automatically for the default general-purpose
    subagent and forked subagents. Compiled/remote agents own their own setup.
    """

    @property
    def name(self) -> str:
        """Use Deep Agents' supported same-name replacement mechanism."""
        return "SummarizationMiddleware"

    def __init__(
        self,
        client: RelevanceScorer | None = None,
        *,
        config: SelectionConfig | None = None,
        token_counter: TokenCounter = _count,
        on_selection: Callable[[SelectionReport], None] | None = None,
    ) -> None:
        self.client = client if client is not None else JevClient()
        self.config = config or SelectionConfig()
        self._count = token_counter
        self._on_selection = on_selection

    def _groups(self, messages: list[AnyMessage]) -> list[_Group]:
        groups = []
        recent_start = len(messages) - self.config.keep_recent
        for indices in _group_indices(messages):
            unit = [messages[i] for i in indices]
            text = "\n\n".join(_text(m) for m in unit)
            pinned = (
                indices[-1] >= recent_start
                or any(isinstance(m, (HumanMessage, SystemMessage)) or _opaque(m) for m in unit)
                or _incomplete(unit)
                or len(text.encode()) > self.config.fragment_bytes
            )
            groups.append(_Group(f"fragment_{indices[0]}", indices, text, pinned))
        return groups

    def _query(self, messages: list[AnyMessage]) -> str:
        # Re-evaluate when either the user's task or the current reasoning step changes.
        user_index = next(
            (i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)),
            -1,
        )
        latest_user = messages[user_index] if user_index >= 0 else None
        latest_ai = next(
            (m for m in reversed(messages[user_index + 1 :]) if isinstance(m, AIMessage)), None
        )
        query = (
            "The latest request defines the active task. Earlier user requests supply "
            "constraints and references for follow-ups; completed or superseded tasks "
            "are not active merely because they appear in that history.\n"
            "Earlier user requests:\n"
            + "\n".join(
                _text(m) for m in messages[: max(0, user_index)] if isinstance(m, HumanMessage)
            )
            + "\nCurrent user request:\n"
            + (_text(latest_user) if latest_user else "")
        )
        if latest_ai is not None:
            query += "\nCurrent assistant step:\n" + _text(latest_ai)
        return query

    def _prepare(self, messages: list[AnyMessage]) -> _Plan:
        before = self._count(messages)
        groups = self._groups(messages)
        query = self._query(messages)
        if before <= self.config.trigger_tokens:
            return _Plan(messages, groups, query, before, "below_trigger")
        protected = [messages[i] for g in groups if g.pinned for i in g.indices]
        if self._count(protected) > self.config.max_tokens:
            raise ContextBudgetExceeded(
                "Protected messages exceed max_tokens; increase the budget or reduce tool output. "
                "No conversation state has been removed."
            )
        if all(g.pinned for g in groups):
            return _Plan(messages, groups, query, before, "no_candidates")
        if len(query.encode()) > self.config.query_bytes:
            raise JevError(
                "Current task exceeds query_bytes; no truncated relevance query was sent."
            )
        return _Plan(messages, groups, query, before)

    def _report(
        self,
        plan: _Plan,
        kept: set[str],
        messages: list[AnyMessage],
        scores: dict[str, float],
        reason: str,
    ) -> None:
        if self._on_selection is not None:
            self._on_selection(
                SelectionReport(
                    before_tokens=plan.before,
                    after_tokens=self._count(messages),
                    kept_fragments=tuple(g.id for g in plan.groups if g.id in kept),
                    omitted_fragments=tuple(g.id for g in plan.groups if g.id not in kept),
                    scores=tuple(scores.items()),
                    reason=reason,
                )
            )

    def _finish(self, plan: _Plan, scores: dict[str, float]) -> list[AnyMessage]:
        if plan.bypass:
            self._report(plan, {g.id for g in plan.groups}, plan.messages, {}, plan.bypass)
            return plan.messages
        kept = {g.id for g in plan.groups if g.pinned}
        candidates = [g for g in plan.groups if not g.pinned]
        # Missing/invalid scores never silently become a discard decision.
        for group in candidates:
            score = scores.get(group.id)
            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not 0 <= score <= 1
            ):
                raise JevError("Missing or invalid relevance score.")
        # Highest relevance first; ties favor recent evidence. Output stays chronological.
        candidates.sort(key=lambda g: (scores[g.id], g.indices[0]), reverse=True)
        for group in candidates:
            if scores[group.id] < self.config.relevance_threshold:
                continue
            tentative = kept | {group.id}
            selected = self._materialize(plan, tentative)
            if self._count(selected) <= self.config.max_tokens:
                kept = tentative
        selected = self._materialize(plan, kept)
        savings = (plan.before - self._count(selected)) / max(1, plan.before)
        if plan.before <= self.config.max_tokens and savings < self.config.min_savings_ratio:
            kept = {g.id for g in plan.groups}
            selected = plan.messages
            reason = "preserve_prefix"
        else:
            reason = "selected"
        self._report(plan, kept, selected, scores, reason)
        return selected

    @staticmethod
    def _materialize(plan: _Plan, kept: set[str]) -> list[AnyMessage]:
        return [plan.messages[i] for g in plan.groups if g.id in kept for i in g.indices]

    def _fallback(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        # Opt-in fail-open keeps all originals, possibly above the token target.
        plan = _Plan(messages, self._groups(messages), "", self._count(messages))
        self._report(plan, {g.id for g in plan.groups}, messages, {}, "jev_error_fail_open")
        return messages

    def select(self, messages: Sequence[AnyMessage]) -> list[AnyMessage]:
        """Select original messages without altering the input or persistent state."""
        original = list(messages)
        try:
            plan = self._prepare(original)
            fragments = [Fragment(g.id, g.text) for g in plan.groups if not g.pinned]
            scores = (
                {} if plan.bypass or not fragments else self.client.score(plan.query, fragments)
            )
            return self._finish(plan, scores)
        except JevError:
            if not self.config.fail_open:
                raise
            return self._fallback(original)

    async def aselect(self, messages: Sequence[AnyMessage]) -> list[AnyMessage]:
        """Asynchronously select original messages without changing persistent state."""
        original = list(messages)
        try:
            plan = self._prepare(original)
            fragments = [Fragment(g.id, g.text) for g in plan.groups if not g.pinned]
            scores = (
                {}
                if plan.bypass or not fragments
                else await self.client.ascore(plan.query, fragments)
            )
            return self._finish(plan, scores)
        except JevError:
            if not self.config.fail_open:
                raise
            return self._fallback(original)

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelCallResult:
        """Filter only the model request, leaving the full graph history untouched."""
        return handler(request.override(messages=self.select(request.messages)))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Filter async model requests, preserving native async I/O and cancellation."""
        return await handler(request.override(messages=await self.aselect(request.messages)))
