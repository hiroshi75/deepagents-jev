"""Exercise the real Deep Agents graph with deterministic chat/JEV models."""

from collections.abc import Sequence
from typing import Any

import pytest
from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from deepagents_jev import JevContextMiddleware, SelectionConfig


class RecordingModel(BaseChatModel):
    replies: list[AIMessage] = Field(default_factory=lambda: [AIMessage("Finished.")])
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "test-recording-model"

    def bind_tools(self, tools: Sequence, **kwargs: Any):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


class DropOld:
    def score(self, query, fragments):
        return {f.id: 0.01 for f in fragments}

    async def ascore(self, query, fragments):
        return self.score(query, fragments)


def selector(reports):
    return JevContextMiddleware(
        DropOld(),
        on_selection=reports.append,
        config=SelectionConfig(
            max_tokens=2000,
            trigger_tokens=0,
            keep_recent=1,
            min_savings_ratio=0,
        ),
    )


def history():
    # Long enough to trigger the normal summarizer with this model's context profile.
    return [
        HumanMessage("Keep my original instructions."),
        *[AIMessage(f"OLD_EVIDENCE_{i}: " + "obsolete " * 600) for i in range(8)],
        HumanMessage("Do the current task."),
    ]


@pytest.mark.parametrize("async_mode", [False, True])
async def test_real_graph_replaces_summarizer_and_preserves_checkpoint(async_mode):
    reports = []
    model = RecordingModel(profile={"max_input_tokens": 1000})
    agent = create_deep_agent(
        model=model,
        middleware=[selector(reports)],
        checkpointer=InMemorySaver(),
    )
    original = history()
    config = {"configurable": {"thread_id": "retained-history"}}
    if async_mode:
        result = await agent.ainvoke({"messages": original}, config)
        saved = await agent.aget_state(config)
    else:
        result = agent.invoke({"messages": original}, config)
        saved = agent.get_state(config)
    assert len(model.seen) == 1  # No extra LLM call for a generated summary.
    assert all("OLD_EVIDENCE" not in m.text for m in model.seen[0])
    assert [m.content for m in result["messages"][:-1]] == [m.content for m in original]
    assert len(saved.values["messages"]) == len(original) + 1
    assert reports[0].omitted_fragments
    assert all(not key.startswith("_summarization") for key in saved.values)


def test_tool_loop_keeps_tool_results_and_full_state():
    reports = []

    def get_fact() -> str:
        """Return a fact needed for the task."""
        return "The answer is 42."

    model = RecordingModel(
        replies=[
            AIMessage("", tool_calls=[{"id": "fact", "name": "get_fact", "args": {}}]),
            AIMessage("42"),
        ]
    )
    agent = create_deep_agent(model=model, tools=[get_fact], middleware=[selector(reports)])
    result = agent.invoke({"messages": history()})
    assert len(model.seen) == 2
    assert any(m.type == "tool" and "42" in m.text for m in model.seen[-1])
    assert any("OLD_EVIDENCE" in m.text for m in result["messages"])
    assert result["messages"][-1].text == "42"


def test_default_subagent_inherits_selection_replacement():
    reports = []
    model = RecordingModel(
        replies=[
            AIMessage(
                "",
                tool_calls=[
                    {
                        "id": "task1",
                        "name": "task",
                        "args": {
                            "subagent_type": "general-purpose",
                            "description": "Find the answer.",
                        },
                    }
                ],
            ),
            AIMessage("Subagent answer."),
            AIMessage("Final answer."),
        ]
    )
    agent = create_deep_agent(model=model, middleware=[selector(reports)])
    result = agent.invoke({"messages": history()})
    assert len(model.seen) == 3
    assert len(reports) == 3  # The middle report is produced inside the subagent graph.
    assert result["messages"][-1].text == "Final answer."


def test_custom_subagent_with_explicit_middleware():
    parent_reports, child_reports = [], []
    child_model = RecordingModel()
    parent_model = RecordingModel(
        replies=[
            AIMessage(
                "",
                tool_calls=[
                    {
                        "id": "task1",
                        "name": "task",
                        "args": {"subagent_type": "researcher", "description": "Find the answer."},
                    }
                ],
            ),
            AIMessage("Final answer."),
        ]
    )
    agent = create_deep_agent(
        model=parent_model,
        middleware=[selector(parent_reports)],
        subagents=[
            {
                "name": "researcher",
                "description": "Research facts.",
                "system_prompt": "Research the requested topic.",
                "model": child_model,
                "middleware": [selector(child_reports)],
            }
        ],
    )
    agent.invoke({"messages": history()})
    assert len(child_reports) == 1 and len(parent_reports) == 2


async def test_async_stream_retains_full_state():
    model = RecordingModel()
    agent = create_deep_agent(model=model, middleware=[selector([])])
    chunks = [chunk async for chunk in agent.astream({"messages": history()}, stream_mode="values")]
    assert chunks[-1]["messages"][-1].text == "Finished."
    assert any("OLD_EVIDENCE" in m.text for m in chunks[-1]["messages"])
    assert all("OLD_EVIDENCE" not in m.text for m in model.seen[0])
