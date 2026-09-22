"""Run a Deep Agent with JEV context selection and an explicitly chosen model."""

import argparse

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver

from deepagents_jev import JevContextMiddleware


def main() -> None:
    """Execute a task; model-provider authentication is separate from .jev."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="LangChain provider:model identifier")
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    agent = create_deep_agent(
        model=args.model,
        middleware=[JevContextMiddleware(on_selection=print)],
        checkpointer=InMemorySaver(),
    )
    result = agent.invoke(
        {"messages": [("user", args.task)]},
        {"configurable": {"thread_id": "jev-example"}},
    )
    print(result["messages"][-1].text)


if __name__ == "__main__":
    main()
