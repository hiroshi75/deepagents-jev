"""Use real JEV with synthetic history; requires only the existing .jev key."""

from dataclasses import asdict
from json import dumps

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepagents_jev import JevContextMiddleware, SelectionConfig


def main() -> None:
    """Demonstrate dropping irrelevant evidence and restoring it after a task change."""
    reports = []
    middleware = JevContextMiddleware(
        config=SelectionConfig(trigger_tokens=0, keep_recent=1, min_savings_ratio=0),
        on_selection=reports.append,
    )
    auth_call = AIMessage(
        "Read authentication code",
        tool_calls=[
            {
                "id": "auth",
                "name": "read_file",
                "args": {"file_path": "auth.py"},
            }
        ],
    )
    auth_result = ToolMessage(
        "auth.py: verify_token decodes JWTs with verify_exp=False. "
        "Expired authentication tokens are incorrectly accepted.",
        tool_call_id="auth",
    )
    css_call = AIMessage(
        "Read CSS",
        tool_calls=[
            {
                "id": "css",
                "name": "read_file",
                "args": {"file_path": "landing.css"},
            }
        ],
    )
    css_result = ToolMessage(
        "landing.css: .hero { background: blue; color: navy; } "
        "The hero heading has poor contrast and is unreadable on the blue background.",
        tool_call_id="css",
    )
    history = [
        HumanMessage("Maintain the web app. Do not deploy changes."),
        auth_call,
        auth_result,
        css_call,
        css_result,
        HumanMessage("Fix the authentication bug: expired tokens are accepted."),
    ]
    selected_auth = middleware.select(history)
    print("Authentication task:", dumps(asdict(reports[-1]), indent=2))
    # Reuse the full original history, exactly as a LangGraph checkpoint does.
    next_history = [
        *history,
        AIMessage("Authentication fixed."),
        HumanMessage("Now fix only the hero heading contrast in landing.css."),
    ]
    selected_css = middleware.select(next_history)
    print("CSS task:", dumps(asdict(reports[-1]), indent=2))
    print(
        dumps(
            {
                "auth_evidence_kept": auth_result in selected_auth,
                "css_evidence_initially_omitted": css_result not in selected_auth,
                "css_evidence_restored": css_result in selected_css,
                "original_history_intact": auth_result in history and css_result in history,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
