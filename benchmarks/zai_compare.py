"""Run real GLM and JEV calls on identical long, synthetic coding histories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import deepagents
import httpx
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.summarization import SummarizationMiddleware
from dotenv import dotenv_values
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    convert_to_openai_messages,
    messages_from_dict,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field, SecretStr

from deepagents_jev import JevClient, JevContextMiddleware, SelectionConfig

ROOT = Path(__file__).resolve().parents[1]
RECORDED = ROOT / "experiments/recorded"
DATASETS = RECORDED / "zai_context_comparison/run1"


def read_key(name: str, filename: str) -> str:
    key = os.environ.get(name) or dotenv_values(ROOT / filename, interpolate=False).get(name)
    if not key:
        raise ValueError(f"Set {name} or add it to {filename}; live runs incur API charges.")
    return key


def validate_output(path: Path) -> None:
    if path.resolve().is_relative_to(RECORDED.resolve()):
        raise ValueError("Choose an output outside experiments/recorded; recordings are immutable.")


ZAI_URL = "https://api.z.ai/api/paas/v4/chat/completions"
PRICES = {"glm_input": 1.4, "glm_cached": 0.26, "glm_output": 4.4, "jev_input": 0.042}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def cost(provider: str, usage: dict) -> tuple[float, float]:
    """Return observed-cache and all-uncached USD estimates, including reasoning output."""
    if provider == "jev":
        amount = usage["input_tokens"] * PRICES["jev_input"] / 1_000_000
        return amount, amount
    if provider == "anthropic":
        writes = usage.get("cache_creation_input_tokens", 0)
        long_writes = usage.get("cache_creation", {}).get("ephemeral_1h_input_tokens", 0)
        short_writes = writes - long_writes
        reads = usage.get("cache_read_input_tokens", 0)
        ordinary = usage["input_tokens"]
        output = usage["output_tokens"] * 25
        return (
            (ordinary * 5 + short_writes * 6.25 + long_writes * 10 + reads * 0.5 + output)
            / 1_000_000,
            ((ordinary + writes + reads) * 5 + output) / 1_000_000,
        )
    prompt, completion = usage["prompt_tokens"], usage["completion_tokens"]
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    if not 0 <= cached <= prompt:
        raise ValueError("Invalid cached token accounting")
    output = completion * PRICES["glm_output"]
    return (
        ((prompt - cached) * PRICES["glm_input"] + cached * PRICES["glm_cached"] + output)
        / 1_000_000,
        (prompt * PRICES["glm_input"] + output) / 1_000_000,
    )


def response_cost(provider: str, response: dict) -> tuple[float, float]:
    """Apply provider billing rules; usage alone is not always billable.

    Anthropic pre-output refusals report usage but are exempt from billing.
    Source: https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback
    No fallback requests are made by this benchmark.
    """
    usage = response.get("usage", {})
    if not usage:
        return 0.0, 0.0
    if (
        provider == "anthropic"
        and response.get("stop_reason") == "refusal"
        and not response.get("content")
        and usage.get("output_tokens", 0) == 0
    ):
        return 0.0, 0.0
    return cost(provider, usage)


class Journal:
    def __init__(self, directory: Path, condition: str, *, resume: bool = False) -> None:
        self.directory, self.condition = directory, condition
        self.turn = 0
        self.events: list[dict] = []
        directory.mkdir(parents=True, exist_ok=True)
        previous = directory / "events.jsonl"
        if resume and previous.exists():
            self.events = [json.loads(line) for line in previous.read_text().splitlines()]
        self._replay_events = list(self.events)
        self.replayed_seconds = 0.0

    def replay_response(self, provider: str, stage: str, payload: dict) -> dict | None:
        """Replay completed calls to reconstruct a lost in-memory graph checkpoint.

        Only randomized archive filenames may differ; all other request fields
        must match. Replayed calls retain their original usage and are not billed twice.
        """
        if not self._replay_events:
            return None
        event = self._replay_events[0]
        raw = json.loads((self.directory / event["raw_file"]).read_text())

        def normalize(value: dict) -> str:
            return re.sub(
                r"/conversation_history/session_[a-f0-9]+\.md",
                "/conversation_history/SESSION.md",
                json.dumps(value, sort_keys=True),
            )

        if (
            event["provider"] != provider
            or event["stage"] != stage
            or normalize(raw["request"]) != normalize(payload)
        ):
            raise RuntimeError("Replay request differs from the saved request; refusing reuse")
        self._replay_events.pop(0)
        self.replayed_seconds += event["elapsed_seconds"]
        print("REPLAY_COMPLETED_CALL", provider, stage, event["call"], flush=True)
        return raw["response"]

    def record(
        self,
        provider: str,
        stage: str,
        payload: dict,
        response: dict,
        elapsed: float,
        status: int = 200,
    ) -> None:
        number = len(self.events) + 1
        raw_path = self.directory / f"call_{number:03d}_{provider}_{stage}.json"
        write_json(raw_path, {"request": payload, "response": response, "http_status": status})
        usage = response.get("usage", {})
        charged, uncached = response_cost(provider, response)
        event = {
            "call": number,
            "condition": self.condition,
            "turn": self.turn,
            "provider": provider,
            "stage": stage,
            "model": response.get("model"),
            "usage": usage,
            "elapsed_seconds": elapsed,
            "estimated_usd": charged,
            "uncached_usd": uncached,
            "raw_file": raw_path.name,
            "http_status": status,
            "stop_reason": response.get("stop_reason"),
            "stop_details": response.get("stop_details"),
        }
        self.events.append(event)
        with (self.directory / "events.jsonl").open("a") as file:
            file.write(json.dumps(event) + "\n")
        if provider != "jev":
            print(json.dumps(event), flush=True)


class MeteredTransport(httpx.BaseTransport):
    """Meter the actual existing JevClient, including every retry and batch."""

    def __init__(self, journal: Journal) -> None:
        self.journal = journal
        self.inner = httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        start = time.perf_counter()
        response = self.inner.handle_request(request)
        response.read()
        try:
            body = response.json()
        except ValueError:
            body = {"invalid_response": True}
        self.journal.record(
            "jev",
            "selection",
            json.loads(request.content),
            body,
            time.perf_counter() - start,
            response.status_code,
        )
        return response

    def close(self) -> None:
        self.inner.close()


class MeteredGLM(BaseChatModel):
    """Minimal LangChain adapter preserving Z.ai's raw usage and billing details."""

    api_key: SecretStr = Field(exclude=True, repr=False)
    journal: Any = Field(exclude=True, repr=False)
    model_name: str = "glm-5.3"
    stage: str = "answer"
    output_limit: int = 2048

    @property
    def _llm_type(self) -> str:
        return "zai-metered-benchmark"

    def bind_tools(self, tools: Sequence, **kwargs: Any):
        # Controlled context/answer test: tools have the same definitions in every arm,
        # but no external evidence retrieval or file edits are allowed in this experiment.
        return self.bind(tools=[convert_to_openai_tool(tool) for tool in tools], tool_choice="none")

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs):
        wire = convert_to_openai_messages(messages)
        payload = {
            "model": self.model_name,
            "messages": wire,
            "thinking": {"type": "enabled"},
            "reasoning_effort": "low",
            "temperature": 0,
            "max_tokens": self.output_limit,
        }
        for name in ("tools", "tool_choice"):
            if name in kwargs:
                payload[name] = kwargs[name]
        start = time.perf_counter()
        with httpx.Client(timeout=httpx.Timeout(300, connect=30)) as client:
            response = client.post(
                ZAI_URL,
                headers={"Authorization": f"Bearer {self.api_key.get_secret_value()}"},
                json=payload,
            )
        body = response.json()
        self.journal.record(
            "zai", self.stage, payload, body, time.perf_counter() - start, response.status_code
        )
        if response.status_code != 200:
            raise RuntimeError(f"Z.ai HTTP {response.status_code}; see credential-free raw record")
        choice = body["choices"][0]
        if choice["finish_reason"] != "stop":
            raise RuntimeError(f"Incomplete GLM response: {choice['finish_reason']}")
        message = AIMessage(
            content=choice["message"].get("content") or "",
            response_metadata={"usage": body["usage"], "model_name": body["model"]},
        )
        if not message.content:
            raise RuntimeError("GLM returned an empty answer")
        return ChatResult(generations=[ChatGeneration(message=message)])


class NoCompaction(AgentMiddleware):
    """Full-context control occupying the same Deep Agents replacement slot."""

    @property
    def name(self) -> str:
        return "SummarizationMiddleware"


def source_chunks(seed: int) -> list[tuple[str, str]]:
    source_root = Path(deepagents.__file__).resolve().parent
    paths = sorted(source_root.rglob("*.py"))
    random.Random(seed).shuffle(paths)
    result = []
    for path in paths:
        lines, buffer, start = path.read_text().splitlines(keepends=True), "", 1
        for index, line in enumerate(lines, 1):
            if buffer and len((buffer + line).encode()) > 6500:
                result.append(
                    (
                        f"refs/deepagents/libs/deepagents/deepagents/{path.relative_to(source_root)}"
                        f":{start}-{index - 1}",
                        buffer,
                    )
                )
                buffer, start = "", index
            buffer += line
        if buffer:
            result.append(
                (
                    f"refs/deepagents/libs/deepagents/deepagents/{path.relative_to(source_root)}:{start}-{len(lines)}",
                    buffer,
                )
            )
    return result


def corpus(target_tokens: int, seed: int) -> tuple[list[BaseMessage], list[dict]]:
    rng = random.Random(seed)
    records = []
    for service in ("orion-auth", "lyra-search", "vega-export"):
        records.append(
            {
                "service": service,
                "lease_seconds": rng.randrange(61, 300),
                "max_payload_bytes": rng.randrange(100_001, 900_000),
                "release_tag": f"build-{rng.randrange(100000, 999999)}-{service[:3]}",
                "owner_team": f"team-{rng.randrange(101, 999)}",
            }
        )
    chosen, size = [], 0
    for path, text in source_chunks(seed):
        chosen.append((path, text))
        size += len(text) / 4 + 45
        if size >= target_tokens:
            break
    if size < target_tokens:
        raise ValueError("Insufficient non-repeated source text")
    positions = {
        int(len(chosen) * fraction): record
        for fraction, record in zip((0.12, 0.48, 0.78), records, strict=True)
    }
    messages: list[BaseMessage] = [
        HumanMessage(
            "We are investigating a software workspace. The following are historical read_file "
            "results, including source files and release ledger records. Keep the ledger's exact "
            "configuration values for later questions. No deployment or edits are authorized."
        )
    ]
    for index, (path, text) in enumerate(chosen):
        entries = [(path, text)]
        if index in positions:
            record = positions[index]
            entries.append((f"release-ledger/{record['service']}.json", json.dumps(record)))
        for path, text in entries:
            call_id = f"read_{len(messages)}"
            messages.extend(
                [
                    AIMessage(
                        "Read workspace file",
                        tool_calls=[
                            {
                                "id": call_id,
                                "name": "read_file",
                                "args": {"file_path": path},
                            }
                        ],
                    ),
                    ToolMessage(text, tool_call_id=call_id, name="read_file"),
                ]
            )
    # Identical protected tail, so all three ledger records remain selectable/compactable.
    messages.extend(
        AIMessage(f"Workspace inspection checkpoint {i}: awaiting the next request.")
        for i in range(6)
    )
    return messages, records


def question(record: dict) -> str:
    return (
        f"Now consult ONLY the release ledger for service {record['service']}. "
        "Return its exact lease_seconds, max_payload_bytes, release_tag and owner_team. "
        "Use a JSON object with exactly those four keys. No tools or explanation. "
        "If a value is unavailable, use null instead of guessing."
    )


def grade(text: str, record: dict) -> dict:
    try:
        value = json.loads(text[text.index("{") : text.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        value = {}
    expected = {key: val for key, val in record.items() if key != "service"}
    return {
        "expected": expected,
        "actual": value,
        "correct_fields": sum(value.get(k) == v for k, v in expected.items()),
        "total_fields": len(expected),
    }


def run_condition(
    base: Path,
    condition: str,
    messages: list[BaseMessage],
    records: list[dict],
    key: str,
    model_factory: Callable[[Journal, str], BaseChatModel] | None = None,
    resume: bool = False,
    question_builder: Callable[[dict], str] = question,
    answer_grader: Callable[[str, dict], dict] = grade,
) -> dict:
    directory = base / condition
    if (directory / "result.json").exists():
        return json.loads((directory / "result.json").read_text())
    if (directory / "events.jsonl").exists() and not resume:
        raise RuntimeError(f"Partial run exists in {directory}; preserve it and use a new run ID")
    journal = Journal(directory, condition, resume=resume)
    model = (
        model_factory(journal, key)
        if model_factory is not None
        else MeteredGLM(
            api_key=SecretStr(key), journal=journal, profile={"max_input_tokens": 1_000_000}
        )
    )
    backend = FilesystemBackend(root_dir=directory / "backend", virtual_mode=True)
    (directory / "backend").mkdir(exist_ok=True)
    selections = []
    if condition == "summarization":
        summary_model = model.model_copy(
            update={
                "stage": "summary",
                "output_limit" if isinstance(model, MeteredGLM) else "max_tokens": 8192,
            }
        )
        middleware = SummarizationMiddleware(
            model=summary_model,
            backend=backend,
            trigger=("tokens", 16_000),
            keep=("messages", 6),
            trim_tokens_to_summarize=None,
        )
    elif condition == "jev":
        client = JevClient(
            api_key=read_key("JEV_API_KEY", ".jev"), transport=MeteredTransport(journal), timeout=30
        )
        middleware = JevContextMiddleware(
            client,
            config=SelectionConfig(max_tokens=16_000, trigger_tokens=4_000, keep_recent=6),
            on_selection=lambda report: selections.append(asdict(report)),
        )
    else:
        middleware = NoCompaction()
    agent = create_deep_agent(
        model=model,
        backend=backend,
        middleware=[middleware],
        checkpointer=InMemorySaver(),
        system_prompt="Answer precisely from the supplied workspace history. Do not invent values.",
    )
    config = {"configurable": {"thread_id": f"{base.name}-{condition}"}, "recursion_limit": 10}
    turns, start = [], time.perf_counter()
    for turn, record in enumerate(records, 1):
        journal.turn = turn
        incoming = (
            [*messages, HumanMessage(question_builder(record))]
            if turn == 1
            else [HumanMessage(question_builder(record))]
        )
        turn_start = time.perf_counter()
        replay_before = journal.replayed_seconds
        result = agent.invoke({"messages": incoming}, config)
        answer = result["messages"][-1].text
        events = [e for e in journal.events if e["turn"] == turn]
        turns.append(
            {
                "turn": turn,
                "service": record["service"],
                "answer": answer,
                "grade": answer_grader(answer, record),
                "elapsed_seconds": time.perf_counter()
                - turn_start
                + journal.replayed_seconds
                - replay_before,
                "estimated_usd": sum(e["estimated_usd"] for e in events),
                "uncached_usd": sum(e["uncached_usd"] for e in events),
            }
        )
        write_json(directory / "progress.json", {"turns": turns, "selections": selections})
    output = {
        "condition": condition,
        "turns": turns,
        "events": journal.events,
        "elapsed_seconds": time.perf_counter() - start + journal.replayed_seconds,
        "estimated_usd": sum(e["estimated_usd"] for e in journal.events),
        "uncached_usd": sum(e["uncached_usd"] for e in journal.events),
        "correct_fields": sum(t["grade"]["correct_fields"] for t in turns),
        "total_fields": sum(t["grade"]["total_fields"] for t in turns),
        "selections": selections,
    }
    write_json(directory / "result.json", output)
    print(
        "CONDITION_COMPLETE",
        base.name,
        condition,
        output["estimated_usd"],
        f"{output['correct_fields']}/{output['total_fields']}",
        flush=True,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, default=DATASETS)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/runs/glm-5.3")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["summarization", "jev", "full"],
        default=["summarization", "jev", "full"],
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Copy and verify fixtures without keys or API calls"
    )
    args = parser.parse_args()
    validate_output(args.output)
    sources = sorted(args.datasets.glob("tokens*/dataset.json"))
    if not sources:
        parser.error("No dataset.json fixtures found")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output already contains data; select a fresh run directory")
    key = None if args.dry_run else read_key("ZAI_API_KEY", ".zai")
    outputs = []
    for source in sources:
        dataset = json.loads(source.read_text())
        manifest = json.loads(source.with_name("manifest.json").read_text())
        if hashlib.sha256(json.dumps(dataset).encode()).hexdigest() != manifest["dataset_sha256"]:
            raise ValueError(f"Dataset digest mismatch: {source.name}")
        base = args.output / source.parent.name
        write_json(base / "dataset.json", dataset)
        write_json(base / "manifest.json", manifest)
        print("DATASET", base.name, manifest["actual_approximate_history_tokens"], flush=True)
        if args.dry_run:
            continue
        for condition in manifest["condition_order"]:
            if condition not in args.conditions:
                continue
            result = run_condition(
                base, condition, messages_from_dict(dataset["messages"]), dataset["records"], key
            )
            outputs.append({"case": base.name, **result})
            write_json(args.output / "results.json", outputs)


if __name__ == "__main__":
    main()
