"""Probe whether a model supports the four provider behaviours the read agent depends on.

The agent loop needs more than "function calling works". It needs, in order of how much
a failure would cost us:

1. one tool call with valid JSON arguments, when the question can only be answered by a tool;
2. thinking disabled, because DashScope Qwen defaults to thinking mode and the loop would
   otherwise have to replay provider-specific reasoning_content on every tool round;
3. several independent tool calls in one turn, which is what the loop's prompt asks for;
4. a forced tool_choice on `respond`, which is how the final answer is read at all.

This probe reports each one separately per model, because a model can pass 1 and fail 4.
Requests are built in the same shape as llm.next_read_agent_turn, but the raw response is
inspected rather than normalized, so the result describes the provider and not our adapter.

Usage (base URL and key come from the environment, as elsewhere):

    CAPTUREFLOW_API_KEY=... CAPTUREFLOW_BASE_URL=... \
        python -m scripts.run_function_calling_probe --output runs/fc-probe
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from openai import OpenAI

from captureflow.read_agent import RESPOND_TOOL, AgentFinalResponse


DEFAULT_MODELS = ("qwen3.8-27b", "qwen3.8-2.4t-a95b", "qwen3.8-max")
DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
REFERENCE_DATE = "2026-09-14"
NEXT_WEEK = ("2026-09-14", "2026-09-20")

SEARCH_CALENDAR = {
    "type": "function",
    "function": {
        "name": "search_calendar",
        "description": "Search stored calendar events. Filter by start_date and/or end_date.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                "keywords": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
    },
}

SEARCH_PROJECT_RECORDS = {
    "type": "function",
    "function": {
        "name": "search_project_records",
        "description": "Search tracker records in the current project. Use kinds and states to list categories.",
        "parameters": {
            "type": "object",
            "properties": {
                "kinds": {
                    "type": "array",
                    "items": {"enum": ["task", "decision", "blocker", "milestone"]},
                },
                "states": {"type": "array", "items": {"enum": ["active", "completed"]}},
                "keywords": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
    },
}

RESPOND_DEFINITION = {
    "type": "function",
    "function": {
        "name": RESPOND_TOOL,
        "description": "Give your final response. Call it once, alone, when you have finished searching.",
        "parameters": AgentFinalResponse.model_json_schema(),
    },
}

SYSTEM = (
    "You are a read-only agent over stored local data. Answer only from what the tools return; "
    f"never answer from memory. Today is {REFERENCE_DATE}."
)


@dataclass
class ProbeOutcome:
    name: str
    passed: bool
    detail: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning_content: bool = False
    finish_reason: str | None = None
    seconds: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None

    def row(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"  [{mark}] {self.name}: {self.detail}"


def _call(
    client: OpenAI,
    model: str,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    disable_thinking: bool,
    tool_choice: dict[str, Any] | None = None,
    timeout: float = 60.0,
) -> tuple[Any, float]:
    request: dict[str, Any] = {
        "model": model,
        "timeout": timeout,
        "messages": messages,
        "tools": tools,
    }
    if tool_choice is not None:
        request["tool_choice"] = tool_choice
    if disable_thinking:
        request["extra_body"] = {"enable_thinking": False}
    started = perf_counter()
    response = client.chat.completions.create(**request)
    return response, perf_counter() - started


def _read(response: Any) -> tuple[list[dict[str, Any]], bool, str | None]:
    choice = response.choices[0]
    message = choice.message
    calls = []
    for raw in getattr(message, "tool_calls", None) or []:
        try:
            arguments = json.loads(raw.function.arguments)
        except (TypeError, json.JSONDecodeError):
            arguments = {"__unparsed__": raw.function.arguments}
        calls.append({"name": raw.function.name, "arguments": arguments})
    # DashScope returns thinking as reasoning_content; its presence means thinking stayed on.
    reasoning = bool(getattr(message, "reasoning_content", None))
    return calls, reasoning, choice.finish_reason


def _usage(response: Any) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None)
    return getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)


def _outcome(
    name: str, response: Any, seconds: float, *, passed: bool, detail: str
) -> ProbeOutcome:
    calls, reasoning, finish_reason = _read(response)
    prompt_tokens, completion_tokens = _usage(response)
    return ProbeOutcome(
        name=name,
        passed=passed,
        detail=detail,
        tool_calls=calls,
        reasoning_content=reasoning,
        finish_reason=finish_reason,
        seconds=round(seconds, 3),
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
    )


def probe_single_tool(client: OpenAI, model: str, *, disable_thinking: bool) -> ProbeOutcome:
    """1 and 2: one tool call with valid arguments, and whether thinking stayed off."""

    name = "single_tool_call" + ("" if disable_thinking else "_thinking_default")
    try:
        response, seconds = _call(
            client,
            model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": f"What is on my calendar next week ({NEXT_WEEK[0]} to {NEXT_WEEK[1]})?",
                },
            ],
            tools=[SEARCH_CALENDAR],
            disable_thinking=disable_thinking,
        )
    except Exception as error:  # the provider refusing the request is itself the finding
        return ProbeOutcome(name=name, passed=False, detail=f"request failed: {error}")

    calls, reasoning, _ = _read(response)
    if not calls:
        return _outcome(name, response, seconds, passed=False, detail="answered without calling a tool")
    if calls[0]["name"] != "search_calendar":
        return _outcome(
            name, response, seconds, passed=False, detail=f"called {calls[0]['name']}"
        )
    if "__unparsed__" in calls[0]["arguments"]:
        return _outcome(name, response, seconds, passed=False, detail="arguments were not valid JSON")
    thinking = " (reasoning_content present)" if reasoning else ""
    return _outcome(
        name,
        response,
        seconds,
        passed=True,
        detail=f"called search_calendar with {json.dumps(calls[0]['arguments'], ensure_ascii=False)}{thinking}",
    )


def probe_parallel_tools(client: OpenAI, model: str) -> ProbeOutcome:
    """3: two independent searches in one turn, which the loop's prompt asks for."""

    try:
        response, seconds = _call(
            client,
            model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": (
                        "下周有什么安排？另外项目里现在有哪些待办任务？"
                        f"（下周是 {NEXT_WEEK[0]} 到 {NEXT_WEEK[1]}）"
                    ),
                },
            ],
            tools=[SEARCH_CALENDAR, SEARCH_PROJECT_RECORDS],
            disable_thinking=True,
        )
    except Exception as error:
        return ProbeOutcome(name="parallel_tool_calls", passed=False, detail=f"request failed: {error}")

    calls, _, _ = _read(response)
    names = [call["name"] for call in calls]
    if len(calls) >= 2:
        detail = f"{len(calls)} calls in one turn: {names}"
    elif calls:
        # Not a failure of the loop: it costs one more round, so the budget must allow it.
        detail = f"one call per turn: {names}"
    else:
        detail = "answered without calling a tool"
    return _outcome("parallel_tool_calls", response, seconds, passed=bool(calls), detail=detail)


def probe_forced_respond(client: OpenAI, model: str) -> ProbeOutcome:
    """4: a forced tool_choice on respond, with arguments valid against the real schema."""

    try:
        response, seconds = _call(
            client,
            model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "下周有什么安排？"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_probe_1",
                            "type": "function",
                            "function": {
                                "name": "search_calendar",
                                "arguments": json.dumps(
                                    {"start_date": NEXT_WEEK[0], "end_date": NEXT_WEEK[1]}
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_probe_1",
                    "content": json.dumps(
                        {
                            "ok": True,
                            "items": [
                                {
                                    "id": "event-1",
                                    "type": "calendar_event",
                                    "title": "机械臂延迟评审会议",
                                    "start_at": "2026-09-16T14:00",
                                }
                            ],
                            "total": 1,
                            "keyword_matches": None,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            tools=[RESPOND_DEFINITION],
            disable_thinking=True,
            tool_choice={"type": "function", "function": {"name": RESPOND_TOOL}},
        )
    except Exception as error:
        return ProbeOutcome(name="forced_respond", passed=False, detail=f"request failed: {error}")

    calls, _, _ = _read(response)
    if not calls or calls[0]["name"] != RESPOND_TOOL:
        return _outcome(
            "forced_respond", response, seconds, passed=False, detail=f"did not call {RESPOND_TOOL}"
        )
    try:
        final = AgentFinalResponse.model_validate(calls[0]["arguments"])
    except Exception as error:
        return _outcome(
            "forced_respond", response, seconds, passed=False, detail=f"arguments rejected: {error}"
        )
    cited = "cited event-1" if "event-1" in final.source_ids else f"source_ids={final.source_ids}"
    return _outcome(
        "forced_respond", response, seconds, passed=True, detail=f"valid reply, {cited}"
    )


def probe_model(client: OpenAI, model: str) -> dict[str, Any]:
    outcomes = [
        probe_single_tool(client, model, disable_thinking=True),
        probe_single_tool(client, model, disable_thinking=False),
        probe_parallel_tools(client, model),
        probe_forced_respond(client, model),
    ]
    required = {"single_tool_call", "forced_respond"}
    usable = all(outcome.passed for outcome in outcomes if outcome.name in required)
    return {
        "model": model,
        "usable_for_agent_loop": usable,
        "input_tokens": _total(outcome.input_tokens for outcome in outcomes),
        "output_tokens": _total(outcome.output_tokens for outcome in outcomes),
        "probes": [outcome.__dict__ for outcome in outcomes],
        "_outcomes": outcomes,
    }


def _total(values) -> int | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--output", type=Path, help="Directory to write probe.json into.")
    arguments = parser.parse_args()

    api_key = os.getenv("CAPTUREFLOW_API_KEY")
    if not api_key:
        print("Set CAPTUREFLOW_API_KEY before probing.")
        return 2
    base_url = os.getenv("CAPTUREFLOW_BASE_URL", DEFAULT_BASE_URL)
    client = OpenAI(api_key=api_key, base_url=base_url)

    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "respond_schema": AgentFinalResponse.model_json_schema(),
        "models": [],
    }
    print(f"base_url: {base_url}\n")
    for model in arguments.models:
        print(f"{model}")
        result = probe_model(client, model)
        for outcome in result.pop("_outcomes"):
            print(outcome.row())
        print(
            f"  -> usable for the agent loop: {result['usable_for_agent_loop']}"
            f"  ({result['input_tokens']} in / {result['output_tokens']} out)\n"
        )
        report["models"].append(result)

    if arguments.output:
        arguments.output.mkdir(parents=True, exist_ok=True)
        destination = arguments.output / "probe.json"
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
