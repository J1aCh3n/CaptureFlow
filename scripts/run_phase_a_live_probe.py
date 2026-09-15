"""Probe Phase A read-agent tool calling with Qwen against an isolated database."""

import argparse
import json
import os
import tempfile
from datetime import date
from pathlib import Path
from time import perf_counter

from captureflow.conversation_router_service import ConversationRouterService
from captureflow.llm import ModelSettings, OpenAICompatibleModel
from captureflow.models import CalendarEvent, Conversation, Message
from captureflow.read_agent import ReadAgentLoop
from captureflow.read_agent_tools import BoundReadTools
from captureflow.storage import CaptureStore


DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.8-flash"
REFERENCE_DATE = date(2026, 9, 14)
EVENT_DATE = date(2026, 9, 20)


class TokenCounter:
    """Record request usage without retaining request or response content."""

    def __init__(self, model: OpenAICompatibleModel) -> None:
        self.requests: list[tuple[int | None, int | None]] = []
        create = model.client.chat.completions.create

        def counted_create(**kwargs):
            response = create(**kwargs)
            usage = getattr(response, "usage", None)
            self.requests.append(
                (getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None))
            )
            return response

        model.client.chat.completions.create = counted_create

    def since(self, first_request: int) -> dict[str, int | None]:
        requests = self.requests[first_request:]
        return {
            "input_tokens": _total(value for value, _ in requests),
            "output_tokens": _total(value for _, value in requests),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--include-router", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.runs <= 10:
        raise SystemExit("--runs must be between 1 and 10")

    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        raise SystemExit("QWEN_API_KEY is not configured")
    settings = ModelSettings(
        api_key=api_key,
        model=os.getenv("QWEN_MODEL", DEFAULT_MODEL),
        base_url=os.getenv("QWEN_BASE_URL", DEFAULT_BASE_URL),
        timeout_seconds=120,
    )
    model = OpenAICompatibleModel(settings)
    tokens = TokenCounter(model)
    report = {
        "model": settings.model,
        "base_url": settings.base_url,
        "direct_runs": [],
        "router_run": None,
    }

    with tempfile.TemporaryDirectory() as directory:
        store, conversation, event = _seed(Path(directory) / "phase-a.sqlite3")
        for run_index in range(args.runs):
            first_request = len(tokens.requests)
            started = perf_counter()
            try:
                result = ReadAgentLoop(model).run(
                    system_prompt=(
                        "You are a read-only local information agent. Always use the available search "
                        "tools before answering. Answer only from returned rows and cite their exact IDs."
                    ),
                    user_input={
                        "question": "What calendar event is stored on 2026-09-20?",
                        "reference_date": REFERENCE_DATE.isoformat(),
                        "timezone": "America/Vancouver",
                    },
                    tools=BoundReadTools.for_conversation(
                        store, conversation.id, "local_answer"
                    ),
                )
                passed = (
                    result.stop_reason == "completed"
                    and result.final_response is not None
                    and event.id in result.final_response.source_ids
                    and result.tool_calls >= 1
                )
                report["direct_runs"].append(
                    {
                        "run": run_index + 1,
                        "passed": passed,
                        "elapsed_seconds": round(perf_counter() - started, 3),
                        "model_calls": result.model_calls,
                        "tool_calls": result.tool_calls,
                        "stop_reason": result.stop_reason,
                        **tokens.since(first_request),
                    }
                )
            except Exception as error:
                report["direct_runs"].append(
                    {
                        "run": run_index + 1,
                        "passed": False,
                        "elapsed_seconds": round(perf_counter() - started, 3),
                        "error_type": type(error).__name__,
                        "error": _safe_error(error, api_key),
                        **tokens.since(first_request),
                    }
                )

        if args.include_router:
            first_request = len(tokens.requests)
            started = perf_counter()
            try:
                turn = ConversationRouterService(
                    store,
                    model,
                    REFERENCE_DATE,
                    answer_mode="agent",
                ).submit(
                    conversation.id,
                    "What calendar events do I have from September 20 through September 22?",
                    "phase-a-router-question",
                )
                cited = {
                    source.reference.source_id
                    for source in store.list_local_context_answer_sources(turn.assistant_message.id)
                }
                report["router_run"] = {
                    "passed": (
                        turn.outcome == "agent_local_answer"
                        and event.id in cited
                        and not turn.operations
                    ),
                    "elapsed_seconds": round(perf_counter() - started, 3),
                    "intent": None if turn.decision is None else turn.decision.intent,
                    "outcome": turn.outcome,
                    "model_calls": turn.model_calls,
                    "tool_calls": turn.tool_calls,
                    "stop_reason": turn.agent_stop_reason,
                    "writes": len(turn.operations),
                    **tokens.since(first_request),
                }
            except Exception as error:
                report["router_run"] = {
                    "passed": False,
                    "elapsed_seconds": round(perf_counter() - started, 3),
                    "error_type": type(error).__name__,
                    "error": _safe_error(error, api_key),
                    **tokens.since(first_request),
                }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    direct_passed = all(run["passed"] for run in report["direct_runs"])
    router_passed = report["router_run"] is None or report["router_run"]["passed"]
    return 0 if direct_passed and router_passed else 1


def _seed(database_path: Path) -> tuple[CaptureStore, Conversation, CalendarEvent]:
    store = CaptureStore(database_path)
    store.initialize()
    conversation = store.save_conversation(Conversation(id="phase-a-chat", title="Phase A"))
    source = store.save_message(
        Message(
            id="phase-a-source",
            conversation_id=conversation.id,
            role="user",
            content="The architecture review is on September 20.",
        )
    )
    event = store.save_calendar_event(
        CalendarEvent(
            id="phase-a-architecture-review",
            conversation_id=conversation.id,
            title="Architecture review",
            description="Review the read-agent tool loop.",
            source_message_id=source.id,
            source_quote=source.content,
            start_at=EVENT_DATE,
            all_day=True,
            timezone="America/Vancouver",
        )
    )
    return store, conversation, event


def _safe_error(error: Exception, api_key: str) -> str:
    return str(error).replace(api_key, "[REDACTED]")[:500]


def _total(values) -> int | None:
    values = list(values)
    return None if not values or None in values else sum(values)


if __name__ == "__main__":
    raise SystemExit(main())
