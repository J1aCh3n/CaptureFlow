"""A provider-neutral, bounded tool loop for read-only agent turns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .read_agent_tools import BoundReadTools, ObservedResource


MAX_AGENT_MODEL_CALLS = 3
MAX_AGENT_TOOL_CALLS = 6
MAX_AGENT_SOURCE_IDS = 12

AgentStopReason = Literal[
    "completed",
    "validation_failed",
    "policy_denied",
    "model_budget_exhausted",
    "tool_budget_exhausted",
]


class AgentFinalResponse(BaseModel):
    """The model's terminal answer; source IDs are verified against the ledger."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    response_kind: Literal["grounded_answer", "clarification", "no_match"]
    reply: str | None = Field(default=None, max_length=8_000)
    source_ids: list[str] = Field(default_factory=list, max_length=MAX_AGENT_SOURCE_IDS)

    @field_validator("reply")
    @classmethod
    def normalize_reply(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("source_ids")
    @classmethod
    def source_ids_are_unique(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("source_ids must be unique")
        return values

    @model_validator(mode="after")
    def response_shape_matches_kind(self) -> "AgentFinalResponse":
        if self.response_kind == "grounded_answer" and not self.source_ids:
            raise ValueError("grounded_answer requires at least one source ID")
        if self.response_kind != "grounded_answer" and self.source_ids:
            raise ValueError("only grounded_answer may cite source IDs")
        if self.response_kind == "clarification" and self.reply is None:
            raise ValueError("clarification requires a question in reply")
        return self


RESPOND_TOOL = "respond"
# The final answer arrives as this tool's arguments. Asked to write JSON content against a
# schema in the prompt, qwen3.7-flash-2026-07-15 echoed the schema instead of an answer.
_RESPOND_DEFINITION = {
    "type": "function",
    "function": {
        "name": RESPOND_TOOL,
        "description": "Give your final response. Call it once, alone, when you have finished searching.",
        "parameters": AgentFinalResponse.model_json_schema(),
    },
}


class AgentToolCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=100)
    arguments: dict[str, Any]


class AgentModelTurn(BaseModel):
    """Normalized provider output: either tool calls or a terminal response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_calls: list[AgentToolCall] = Field(default_factory=list)
    final_response: AgentFinalResponse | None = None

    @model_validator(mode="after")
    def exactly_one_next_step(self) -> "AgentModelTurn":
        if bool(self.tool_calls) == (self.final_response is not None):
            raise ValueError("a model turn must contain tool calls or one final response")
        call_ids = [call.id for call in self.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("tool call IDs must be unique within a model turn")
        return self

    def assistant_message(self) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in self.tool_calls
            ],
        }


class ReadAgentModel(Protocol):
    def next_read_agent_turn(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AgentModelTurn: ...


@dataclass
class ReadAgentUsage:
    """What a run has spent so far. The caller owns it, so a run that raises still reports it."""

    model_calls: int = 0
    tool_calls: int = 0
    model_seconds: float = 0.0
    tool_seconds: float = 0.0


@dataclass(frozen=True)
class ReadAgentRun:
    final_response: AgentFinalResponse | None
    observed_resources: dict[str, ObservedResource]
    model_calls: int
    tool_calls: int
    model_seconds: float
    tool_seconds: float
    stop_reason: AgentStopReason


class ReadAgentState(TypedDict, total=False):
    messages: list[dict[str, Any]]
    tools: BoundReadTools
    model_turn: AgentModelTurn
    final_response: AgentFinalResponse | None
    usage: ReadAgentUsage
    repair_attempted: bool
    stop_reason: AgentStopReason | None


class ReadAgentLoop:
    """Execute a finite read/tool loop without knowing the router's intent labels."""

    def __init__(
        self,
        model: ReadAgentModel,
        *,
        max_model_calls: int = MAX_AGENT_MODEL_CALLS,
        max_tool_calls: int = MAX_AGENT_TOOL_CALLS,
    ) -> None:
        if max_model_calls < 1 or max_tool_calls < 1:
            raise ValueError("agent budgets must be positive")
        self.model = model
        self.max_model_calls = max_model_calls
        self.max_tool_calls = max_tool_calls
        self.graph = self._build_graph()

    def run(
        self,
        *,
        system_prompt: str,
        user_input: dict[str, Any],
        tools: BoundReadTools,
        usage: ReadAgentUsage | None = None,
    ) -> ReadAgentRun:
        usage = ReadAgentUsage() if usage is None else usage
        initial: ReadAgentState = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"{system_prompt}\nCall independent search tools together when useful. When you "
                        f"have enough evidence or no matching results, call {RESPOND_TOOL} once, alone, "
                        "with the final response; never write the final answer as message text."
                    ),
                },
                {"role": "user", "content": json.dumps(user_input, ensure_ascii=False)},
            ],
            "tools": tools,
            "final_response": None,
            "usage": usage,
            "repair_attempted": False,
            "stop_reason": None,
        }
        state = self.graph.invoke(initial)
        stop_reason = state.get("stop_reason")
        if stop_reason is None:
            raise RuntimeError("read agent graph ended without a stop reason")
        return ReadAgentRun(
            final_response=state.get("final_response"),
            observed_resources=dict(tools.observed_resources),
            model_calls=usage.model_calls,
            tool_calls=usage.tool_calls,
            model_seconds=usage.model_seconds,
            tool_seconds=usage.tool_seconds,
            stop_reason=stop_reason,
        )

    def _build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(ReadAgentState)
        graph.add_node("call_model", self._call_model)
        graph.add_node("run_tools", self._run_tools)
        graph.add_node("validate_final", self._validate_final)
        graph.add_edge(START, "call_model")
        graph.add_conditional_edges(
            "call_model", _after_model, [END, "run_tools", "validate_final"]
        )
        graph.add_conditional_edges("run_tools", _continue_or_end, [END, "call_model"])
        graph.add_conditional_edges("validate_final", _continue_or_end, [END, "call_model"])
        return graph.compile()

    def _call_model(self, state: ReadAgentState) -> ReadAgentState:
        usage = state["usage"]
        if usage.model_calls >= self.max_model_calls:
            return {"stop_reason": "model_budget_exhausted"}
        # A call counts once it is made, including one that fails.
        usage.model_calls += 1
        started = perf_counter()
        try:
            turn = AgentModelTurn.model_validate(
                self.model.next_read_agent_turn(
                    messages=state["messages"],
                    tools=[*state["tools"].definitions_for_model(), _RESPOND_DEFINITION],
                )
            )
        finally:
            usage.model_seconds += perf_counter() - started
        update: ReadAgentState = {"model_turn": turn}
        # The single repair is only for correcting citations, not for making an
        # invented citation valid after the fact with another lookup.
        if state["repair_attempted"] and turn.tool_calls:
            update["stop_reason"] = "validation_failed"
        return update

    def _run_tools(self, state: ReadAgentState) -> ReadAgentState:
        usage = state["usage"]
        calls = state["model_turn"].tool_calls
        if usage.tool_calls + len(calls) > self.max_tool_calls:
            return {"stop_reason": "tool_budget_exhausted"}

        messages = [*state["messages"], state["model_turn"].assistant_message()]
        stop_reason: AgentStopReason | None = None
        started = perf_counter()
        try:
            for call in calls:
                usage.tool_calls += 1
                result = state["tools"].invoke(call.name, call.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result.model_dump(mode="json"), ensure_ascii=False),
                    }
                )
                if result.error is not None and result.error.category == "policy_denied":
                    stop_reason = "policy_denied"
                    break
        finally:
            usage.tool_seconds += perf_counter() - started
        return {"messages": messages, "stop_reason": stop_reason}

    def _validate_final(self, state: ReadAgentState) -> ReadAgentState:
        response = state["model_turn"].final_response
        assert response is not None
        unobserved = [
            source_id
            for source_id in response.source_ids
            if source_id not in state["tools"].observed_resources
        ]
        if not unobserved:
            return {"final_response": response, "stop_reason": "completed"}
        if state["repair_attempted"] or state["usage"].model_calls >= self.max_model_calls:
            return {"stop_reason": "validation_failed"}
        messages = [
            *state["messages"],
            {
                "role": "user",
                "content": (
                    "Your final source_ids included IDs that no tool returned in this run. "
                    f"Call {RESPOND_TOOL} again with a corrected response using only IDs from tool results; "
                    "do not make another tool call merely to justify an invented ID."
                ),
            },
        ]
        return {"messages": messages, "repair_attempted": True}


def _after_model(state: ReadAgentState) -> str:
    if state.get("stop_reason") is not None:
        return END
    return "run_tools" if state["model_turn"].tool_calls else "validate_final"


def _continue_or_end(state: ReadAgentState) -> str:
    return END if state.get("stop_reason") is not None else "call_model"
