"""A provider-neutral, bounded tool loop for read-only agent turns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .read_agent_tools import BoundReadTools, ObservedResource, ReadToolError, ReadToolResult


MAX_AGENT_MODEL_CALLS = 3
MAX_AGENT_TOOL_CALLS = 6

AgentStopReason = Literal[
    "completed",
    "handed_off",
    "validation_failed",
    "model_budget_exhausted",
    "tool_budget_exhausted",
]


class AgentFinalResponse(BaseModel):
    """The model's terminal answer; source IDs are verified against the ledger."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reply: str = Field(min_length=1, max_length=8_000)
    source_ids: list[str] = Field(default_factory=list)

    @field_validator("reply")
    @classmethod
    def normalize_reply(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("reply must contain text")
        return value

    @field_validator("source_ids")
    @classmethod
    def deduplicate_source_ids(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))


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


class HandOffRequest(BaseModel):
    """What the model says the message asks to change; the router reads the message itself."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request: str = Field(min_length=1, max_length=300)

    @field_validator("request")
    @classmethod
    def normalize_request(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("request must contain text")
        return value


HAND_OFF_TOOL = "hand_off_write"
# Writing stays behind the router's validated path, so the loop only reports that a write
# was asked for and stops. Calling it once and stopping is what keeps one turn to one write.
_HAND_OFF_DEFINITION = {
    "type": "function",
    "function": {
        "name": HAND_OFF_TOOL,
        "description": (
            "Call this when the message asks to create, change or delete a calendar event, an "
            "idea, or a tracker record such as a task, decision or blocker. CaptureFlow carries "
            "out the change itself after this call, so call it alone and do not search first. A "
            "note is not one of these: save_note stores it, when that tool is offered."
        ),
        "parameters": HandOffRequest.model_json_schema(),
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
    hand_off: str | None
    observed_resources: dict[str, ObservedResource]
    model_calls: int
    tool_calls: int
    model_seconds: float
    tool_seconds: float
    stop_reason: AgentStopReason


class ReadAgentState(TypedDict, total=False):
    messages: list[dict[str, Any]]
    tools: BoundReadTools
    hand_off_allowed: bool
    model_turn: AgentModelTurn | None
    final_response: AgentFinalResponse | None
    hand_off: str | None
    usage: ReadAgentUsage
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
        hand_off: bool = False,
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
            "hand_off_allowed": hand_off,
            "final_response": None,
            "hand_off": None,
            "usage": usage,
            "stop_reason": None,
        }
        state = self.graph.invoke(initial)
        stop_reason = state.get("stop_reason")
        if stop_reason is None:
            raise RuntimeError("read agent graph ended without a stop reason")
        return ReadAgentRun(
            final_response=state.get("final_response"),
            hand_off=state.get("hand_off"),
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
            "call_model", _after_model, [END, "call_model", "run_tools", "validate_final"]
        )
        graph.add_conditional_edges("run_tools", _continue_or_end, [END, "call_model"])
        graph.add_conditional_edges("validate_final", _continue_or_end, [END, "call_model"])
        return graph.compile()

    def _call_model(self, state: ReadAgentState) -> ReadAgentState:
        from .llm import InvalidStructuredOutput

        usage = state["usage"]
        if usage.model_calls >= self.max_model_calls:
            return {"stop_reason": "model_budget_exhausted"}
        # Reserve the last call for an answer, even when searches have budget left.
        finishing = (
            usage.model_calls == self.max_model_calls - 1
            or usage.tool_calls >= self.max_tool_calls
        )
        messages = state["messages"]
        # A write can still be handed back on the reserved call: offering only respond would
        # force an answer to a message that asked for a change. Once this turn has stored a
        # change, handing another one back would make two, which the stable IDs of a prepared
        # operation cannot represent.
        hand_off_open = bool(state.get("hand_off_allowed")) and not state["tools"].wrote_this_turn
        closing = [_RESPOND_DEFINITION]
        if hand_off_open:
            closing.append(_HAND_OFF_DEFINITION)
        definitions = [*state["tools"].definitions_for_model(), *closing]
        if finishing:
            definitions = closing
            messages = [*messages, {
                "role": "user",
                "content": (
                    "The search budget for this turn is finished. Call respond now using the "
                    "evidence already returned. Say what remains unconfirmed if searches were "
                    "incomplete; do not infer that missing evidence means no data exists."
                ),
            }]
        # A call counts once it is made, including one that fails.
        usage.model_calls += 1
        started = perf_counter()
        try:
            turn = AgentModelTurn.model_validate(
                self.model.next_read_agent_turn(
                    messages=messages,
                    tools=definitions,
                )
            )
        except (InvalidStructuredOutput, ValidationError) as error:
            if usage.model_calls >= self.max_model_calls:
                return {"model_turn": None, "stop_reason": "validation_failed"}
            return {
                "model_turn": None,
                "messages": [*messages, {
                    "role": "user",
                    "content": f"Your response could not be read: {error}. Use the offered tools or call respond alone with a valid reply.",
                }],
            }
        finally:
            usage.model_seconds += perf_counter() - started
        update: ReadAgentState = {"model_turn": turn, "messages": messages}
        handed_off = next(
            (call for call in turn.tool_calls if call.name == HAND_OFF_TOOL), None
        )
        if handed_off is not None and hand_off_open:
            try:
                update["hand_off"] = HandOffRequest.model_validate(handed_off.arguments).request
            except ValidationError:
                # The restatement is only a note in the trace; the router reads the message
                # itself, so an unreadable one is not worth a round trip. stop_reason carries
                # the hand-off either way.
                update["hand_off"] = None
            update["stop_reason"] = "handed_off"
            return update
        if finishing and turn.tool_calls:
            update["stop_reason"] = (
                "tool_budget_exhausted" if usage.tool_calls >= self.max_tool_calls
                else "model_budget_exhausted"
            )
        return update

    def _run_tools(self, state: ReadAgentState) -> ReadAgentState:
        usage = state["usage"]
        turn = state["model_turn"]
        assert turn is not None
        messages = [*state["messages"], turn.assistant_message()]
        started = perf_counter()
        try:
            for call in turn.tool_calls:
                if usage.tool_calls >= self.max_tool_calls:
                    result = ReadToolResult(ok=False, error=ReadToolError(
                        category="retryable", message="Tool-call budget exhausted; this search was not run."
                    ))
                else:
                    usage.tool_calls += 1
                    result = state["tools"].invoke(call.name, call.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result.model_dump(mode="json"), ensure_ascii=False),
                    }
                )
        finally:
            usage.tool_seconds += perf_counter() - started
        return {"messages": messages}

    def _validate_final(self, state: ReadAgentState) -> ReadAgentState:
        turn = state["model_turn"]
        assert turn is not None
        response = turn.final_response
        assert response is not None
        unobserved = [
            source_id
            for source_id in response.source_ids
            if source_id not in state["tools"].observed_resources
        ]
        if not unobserved:
            return {"final_response": response, "stop_reason": "completed"}
        if state["usage"].model_calls >= self.max_model_calls:
            return {"stop_reason": "validation_failed"}
        messages = [
            *state["messages"],
            {"role": "assistant", "content": response.model_dump_json()},
            {
                "role": "user",
                "content": (
                    "Your final source_ids included IDs that no tool returned in this run. "
                    f"Unobserved IDs: {json.dumps(unobserved, ensure_ascii=False)}. "
                    "Correct the reply and citations, or search for the needed evidence with the offered tools."
                ),
            },
        ]
        return {"messages": messages}


def _after_model(state: ReadAgentState) -> str:
    if state.get("stop_reason") is not None:
        return END
    turn = state["model_turn"]
    if turn is None:
        return "call_model"
    return "run_tools" if turn.tool_calls else "validate_final"


def _continue_or_end(state: ReadAgentState) -> str:
    return END if state.get("stop_reason") is not None else "call_model"
