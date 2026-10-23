"""Record one live run's model answers, then drive the same code from them for free.

A live run used to be the only test covering the whole turn, so every change had to buy
a new one. Almost every defect we have found since Phase A -- a reply saved before its
sources, a citation that passed re-reading after its row changed, a search that crossed
into another project -- lives below the model and does not need one. Recording the model's
normalized answers separates the two: our own behaviour is then verified offline, and a
live run is only needed to ask what the model does.

What a recording covers is everything after the model: routing, validation, SQLite writes,
idempotency, recovery, scope binding, the provenance ledger, source cards, reply text. What
it cannot cover is the model's own behaviour, and the provider normalization in
llm.OpenAICompatibleModel, which turns a raw response into these objects; test_llm.py
covers that from recorded response shapes instead.

A recording is tied to the requests that produced it. Replaying it against a different
request would answer a question the model was never asked, so the request is digested and
a mismatch is refused rather than papered over. Changing a prompt therefore means
re-recording, which is the intended cost of changing a prompt.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from captureflow.llm import InvalidStructuredOutput, ModelError
from captureflow.models import ConversationDecision
from captureflow.read_agent import AgentModelTurn


RECORDING_VERSION = 1

# The two calls a turn can make, with the schema each answer is read back as.
RESPONSE_SCHEMAS: dict[str, type[BaseModel]] = {
    "propose_conversation_decision": ConversationDecision,
    "next_read_agent_turn": AgentModelTurn,
}

# The loop branches on InvalidStructuredOutput, so a recording that flattened every
# failure into one class would replay the wrong control flow.
_ERROR_CLASSES: dict[str, type[Exception]] = {
    "InvalidStructuredOutput": InvalidStructuredOutput,
    "ModelError": ModelError,
}


def request_digest(method: str, request: dict[str, Any]) -> str:
    """Identify a request by content, so a changed prompt or context cannot replay silently."""

    canonical = json.dumps(
        {"method": method, "request": request}, sort_keys=True, ensure_ascii=False, default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RecordingModel:
    """Delegate to a real model and keep each normalized answer for later replay."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def propose_conversation_decision(self, **kwargs) -> ConversationDecision:
        return self._call("propose_conversation_decision", kwargs)

    def next_read_agent_turn(self, **kwargs) -> AgentModelTurn:
        return self._call("next_read_agent_turn", kwargs)

    def _call(self, method: str, kwargs: dict[str, Any]) -> Any:
        entry: dict[str, Any] = {"method": method, "request_sha256": request_digest(method, kwargs)}
        try:
            response = getattr(self.model, method)(**kwargs)
        except Exception as error:
            # A call that failed is part of the run: the loop reacts to it, so replay must too.
            entry["error"] = {"type": type(error).__name__, "message": str(error)}
            self.calls.append(entry)
            raise
        entry["response"] = _dump(response)
        self.calls.append(entry)
        return response

    def recording(self, **header: Any) -> dict[str, Any]:
        return {
            "version": RECORDING_VERSION,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **header,
            "calls": list(self.calls),
        }


class ReplayModel:
    """Answer from a recording, refusing one that no longer fits the run asking for it."""

    def __init__(self, recording: dict[str, Any]) -> None:
        version = recording.get("version")
        if version != RECORDING_VERSION:
            raise AssertionError(
                f"recording version {version} cannot be replayed by version {RECORDING_VERSION}"
            )
        self.calls = list(recording["calls"])
        self.position = 0

    def propose_conversation_decision(self, **kwargs) -> ConversationDecision:
        return self._next("propose_conversation_decision", kwargs)

    def next_read_agent_turn(self, **kwargs) -> AgentModelTurn:
        return self._next("next_read_agent_turn", kwargs)

    def _next(self, method: str, kwargs: dict[str, Any]) -> Any:
        number = self.position + 1
        if self.position >= len(self.calls):
            raise AssertionError(
                f"the recording has no call {number}, but {method} was asked for; re-record"
            )
        call = self.calls[self.position]
        self.position += 1
        if call["method"] != method:
            raise AssertionError(
                f"recorded call {number} is {call['method']}, but {method} was asked for; re-record"
            )
        if call["request_sha256"] != request_digest(method, kwargs):
            raise AssertionError(
                f"call {number} ({method}) sends a different request than was recorded; re-record"
            )
        if "error" in call:
            error = call["error"]
            raise _ERROR_CLASSES.get(error["type"], ModelError)(error["message"])
        return RESPONSE_SCHEMAS[method].model_validate(call["response"])


def _dump(response: Any) -> Any:
    return response.model_dump(mode="json") if isinstance(response, BaseModel) else response
