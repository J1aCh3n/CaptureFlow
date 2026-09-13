"""The one model call per chat turn: an OpenAI-compatible adapter and its prompt."""

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .models import ConversationDecision, ProjectAnswer


class ModelError(RuntimeError):
    """A configured model could not supply a schema-valid response."""


class InvalidStructuredOutput(ModelError):
    """The model answered, but not with a decision that fits the schema and its inputs."""


class ContextualConversationModel(Protocol):
    def propose_conversation_decision(
        self,
        current_message: dict[str, str],
        recent_messages: list[dict[str, str]],
        candidate_events: list[dict],
        candidate_saved_items: list[dict],
        candidate_project_records: list[dict],
        reference_date: str,
        timezone: str,
        pending_clarification: dict | None = None,
        context_truncated: bool = False,
        calendar: list[dict[str, str]] | None = None,
        reference_time: str | None = None,
    ) -> ConversationDecision: ...

    def answer_project_question(
        self,
        question: dict[str, str],
        recent_messages: list[dict[str, str]],
        project_snapshot: dict,
        reference_date: str,
    ) -> ProjectAnswer: ...


@dataclass(frozen=True)
class ModelSettings:
    api_key: str
    model: str
    base_url: str | None = None
    timeout_seconds: float = 30

    @classmethod
    def from_environment(cls) -> "ModelSettings":
        api_key = os.getenv("CAPTUREFLOW_API_KEY")
        model = os.getenv("CAPTUREFLOW_MODEL")
        if not api_key or not model:
            raise ModelError("Set CAPTUREFLOW_API_KEY and CAPTUREFLOW_MODEL before chatting.")
        try:
            timeout_seconds = float(os.getenv("CAPTUREFLOW_MODEL_TIMEOUT_SECONDS", "30"))
        except ValueError as error:
            raise ModelError("CAPTUREFLOW_MODEL_TIMEOUT_SECONDS must be numeric.") from error
        if timeout_seconds <= 0:
            raise ModelError("CAPTUREFLOW_MODEL_TIMEOUT_SECONDS must be positive.")
        return cls(
            api_key=api_key,
            model=model,
            base_url=os.getenv("CAPTUREFLOW_BASE_URL"),
            timeout_seconds=timeout_seconds,
        )


Schema = TypeVar("Schema", bound=BaseModel)


class OpenAICompatibleModel:
    """A small adapter for providers exposing OpenAI-compatible chat completions."""

    def __init__(self, settings: ModelSettings) -> None:
        self.settings = settings
        self.client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)

    def propose_conversation_decision(
        self,
        current_message: dict[str, str],
        recent_messages: list[dict[str, str]],
        candidate_events: list[dict],
        candidate_saved_items: list[dict],
        candidate_project_records: list[dict],
        reference_date: str,
        timezone: str,
        pending_clarification: dict | None = None,
        context_truncated: bool = False,
        calendar: list[dict[str, str]] | None = None,
        reference_time: str | None = None,
    ) -> ConversationDecision:
        """Make one untrusted, context-aware routing proposal; storage remains downstream."""

        decision = self._complete(
            ConversationDecision,
            system=(
                "You are CaptureFlow's Phase 10 contextual conversation router. Return JSON only. "
                "You do not execute tools or write SQLite. Choose exactly one intent and use only "
                "the supplied current_message, recent_messages, candidates, reference_date, reference_time, "
                "timezone, calendar, and pending clarification. This is not a general chatbot. "
                "A definite user statement that a meeting or event is still going to happen on an explicit "
                "date must use calendar_create, even when it is not phrased as a command. Use the event title "
                "as action content, keep agenda or attendance text in description as an exact continuous "
                "quote from the current message, because an unquoted description is dropped. "
                "Normalize every date and time a calendar action states: date is YYYY-MM-DD, start_time and "
                "end_time are 24-hour HH:MM, and duration_minutes is a whole number of minutes. Resolve words "
                "such as 今天 or tomorrow from reference_date, and look weekday phrases such as 下周三 up in the "
                "supplied calendar table instead of computing them. Decide the part of the day from the "
                "message's own context, such as 晚饭后十点 meaning 22:00; when the message does not determine it, "
                "use clarify rather than guessing. Return null for anything the message does not state, never "
                "invent a default duration, and never write a UTC offset, because Python attaches the timezone. "
                "end_time and duration_minutes are mutually exclusive. reference_time, when supplied, is "
                "the current local time: resolve a time stated relative to now, such as 半小时后 or in two "
                "hours, from reference_date and reference_time. When a message schedules something for a "
                "clock time today that is already earlier than reference_time, use clarify instead of moving "
                "it to another day. "
                "A meeting the user reports as already held or finished is not a future Calendar event. Keep "
                "its minutes with save_note, using the reported meeting text as content; do not create a "
                "Calendar event for it. Judge this from the message's own tense and wording relative to "
                "reference_date, never from an assumption about the user's schedule. "
                "When a message states a repeating schedule such as 每周一 or every Monday, set recurring to "
                "true instead of silently choosing one occurrence. When it states a timezone other than the "
                "supplied timezone, put that zone's IANA name in the action's timezone field and keep the times "
                "as stated, without converting them; otherwise leave timezone null. An update carries only what "
                "the user changed, and a relative change such as 提前一小时 is resolved from the candidate "
                "event's current start_at. A user-owned task or deadline "
                "that is not a scheduled event must use tracker_change, not Calendar. When such a task "
                "states a deadline, normalize it the same way as a calendar date: due_date is YYYY-MM-DD and "
                "due_time, if stated, is 24-hour HH:MM. A repeating deadline sets recurring to true. Leave "
                "due_at null, because Python sets it. Only a task may carry those fields, and a deadline is "
                "never invented. "
                "A tracker_change uses create_record with kind, content, and an exact source_quote; "
                "transition_record with only one candidate target_record_id, target_state completed or "
                "resolved, and an exact source_quote, and never kind, content, or rationale; "
                "supersede_record replacing one active candidate decision or constraint; or "
                "set_deadline with one candidate task target, the normalized deadline fields, and an "
                "exact source_quote, which is how a deadline is added or corrected on a task that "
                "already exists. Finishing or "
                "resolving an existing candidate record is transition_record, not a new record. "
                "Use calendar_update only when exactly one supplied candidate_events ID is the intended "
                "active event. Its target_event_id must be copied exactly from that candidate; never invent "
                "an ID. A correction may cite the earlier meeting message and the current correction message. "
                "If its target or replacement information is ambiguous, use clarify with a pending calendar "
                "action and one concise question; do not create a tracker record. "
                "Use calendar_delete when the user asks to remove, cancel, or clear exactly one supplied "
                "active candidate event. Its delete_event action carries only target_event_id and evidence; "
                "it never carries content, description, or any date, time, or duration field. If the "
                "request could name more than one candidate event, or names none of them, use clarify with a "
                "pending delete_event action instead of guessing which event to remove. "
                "One turn commits exactly one local action. When the current message clearly states a "
                "second, different local item as well — for example finished meeting minutes plus a separate "
                "future deadline — commit one of them, put the remaining item in deferred_action, and ask "
                "about it in one short follow_up_question. When one of the two items is a tracker change, "
                "commit that one and defer the other, because a deferred_action may only be create_event, "
                "save_idea, or save_note, must cite the current message, and is not executed until the user "
                "confirms it. Leave both fields null when the message states only one item; never invent a "
                "second item or split one item into two. "
                "Neither answer intent proposes a write, and CaptureFlow writes both answers itself. "
                "local_answer is a question about one or a few specific stored items: a Calendar event, an "
                "idea or note, or a single project record such as one task's deadline. Put the IDs of the "
                "supplied candidates the question is about in source_ids, or leave source_ids empty when none "
                "of them is it. project_answer is a question about the project as a whole: progress, outstanding "
                "work, blockers, deadlines across tasks, or why a decision changed; CaptureFlow gathers the "
                "project's records for it, so source_ids and reply are not needed. Save only an explicit idea or "
                "note with save_idea or save_note. Use reply_only for unsupported conversation. "
                "Every local action needs one to three EvidenceReference entries. Each entry must copy an "
                "exact quote from one supplied recent_messages message and its exact message_id; do not cite "
                "the same message twice. One entry must always quote current_message exactly, even when that "
                "message only confirms an earlier pending item; cite the earlier message as a second entry "
                "when the committed content comes from it. "
                "Any candidate ID you provide may reference only supplied candidates. "
                "Never invent a source quote, candidate ID, date, time, task, meeting, project "
                "fact, or external information. If context_truncated is true, do not imply older messages were seen."
            ),
            payload={
                "current_message": current_message,
                "recent_messages": recent_messages,
                "candidate_events": candidate_events,
                "candidate_saved_items": candidate_saved_items,
                "candidate_project_records": candidate_project_records,
                "reference_date": reference_date,
                "reference_time": reference_time,
                "timezone": timezone,
                "pending_clarification": pending_clarification,
                "context_truncated": context_truncated,
                "calendar": calendar or [],
            },
        )
        self._validate_contextual_references(
            decision,
            recent_messages,
            candidate_events,
            candidate_saved_items,
            candidate_project_records,
        )
        return decision

    def answer_project_question(
        self,
        question: dict[str, str],
        recent_messages: list[dict[str, str]],
        project_snapshot: dict,
        reference_date: str,
    ) -> ProjectAnswer:
        """Write a project-level answer from the supplied snapshot; Python checks its citations."""

        return self._complete(
            ProjectAnswer,
            system=(
                "You are CaptureFlow's project answer writer. Return JSON only. Answer the question about "
                "the project using only project_snapshot and recent_messages. The snapshot lists open records, "
                "recently completed or resolved records, and superseded decisions and constraints with their "
                "rationale and the record that replaced each, plus upcoming project Calendar events. For "
                "progress or outstanding work, use open and recently closed records. For why something "
                "changed, use the superseded record, its rationale, and the record that replaced it. Do not "
                "write dates or clock times in reply: CaptureFlow renders them from the stored records you "
                "cite. List in source_ids every record or event ID the reply relies on, copied exactly from "
                "the snapshot, at most six. If the snapshot does not contain the answer, say so plainly and "
                "leave source_ids empty. Never invent a record, decision, reason, deadline, or ID. Reply in "
                "the language of the question."
            ),
            payload={
                "question": question,
                "recent_messages": recent_messages,
                "project_snapshot": project_snapshot,
                "reference_date": reference_date,
            },
        )

    @staticmethod
    def _validate_contextual_references(
        decision: ConversationDecision,
        recent_messages: list[dict[str, str]],
        candidate_events: list[dict],
        candidate_saved_items: list[dict],
        candidate_project_records: list[dict],
    ) -> None:
        messages_by_id = {
            message["message_id"]: message["content"]
            for message in recent_messages
            if "message_id" in message and "content" in message
        }
        event_ids = {candidate["id"] for candidate in candidate_events if "id" in candidate}
        saved_item_ids = {
            candidate["id"] for candidate in candidate_saved_items if "id" in candidate
        }
        record_ids = {
            candidate["id"] for candidate in candidate_project_records if "id" in candidate
        }
        for action in (decision.action, decision.pending_action, decision.deferred_action):
            if action is None:
                continue
            if action.target_event_id is not None and action.target_event_id not in event_ids:
                raise InvalidStructuredOutput("The model selected a calendar event outside the supplied candidates.")
            for evidence in action.evidence:
                source_message = messages_by_id.get(evidence.message_id)
                if source_message is None:
                    raise InvalidStructuredOutput("The model cited a message outside the supplied recent context.")
                if _normalized_text(evidence.source_quote) not in _normalized_text(source_message):
                    raise InvalidStructuredOutput("The model cited text that is not present in its supplied message.")
        if any(
            change.target_record_id is not None and change.target_record_id not in record_ids
            for change in decision.project_changes
        ):
            raise InvalidStructuredOutput("The model selected a project record outside the supplied candidates.")
        if any(source_id not in event_ids | saved_item_ids | record_ids for source_id in decision.source_ids):
            raise InvalidStructuredOutput("The model cited a local answer source outside the supplied candidates.")

    def _complete(self, schema: type[Schema], system: str, payload: dict[str, Any]) -> Schema:
        prompt = json.dumps(
            {"schema": schema.model_json_schema(), "input": payload}, ensure_ascii=False
        )
        try:
            response = self.client.chat.completions.create(
                model=self.settings.model,
                timeout=self.settings.timeout_seconds,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content
            if not content:
                raise ModelError("The model returned an empty response.")
            return schema.model_validate_json(content)
        except ValidationError as error:
            raise InvalidStructuredOutput(
                f"The model returned invalid structured output: {describe_validation_error(error)}"
            ) from error
        except (IndexError, ValueError) as error:
            raise ModelError(f"The model returned invalid structured output: {error}") from error
        except Exception as error:
            if isinstance(error, ModelError):
                raise
            raise ModelError(f"Model request failed: {error}") from error


def describe_validation_error(error: ValidationError) -> str:
    """Name each failing field and rule without echoing the model's values back."""

    return "; ".join(
        f"{'.'.join(str(part) for part in detail['loc']) or '<root>'}: {detail['msg']}"
        for detail in error.errors(include_input=False, include_url=False)
    )


def _normalized_text(value: str) -> str:
    return " ".join(value.split())
