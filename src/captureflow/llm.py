"""The one model call per chat turn: an OpenAI-compatible adapter and its prompt."""

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .models import ConversationDecision, ProjectAnswer
from .read_agent import AgentFinalResponse, AgentModelTurn, AgentToolCall


class ModelError(RuntimeError):
    """A configured model could not supply a schema-valid response."""


class InvalidStructuredOutput(ModelError):
    """The model answered, but not with a decision that fits the schema and its inputs."""


class ContextualConversationModel(Protocol):
    def propose_conversation_decision(
        self,
        current_message: dict[str, str],
        recent_messages: list[dict],
        candidate_events: list[str],
        candidate_saved_items: list[str],
        candidate_project_records: list[str],
        reference_date: str,
        timezone: str,
        truncated_candidate_lists: list[str] | None = None,
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

    def next_read_agent_turn(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AgentModelTurn: ...


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
        recent_messages: list[dict],
        candidate_events: list[str],
        candidate_saved_items: list[str],
        candidate_project_records: list[str],
        reference_date: str,
        timezone: str,
        truncated_candidate_lists: list[str] | None = None,
        context_truncated: bool = False,
        calendar: list[dict[str, str]] | None = None,
        reference_time: str | None = None,
    ) -> ConversationDecision:
        """Make one untrusted, context-aware routing proposal; Python validates it downstream."""

        return self._complete(
            ConversationDecision,
            system=(
                "You are CaptureFlow's contextual conversation router. Return JSON only. You do not execute "
                "tools or write SQLite. Choose exactly one intent from current_message, recent_messages, the "
                "candidate lists, reference_date, reference_time, timezone, and calendar. This is not a general "
                "chatbot. "
                "What you see: recent_messages is the recent conversation. An older long message appears only "
                "as a reference with its ID and first characters; do not assume what the rest says. Each "
                "candidate is one line starting with its ID: candidate_events are Calendar events (ID, title, "
                "local time), candidate_project_records are this project's active tracker records (ID, kind, "
                "content, state, deadline), and candidate_saved_items are ideas and notes. "
                "truncated_candidate_lists names the lists cut at their bound, so an item missing from them may "
                "still exist. If context_truncated is true, older messages were not supplied. "
                "Stored state is current: the candidate lines show each item as it is now. Use the conversation "
                "only to understand what the user refers to; when an earlier message or reply disagrees with a "
                "candidate line, the candidate line is right. "
                "Only current_message can ask for a write. A request in an earlier message was already handled "
                "in its own turn; never carry it out again, unless current_message answers the question you "
                "asked about that request, which completes it. "
                "Finding a target: calendar_update, calendar_delete, and a tracker change that targets a record "
                "may name an existing item only when it is found. It is found when the current conversation "
                "mentioned it, or when current_message names the item itself, such as by its title. An "
                "item that appeared only in an older, different conversation is not found, and a demonstrative "
                "such as 这个会议 or that meeting in a conversation that never mentioned the item finds nothing. "
                "When the target is not found, or could be more than one candidate, use clarify: ask in reply "
                "which one the user means, in the user's language. When the user then names it, complete the "
                "original request from the conversation. "
                "clarify is a plain question in reply. Ask one when a calendar date, time, or target cannot be "
                "determined; the next turn continues from the conversation. "
                "A definite user statement that a meeting or event is still going to happen on an explicit "
                "date must use calendar_create, even when it is not phrased as a command. Use the event title "
                "as action content, and put agenda or attendance details in description, in your own short "
                "words. "
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
                "event's current time. Use calendar_update or calendar_delete only with one found candidate, "
                "copying its ID exactly into target_event_id; never invent an ID. A delete_event action carries "
                "only target_event_id and source_message_ids, never content, description, or any date, time, "
                "or duration. CaptureFlow asks the user to confirm a deletion before it happens. "
                "Every calendar, idea, or note action lists in source_message_ids the IDs of the user messages "
                "its content comes from, such as an earlier message that stated the event and the current "
                "message that answered your question. CaptureFlow always counts current_message. "
                "A user-owned task or deadline that is not a scheduled event must use tracker_change, not "
                "Calendar. When such a task states a deadline, normalize it the same way as a calendar date: "
                "due_date is YYYY-MM-DD and due_time, if stated, is 24-hour HH:MM. A repeating deadline sets "
                "recurring to true. Leave due_at null, because Python sets it. Only a task may carry those "
                "fields, and a deadline is never invented. "
                "A tracker_change uses create_record with kind and content; transition_record with one found "
                "candidate target_record_id and target_state completed or resolved, and never kind, content, "
                "or rationale; supersede_record replacing one found active candidate decision or constraint; "
                "or set_deadline with one found candidate task and the normalized deadline fields, which is how "
                "a deadline is added or corrected on a task that already exists. Finishing or resolving an "
                "existing candidate record is transition_record, not a new record. Every project change "
                "carries source_quote, a short exact quote of the words it comes from, and source_message_id, "
                "the ID of the user message that quote is in; a pasted text saved over several turns is quoted "
                "from the message that contains it. "
                "Pasted long text such as meeting minutes: save what you can in this turn with the intents above, "
                "and say in reply what is still unsaved, so the user can ask you to continue. "
                "One turn commits one local action or up to three tracker changes. When current_message clearly "
                "states a second, different item as well, commit one and ask about the other in one short "
                "follow_up_question; never invent a second item or split one item into two. "
                "Neither answer intent proposes a write, and CaptureFlow writes both answers itself. "
                "local_answer is a question about one or a few specific stored items: a Calendar event, an "
                "idea or note, or a single project record such as one task's deadline. Put the IDs of the "
                "candidates the question is about in source_ids, or leave source_ids empty when none of them is "
                "it. project_answer is a question about the project as a whole: progress, outstanding work, "
                "blockers, deadlines across tasks, or why a decision changed; CaptureFlow gathers the project's "
                "records for it, so source_ids and reply are not needed. Save only an explicit idea or note with "
                "save_idea or save_note. Use reply_only for unsupported conversation. "
                "Never invent a candidate ID, message ID, date, time, task, meeting, project fact, or external "
                "information."
            ),
            payload={
                "current_message": current_message,
                "recent_messages": recent_messages,
                "candidate_events": candidate_events,
                "candidate_saved_items": candidate_saved_items,
                "candidate_project_records": candidate_project_records,
                "truncated_candidate_lists": truncated_candidate_lists or [],
                "reference_date": reference_date,
                "reference_time": reference_time,
                "timezone": timezone,
                "context_truncated": context_truncated,
                "calendar": calendar or [],
            },
        )

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

    def next_read_agent_turn(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AgentModelTurn:
        """Normalize one OpenAI-compatible tool-calling response for the agent loop."""

        try:
            request: dict[str, Any] = {
                "model": self.settings.model,
                "timeout": self.settings.timeout_seconds,
                "response_format": {"type": "json_object"},
                "messages": messages,
                "tools": tools,
            }
            if (
                self.settings.model.casefold().startswith("qwen")
                and self.settings.base_url is not None
                and "aliyuncs.com" in self.settings.base_url.casefold()
            ):
                # DashScope Qwen 3.x defaults to thinking mode. Its function-calling
                # guide recommends disabling thinking, which also avoids having to
                # replay provider-specific reasoning_content on every tool round.
                request["extra_body"] = {"enable_thinking": False}
            response = self.client.chat.completions.create(**request)
            message = response.choices[0].message
            raw_calls = getattr(message, "tool_calls", None) or []
            if raw_calls:
                calls: list[AgentToolCall] = []
                for raw_call in raw_calls:
                    arguments = json.loads(raw_call.function.arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("tool arguments must be a JSON object")
                    calls.append(
                        AgentToolCall(
                            id=raw_call.id,
                            name=raw_call.function.name,
                            arguments=arguments,
                        )
                    )
                return AgentModelTurn(tool_calls=calls)
            if not message.content:
                raise ModelError("The model returned neither tool calls nor a final response.")
            return AgentModelTurn(
                final_response=AgentFinalResponse.model_validate_json(message.content)
            )
        except ValidationError as error:
            raise InvalidStructuredOutput(
                f"The model returned invalid agent output: {describe_validation_error(error)}"
            ) from error
        except (IndexError, ValueError, json.JSONDecodeError) as error:
            raise InvalidStructuredOutput("The model returned invalid agent tool arguments.") from error
        except Exception as error:
            if isinstance(error, ModelError):
                raise
            raise ModelError(f"Model request failed: {error}") from error

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
