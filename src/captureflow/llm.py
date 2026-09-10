import json
import os
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .models import (
    CapturePlan,
    ConversationDecision,
    GroundingReport,
    LocalContextAnswerDecision,
    MemoryAgentDraft,
    MemoryItem,
    OrganizerDraft,
    Project,
    ProjectAnswerDecision,
    ProjectConversationDecision,
    ProjectTrackerDecision,
    TurnDecision,
)


class ModelError(RuntimeError):
    """A configured model could not supply a schema-valid response."""


class InvalidStructuredOutput(ModelError):
    """A model response can be shown to the single bounded repair pass."""

    def __init__(self, message: str, raw_output: str) -> None:
        super().__init__(message)
        self.raw_output = raw_output


class PhaseTwoModel(Protocol):
    def plan(self, source_text: str) -> CapturePlan: ...

    def repair_plan(
        self, source_text: str, invalid_plan: CapturePlan | str, validation_error: str
    ) -> CapturePlan: ...

    def organize(
        self,
        source_text: str,
        plan: CapturePlan,
        feedback: GroundingReport | None = None,
    ) -> OrganizerDraft: ...

    def critique(self, source_text: str, draft: OrganizerDraft) -> GroundingReport: ...


class MemoryAgentModel(Protocol):
    def propose_memories(
        self, source_text: str, retrieved_memories: list[MemoryItem]
    ) -> MemoryAgentDraft: ...


class PhaseThreeModel(PhaseTwoModel, MemoryAgentModel, Protocol):
    """The bounded Phase 2 and Phase 3 model surface."""


class TurnRouterModel(Protocol):
    def route_turn(
        self,
        user_message: str,
        reference_date: str,
        timezone: str,
        clarification_context: dict | None = None,
    ) -> TurnDecision: ...

    def repair_turn(
        self,
        user_message: str,
        reference_date: str,
        timezone: str,
        invalid_decision: TurnDecision | str,
        validation_error: str,
        clarification_context: dict | None = None,
    ) -> TurnDecision: ...


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
    ) -> ConversationDecision: ...


class ProjectConversationModel(Protocol):
    def respond_to_project_turn(
        self,
        project: Project,
        recent_messages: list[dict[str, str]],
        context_truncated: bool,
    ) -> ProjectConversationDecision: ...


class ProjectTrackerModel(Protocol):
    def propose_project_changes(
        self,
        project: Project,
        recent_messages: list[dict[str, str]],
        candidate_records: list[dict[str, str]],
        context_truncated: bool,
    ) -> ProjectTrackerDecision: ...


class ProjectRetrievalModel(Protocol):
    def answer_project_query(
        self,
        project: Project,
        recent_messages: list[dict[str, str]],
        retrieved_records: list[dict],
        answer_kind: str,
        context_truncated: bool,
    ) -> ProjectAnswerDecision: ...


class LocalContextModel(Protocol):
    def answer_local_context(
        self,
        user_message: str,
        runtime_context: dict[str, str | None],
        events: list[dict],
        saved_items: list[dict],
        project_records: list[dict],
    ) -> LocalContextAnswerDecision: ...

    def repair_local_context_answer(
        self,
        user_message: str,
        runtime_context: dict[str, str | None],
        events: list[dict],
        saved_items: list[dict],
        project_records: list[dict],
        invalid_decision: LocalContextAnswerDecision | str,
        validation_error: str,
    ) -> LocalContextAnswerDecision: ...


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
            raise ModelError("Set CAPTUREFLOW_API_KEY and CAPTUREFLOW_MODEL before analyzing.")
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

    def plan(self, source_text: str) -> CapturePlan:
        return self._complete(
            CapturePlan,
            system=(
                "You are CaptureFlow's Coordinator. Return JSON only. Plan only the "
                "implemented actions summarize, extract_task, and inspect_memory; do not emit "
                "dependencies or conditions. Select inspect_memory for durable user-specific "
                "preferences, decisions, facts, or ideas that could be useful in a later "
                "conversation. This includes explicit first-person statements even when no "
                "related memory exists yet. Select extract_task only for an actual requested, "
                "assigned, promised, or otherwise actionable next step owned by the user, the "
                "recipient, or their explicitly identified team. A reported obligation of a "
                "third party in a news article, report, quote, or other source is context to "
                "summarize, not the user's task; for example, 'The transportation office must "
                "publish a report' is not a user task. A statement of a "
                "preference, a completed decision, or an idea is not by itself a task. Select "
                "both actions only when the source independently contains durable information "
                "and a next step. Each source_span must be copied verbatim from the source, "
                "apart from whitespace normalization. For a message with one explicit action "
                "request, choose extract_task only; explanatory context does not by itself need "
                "a summary. Examples: 'I prefer concise meeting notes', 'I decided to use "
                "PostgreSQL', 'I have an idea for a meeting assistant', and 'My meeting notes "
                "should be concise' require inspect_memory, not extract_task. 'Please pay the "
                "utility bill' requires extract_task. Duplicate checking happens downstream, so "
                "never omit inspect_memory merely because the information may already be saved. "
                "A subject line, a label, or an instruction not to retain material is never a "
                "task. A credential disclosure must use inspect_memory only so the safety layer "
                "can prevent persistence; never create a task from it. "
                "Never return an empty step list for an executable local input. "
                "Summaries must consolidate useful context rather than restate individual source "
                "sentences. This application cannot "
                "browse the web, search local businesses, or access current external information. "
                "For such a request, return no steps and one concise question asking whether to "
                "save it as a manual research task instead."
            ),
            payload={"source_text": source_text},
        )

    def route_turn(
        self,
        user_message: str,
        reference_date: str,
        timezone: str,
        clarification_context: dict | None = None,
    ) -> TurnDecision:
        return self._complete(
            TurnDecision,
            system=(
                "You are CaptureFlow's V2 TurnRouter. Return JSON only. This is not a "
                "general chatbot. Choose execute only for explicit local calendar events, "
                "ideas, or notes. Choose reply_only for unsupported/general conversation "
                "and keep the reply short. A definite first-person or subjectless statement "
                "that the user has a scheduled event is an explicit local calendar event even "
                "when it is not phrased as a command. Use one create_event action for that "
                "event; copy surrounding agenda or attendance details into description and keep "
                "them in the same source_quote rather than creating separate notes. Description "
                "must be an exact continuous quote from the user message. An execute action needs "
                "content and an exact, "
                "continuous source_quote copied from the user message except normalized "
                "whitespace. create_event also needs an explicit date_expression copied "
                "from the message; optional time_expression must be an explicit HH:MM, "
                "H:MM AM/PM, or supported Chinese phrase such as 下午两点, 上午九点半, "
                "or 晚上八点. For an explicit time range, time_expression is its start and "
                "end_time_expression is its end; otherwise end_time_expression is null. Never "
                "invent a default duration. An event lacking a date must choose clarify with no "
                "actions, one concise clarification_question, and pending_action describing "
                "the event with no date_expression. save_idea and save_note never include "
                "description, date_expression, time_expression, or end_time_expression. Do not "
                "make more than three independent "
                "actions, do not infer details, and never include private reasoning."
            ),
            payload={
                "user_message": user_message,
                "reference_date": reference_date,
                "timezone": timezone,
                "clarification_context": clarification_context,
            },
        )

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
    ) -> ConversationDecision:
        """Make one untrusted, context-aware routing proposal; storage remains downstream."""

        decision = self._complete(
            ConversationDecision,
            system=(
                "You are CaptureFlow's Phase 10 contextual conversation router. Return JSON only. "
                "You do not execute tools or write SQLite. Choose exactly one intent and use only "
                "the supplied current_message, recent_messages, candidates, reference_date, timezone, "
                "and pending clarification. This is not a general chatbot. "
                "A definite user statement that a meeting or event is scheduled with an explicit date "
                "must use calendar_create, even when it is not phrased as a command. Use the event title "
                "as action content, keep agenda or attendance text in description, and copy explicit date "
                "and time expressions rather than resolving them yourself. A user-owned task or deadline "
                "that is not a scheduled event must use tracker_change, not Calendar. "
                "Use calendar_update only when exactly one supplied candidate_events ID is the intended "
                "active event. Its target_event_id must be copied exactly from that candidate; never invent "
                "an ID. A correction may cite the earlier meeting message and the current correction message. "
                "If its target or replacement information is ambiguous, use clarify with a pending calendar "
                "action and one concise question; do not create a tracker record. "
                "A meeting question is local_answer and must not propose a write. Project status/history "
                "questions are project_answer and must not propose a write. Save only an explicit idea or "
                "note with save_idea or save_note. Use reply_only for unsupported conversation. "
                "Every local action needs one to three EvidenceReference entries. Each entry must copy an "
                "exact quote from one supplied recent_messages message and its exact message_id; do not cite "
                "the same message twice. Any candidate ID you provide may reference only supplied candidates. "
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
                "timezone": timezone,
                "pending_clarification": pending_clarification,
                "context_truncated": context_truncated,
            },
        )
        self._validate_contextual_references(
            decision,
            recent_messages,
            candidate_events,
            candidate_project_records,
        )
        return decision

    def repair_turn(
        self,
        user_message: str,
        reference_date: str,
        timezone: str,
        invalid_decision: TurnDecision | str,
        validation_error: str,
        clarification_context: dict | None = None,
    ) -> TurnDecision:
        invalid_payload = (
            invalid_decision.model_dump(mode="json")
            if isinstance(invalid_decision, TurnDecision)
            else invalid_decision
        )
        return self._complete(
            TurnDecision,
            system=(
                "You are CaptureFlow's V2 TurnRouter repair pass. Return corrected JSON only. "
                "Use only execute, clarify, or reply_only. Preserve exact source quotes and "
                "never invent content, dates, times, or unsupported actions."
            ),
            payload={
                "user_message": user_message,
                "reference_date": reference_date,
                "timezone": timezone,
                "clarification_context": clarification_context,
                "invalid_decision": invalid_payload,
                "validation_error": validation_error,
            },
        )

    def respond_to_project_turn(
        self,
        project: Project,
        recent_messages: list[dict[str, str]],
        context_truncated: bool,
    ) -> ProjectConversationDecision:
        return self._complete(
            ProjectConversationDecision,
            system=(
                "You are CaptureFlow's Phase 5 project conversation responder. Return JSON only. "
                "This is not a general chatbot and you cannot create, update, complete, delete, "
                "or otherwise change project tracker records in this phase. Use only the supplied "
                "project and recent conversation messages. Classify the current user turn as one of "
                "new_subject, follow_up, correction, cancellation, clarification_answer, or clarify. "
                "Use clarify only when a necessary referent is absent or ambiguous; then ask one "
                "concise clarification_question. For every other kind, clarification_question must "
                "be null. Reply concisely about the ongoing project discussion, do not claim memory "
                "outside the supplied messages, do not invent tracker state, and do not propose tools "
                "or actions. If context_truncated is true, do not imply that you saw older history."
            ),
            payload={
                "project": {
                    "id": project.id,
                    "name": project.name,
                    "description": project.description,
                },
                "recent_messages": recent_messages,
                "context_truncated": context_truncated,
            },
        )

    def propose_project_changes(
        self,
        project: Project,
        recent_messages: list[dict[str, str]],
        candidate_records: list[dict[str, str]],
        context_truncated: bool,
    ) -> ProjectTrackerDecision:
        return self._complete(
            ProjectTrackerDecision,
            system=(
                "You are CaptureFlow's Phase 6 project tracker proposer. Return JSON only. "
                "You do not write to SQLite or execute tools; you only propose at most three "
                "changes for deterministic local validation. Use the supplied project, recent "
                "conversation messages, and candidate_records only. A no-change turn has an empty "
                "changes list. Allowed changes are: create_record with kind, content, and an exact "
                "source_quote from the current user message; transition_record with one exact "
                "candidate target_record_id and target_state completed or resolved; or "
                "supersede_record with an exact active candidate decision/constraint ID, matching "
                "kind, replacement content, and exact source_quote. Never invent a target ID, "
                "record state, project fact, rationale, or source quote. Do not propose a tracker "
                "change for small talk, questions, acknowledgements, uncertain referents, or a "
                "credential. Classify the turn as new_subject, follow_up, correction, cancellation, "
                "clarification_answer, or clarify. A clarify turn needs one concise question and "
                "must have no changes. Reply concisely; do not claim that a proposed change has "
                "already been applied. If context_truncated is true, do not imply older history was seen."
            ),
            payload={
                "project": {
                    "id": project.id,
                    "name": project.name,
                    "description": project.description,
                },
                "recent_messages": recent_messages,
                "candidate_records": candidate_records,
                "context_truncated": context_truncated,
            },
        )

    def answer_project_query(
        self,
        project: Project,
        recent_messages: list[dict[str, str]],
        retrieved_records: list[dict],
        answer_kind: str,
        context_truncated: bool,
    ) -> ProjectAnswerDecision:
        return self._complete(
            ProjectAnswerDecision,
            system=(
                "You are CaptureFlow's Phase 7 project retrieval responder. Return JSON only. "
                "Answer only from the supplied project, recent conversation messages, and "
                "retrieved_records. This is not a general chatbot: do not create, edit, complete, "
                "or propose tracker changes. The answer_kind is current_status or history. State "
                "only facts supported by retrieved_records and include one to six exact "
                "supporting_record_ids from that list. Do not invent IDs, sources, project facts, "
                "or external information. For current_status, do not treat a superseded record as "
                "current. For history, explain the historical record when present. If the records "
                "are insufficient, say so rather than filling the gap. If context_truncated is true, "
                "do not imply that older conversation messages were seen."
            ),
            payload={
                "project": {
                    "id": project.id,
                    "name": project.name,
                    "description": project.description,
                },
                "recent_messages": recent_messages,
                "retrieved_records": retrieved_records,
                "answer_kind": answer_kind,
                "context_truncated": context_truncated,
            },
        )

    def answer_local_context(
        self,
        user_message: str,
        runtime_context: dict[str, str | None],
        events: list[dict],
        saved_items: list[dict],
        project_records: list[dict],
    ) -> LocalContextAnswerDecision:
        return self._complete(
            LocalContextAnswerDecision,
            system=(
                "You are CaptureFlow's Phase 9.1 local-context responder. Return JSON only. "
                "Answer only from the supplied local Calendar events, same-project tracker records, "
                "and saved Notes/Ideas. "
                "A Calendar event description is stored source-grounded agenda or attendance "
                "context for that event. "
                "This is not a general chatbot: you cannot access an external calendar, hidden "
                "conversation history, tools, or other records, and you must not create or change "
                "any local record. Select IDs only from the supplied records. Every summary point "
                "and every preparation suggestion must cite one or more selected source IDs. "
                "Only a Calendar event establishes that an activity is scheduled or establishes "
                "its date or time. A project task requesting creation of a calendar event does not "
                "mean that event exists. Project decisions may establish meeting purpose or agenda. "
                "Preparation suggestions are optional inferences, not stored tasks or confirmed "
                "requirements; do not invent an explicit preparation task. Do not state a date or "
                "time in a summary point: CaptureFlow renders exact Calendar time locally after "
                "validation. The supplied records can establish only part of a multi-part question. "
                "When any supplied record supports any requested detail, use answer, select that "
                "record, state only the supported detail, and list each unsupported requested detail "
                "in missing_information. A stored meeting purpose, content, or process establishes "
                "agenda coverage for this purpose; do not list agenda as missing in that case. Use "
                "insufficient_context only when none of the supplied records supports any requested "
                "detail. If multiple Calendar events plausibly match "
                "a singular question, use clarify with one concise question instead of arbitrarily "
                "selecting one."
            ),
            payload={
                "user_message": user_message,
                "runtime_context": runtime_context,
                "events": events,
                "saved_items": saved_items,
                "project_records": project_records,
            },
        )

    def repair_local_context_answer(
        self,
        user_message: str,
        runtime_context: dict[str, str | None],
        events: list[dict],
        saved_items: list[dict],
        project_records: list[dict],
        invalid_decision: LocalContextAnswerDecision | str,
        validation_error: str,
    ) -> LocalContextAnswerDecision:
        invalid_payload = (
            invalid_decision.model_dump(mode="json")
            if isinstance(invalid_decision, LocalContextAnswerDecision)
            else invalid_decision
        )
        return self._complete(
            LocalContextAnswerDecision,
            system=(
                "You are CaptureFlow's Phase 9.1 local-context repair pass. Return corrected "
                "JSON only. Use only supplied source IDs. Every answer point and suggestion must "
                "cite selected source IDs. When any supplied record supports part of the question, "
                "return answer with that supported part and list only the remaining gaps in "
                "missing_information; do not discard selected evidence as insufficient_context. "
                "Only Calendar events establish scheduled activities, dates, or times; a project "
                "task to create an event is not an event. Never invent a time or an unrecorded "
                "preparation task, and use clarify rather "
                "than arbitrarily selecting among multiple plausible events."
            ),
            payload={
                "user_message": user_message,
                "runtime_context": runtime_context,
                "events": events,
                "saved_items": saved_items,
                "project_records": project_records,
                "invalid_decision": invalid_payload,
                "validation_error": validation_error,
            },
        )

    def propose_memories(
        self, source_text: str, retrieved_memories: list[MemoryItem]
    ) -> MemoryAgentDraft:
        return self._complete(
            MemoryAgentDraft,
            system=(
                "You are CaptureFlow's Memory Agent. Return JSON only. Propose at most five "
                "compact long-term memories only when the source contains durable user-specific "
                "information. Every proposal needs an exact source quote. Mark is_explicit false "
                "for any inference. Use related_memory_ids only from the provided retrieved set "
                "when a proposal conflicts with it. Never include secrets, credentials, or full "
                "captures. An explicit request to create, send, or complete a task is not a "
                "durable decision or idea. Return an empty proposal list when memory is not useful."
            ),
            payload={
                "source_text": source_text,
                "retrieved_memories": [item.model_dump(mode="json") for item in retrieved_memories],
            },
        )

    def repair_plan(
        self, source_text: str, invalid_plan: CapturePlan | str, validation_error: str
    ) -> CapturePlan:
        invalid_payload = (
            invalid_plan.model_dump(mode="json")
            if isinstance(invalid_plan, CapturePlan)
            else invalid_plan
        )
        return self._complete(
            CapturePlan,
            system=(
                "You are CaptureFlow's Coordinator repair pass. Return JSON only. Correct the "
                "invalid plan using only summarize, extract_task, and inspect_memory. You may "
                "add, remove, or replace steps as needed to satisfy the supplied validation "
                "error, but never invent source content. Extract tasks only when the action is "
                "owned by the user, recipient, or their explicitly identified team; reported "
                "third-party obligations belong in summaries, not the user's task list. Durable user-specific preferences, "
                "decisions, facts, and ideas require inspect_memory and are not tasks unless the "
                "source also contains an actionable next step. Never return an empty step list "
                "for an executable local input. Every source_span must be copied verbatim from "
                "the supplied source, apart from whitespace normalization."
            ),
            payload={
                "source_text": source_text,
                "invalid_plan": invalid_payload,
                "validation_error": validation_error,
            },
        )

    def organize(
        self,
        source_text: str,
        plan: CapturePlan,
        feedback: GroundingReport | None = None,
    ) -> OrganizerDraft:
        return self._complete(
            OrganizerDraft,
            system=(
                "You are CaptureFlow's Organizer. Return JSON only. Produce output only for "
                "actions in the supplied plan. Every claim and task needs one exact, continuous "
                "source quote sufficient to support it. If there is no summarize step, return an "
                "empty summary_claims list. If a task uses details from multiple clauses, quote "
                "the continuous span containing all of them; otherwise preserve only the quoted "
                "scope. Do not infer facts or dates. Set due_expression only for an explicit "
                "calendar expression, never for urgency wording such as 'at your earliest "
                "convenience' or 'ASAP'; retain relevant urgency in task text. For a long article, "
                "return one or two non-overlapping, useful summary claims and omit weak details. "
                "For an idea or product proposal, make the first claim synthesize its objective, "
                "input-to-outcome flow, and any explicit risk-handling principle; an optional "
                "second claim may summarize the proposed technical pipeline. Treat suggested "
                "technologies as options, not committed implementation, and do not turn an idea "
                "into a task unless the source assigns an action. "
                "If grounding feedback is supplied, correct every cited issue. Preserve source "
                "wording: never silently correct a possible typo. For a materially ambiguous "
                "term, add a review_note quoting the original term."
            ),
            payload={
                "source_text": source_text,
                "plan": plan.model_dump(mode="json"),
                "grounding_feedback": None if feedback is None else feedback.model_dump(mode="json"),
            },
        )

    def critique(self, source_text: str, draft: OrganizerDraft) -> GroundingReport:
        return self._complete(
            GroundingReport,
            system=(
                "You are CaptureFlow's source-grounding critic. Return JSON only. Check only "
                "whether every claim, task, and review note is entailed by its cited source "
                "quote without adding facts or changing meaning. A concise paraphrase or a "
                "restatement passes when it is entailed. Do not judge usefulness, depth, style, "
                "brevity, or whether a summary combines multiple sentences. "
                "Date resolution is a deterministic typed-tool result and is not a source "
                "claim: assess whether the original due_expression is cited, but never require "
                "the resolved calendar date or its year to occur in the source quote. "
                "For a task that combines details from separate clauses, its quote must contain "
                "all details needed to entail the task. Review notes must preserve original "
                "source wording. Request revise only for unsupported content and return "
                "item-specific feedback."
            ),
            payload={"source_text": source_text, "draft": draft.model_dump(mode="json")},
        )

    @staticmethod
    def _validate_contextual_references(
        decision: ConversationDecision,
        recent_messages: list[dict[str, str]],
        candidate_events: list[dict],
        candidate_project_records: list[dict],
    ) -> None:
        messages_by_id = {
            message["message_id"]: message["content"]
            for message in recent_messages
            if "message_id" in message and "content" in message
        }
        event_ids = {candidate["id"] for candidate in candidate_events if "id" in candidate}
        record_ids = {
            candidate["id"] for candidate in candidate_project_records if "id" in candidate
        }
        for action in (decision.action, decision.pending_action):
            if action is None:
                continue
            if action.target_event_id is not None and action.target_event_id not in event_ids:
                raise InvalidStructuredOutput(
                    "The model selected a calendar event outside the supplied candidates.",
                    decision.model_dump_json(),
                )
            for evidence in action.evidence:
                source_message = messages_by_id.get(evidence.message_id)
                if source_message is None:
                    raise InvalidStructuredOutput(
                        "The model cited a message outside the supplied recent context.",
                        decision.model_dump_json(),
                    )
                if _normalized_text(evidence.source_quote) not in _normalized_text(source_message):
                    raise InvalidStructuredOutput(
                        "The model cited text that is not present in its supplied message.",
                        decision.model_dump_json(),
                    )
        if any(
            change.target_record_id is not None and change.target_record_id not in record_ids
            for change in decision.project_changes
        ):
            raise InvalidStructuredOutput(
                "The model selected a project record outside the supplied candidates.",
                decision.model_dump_json(),
            )

    def _complete(self, schema: type[Schema], system: str, payload: dict[str, Any]) -> Schema:
        prompt = json.dumps(
            {"schema": schema.model_json_schema(), "input": payload}, ensure_ascii=False
        )
        content: str | None = None
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
                f"The model returned invalid structured output: {error}", content or ""
            ) from error
        except (IndexError, ValueError) as error:
            raise ModelError(f"The model returned invalid structured output: {error}") from error
        except Exception as error:
            if isinstance(error, ModelError):
                raise
            raise ModelError(f"Model request failed: {error}") from error


def _normalized_text(value: str) -> str:
    return " ".join(value.split())


class ScriptedModel:
    """Test double: supplies declared model outputs without a live API key."""

    def __init__(
        self,
        plans: list[CapturePlan | dict[str, Any]],
        drafts: list[OrganizerDraft | dict[str, Any]],
        reports: list[GroundingReport | dict[str, Any]],
        repair_plans: list[CapturePlan | dict[str, Any]] | None = None,
        memory_drafts: list[MemoryAgentDraft | dict[str, Any]] | None = None,
    ) -> None:
        self.plans = plans
        self.drafts = drafts
        self.reports = reports
        self.repair_plans = repair_plans or []
        self.memory_drafts = memory_drafts or []
        self.calls: list[str] = []
        self.critique_drafts: list[OrganizerDraft] = []

    def plan(self, source_text: str) -> CapturePlan:
        self.calls.append("plan")
        candidate = self._next(self.plans, "plan")
        try:
            return CapturePlan.model_validate(candidate)
        except ValidationError as error:
            raw_output = json.dumps(candidate, ensure_ascii=False, default=str)
            raise InvalidStructuredOutput(
                f"The model returned invalid structured output: {error}", raw_output
            ) from error

    def organize(
        self,
        source_text: str,
        plan: CapturePlan,
        feedback: GroundingReport | None = None,
    ) -> OrganizerDraft:
        self.calls.append("organize")
        return OrganizerDraft.model_validate(self._next(self.drafts, "organize"))

    def repair_plan(
        self, source_text: str, invalid_plan: CapturePlan | str, validation_error: str
    ) -> CapturePlan:
        self.calls.append("repair_plan")
        return CapturePlan.model_validate(self._next(self.repair_plans, "plan repair"))

    def critique(self, source_text: str, draft: OrganizerDraft) -> GroundingReport:
        self.calls.append("critique")
        self.critique_drafts.append(draft)
        return GroundingReport.model_validate(self._next(self.reports, "critique"))

    def propose_memories(
        self, source_text: str, retrieved_memories: list[MemoryItem]
    ) -> MemoryAgentDraft:
        self.calls.append("memory")
        return MemoryAgentDraft.model_validate(self._next(self.memory_drafts, "memory"))

    @staticmethod
    def _next(values: list[Schema | dict[str, Any]], stage: str) -> Schema | dict[str, Any]:
        if not values:
            raise ModelError(f"No scripted {stage} response remains.")
        return values.pop(0)
