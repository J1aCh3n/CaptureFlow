import json
import os
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .models import CapturePlan, GroundingReport, MemoryAgentDraft, MemoryItem, OrganizerDraft


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


@dataclass(frozen=True)
class ModelSettings:
    api_key: str
    model: str
    base_url: str | None = None

    @classmethod
    def from_environment(cls) -> "ModelSettings":
        api_key = os.getenv("CAPTUREFLOW_API_KEY")
        model = os.getenv("CAPTUREFLOW_MODEL")
        if not api_key or not model:
            raise ModelError("Set CAPTUREFLOW_API_KEY and CAPTUREFLOW_MODEL before analyzing.")
        return cls(api_key=api_key, model=model, base_url=os.getenv("CAPTUREFLOW_BASE_URL"))


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
                "assigned, promised, or otherwise actionable next step. A statement of a "
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
                "Never return an empty step list for an executable local input. "
                "Summaries must consolidate useful context rather than restate individual source "
                "sentences. This application cannot "
                "browse the web, search local businesses, or access current external information. "
                "For such a request, return no steps and one concise question asking whether to "
                "save it as a manual research task instead."
            ),
            payload={"source_text": source_text},
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
                "captures. Return an empty proposal list when memory is not useful."
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
                "error, but never invent source content. Durable user-specific preferences, "
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

    def _complete(self, schema: type[Schema], system: str, payload: dict[str, Any]) -> Schema:
        prompt = json.dumps(
            {"schema": schema.model_json_schema(), "input": payload}, ensure_ascii=False
        )
        content: str | None = None
        try:
            response = self.client.chat.completions.create(
                model=self.settings.model,
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
