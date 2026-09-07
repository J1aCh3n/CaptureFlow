import json
import os
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .models import CapturePlan, GroundingReport, OrganizerDraft


class ModelError(RuntimeError):
    """A configured model could not supply a schema-valid response."""


class PhaseTwoModel(Protocol):
    def plan(self, source_text: str) -> CapturePlan: ...

    def organize(
        self,
        source_text: str,
        plan: CapturePlan,
        feedback: GroundingReport | None = None,
    ) -> OrganizerDraft: ...

    def critique(self, source_text: str, draft: OrganizerDraft) -> GroundingReport: ...


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
                "implemented actions summarize and extract_task; do not emit dependencies or "
                "conditions. Use exact source spans. For a short, direct personal to-do list, "
                "choose extract_task only and omit summarize. Summaries must consolidate useful "
                "context rather than restate individual source sentences. This application cannot "
                "browse the web, search local businesses, or access current external information. "
                "For such a request, return no steps and one concise question asking whether to "
                "save it as a manual research task instead."
            ),
            payload={"source_text": source_text},
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
                "You are CaptureFlow's Organizer. Return JSON only. Produce claims and "
                "tasks only for actions in the supplied plan. Every item must include an "
                "When the supplied plan has no summarize step, return summary_claims as an "
                "empty list even if the source includes useful context. "
                "exact, continuous quote from the source that is sufficient by itself to support "
                "the item. When a task combines a problem, requested action, or deadline from "
                "different clauses, quote the continuous source span containing all required "
                "clauses; otherwise preserve only the scope stated in the quote. If grounding "
                "feedback is supplied, correct every cited issue before returning. Do not infer "
                "missing facts or dates. Do not create "
                "a summary that merely repeats a task or one source sentence. Preserve source "
                "wording exactly: never silently correct a possible typo. If a term seems "
                "materially ambiguous, add a review_note asking the user to confirm it and quote "
                "the original term."
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
                "You are CaptureFlow's grounding critic. Return JSON only. Mark pass only "
                "when every paraphrased claim and task is entailed by its cited quote. "
                "For a task that combines details from separate clauses, its quote must contain "
                "all details needed to entail the task. "
                "Also request revision when a summary merely restates one source sentence or a "
                "task without adding useful consolidation. Review notes must preserve original "
                "source wording. Otherwise request revise with item-specific feedback."
            ),
            payload={"source_text": source_text, "draft": draft.model_dump(mode="json")},
        )

    def _complete(self, schema: type[Schema], system: str, payload: dict[str, Any]) -> Schema:
        prompt = json.dumps(
            {"schema": schema.model_json_schema(), "input": payload}, ensure_ascii=False
        )
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
        except (IndexError, ValidationError, ValueError) as error:
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
    ) -> None:
        self.plans = plans
        self.drafts = drafts
        self.reports = reports
        self.calls: list[str] = []

    def plan(self, source_text: str) -> CapturePlan:
        self.calls.append("plan")
        return CapturePlan.model_validate(self._next(self.plans, "plan"))

    def organize(
        self,
        source_text: str,
        plan: CapturePlan,
        feedback: GroundingReport | None = None,
    ) -> OrganizerDraft:
        self.calls.append("organize")
        return OrganizerDraft.model_validate(self._next(self.drafts, "organize"))

    def critique(self, source_text: str, draft: OrganizerDraft) -> GroundingReport:
        self.calls.append("critique")
        return GroundingReport.model_validate(self._next(self.reports, "critique"))

    @staticmethod
    def _next(values: list[Schema | dict[str, Any]], stage: str) -> Schema | dict[str, Any]:
        if not values:
            raise ModelError(f"No scripted {stage} response remains.")
        return values.pop(0)
