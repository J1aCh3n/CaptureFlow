from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .llm import MemoryAgentModel, ModelError, PhaseThreeModel
from .memory import contains_credential_disclosure
from .memory_service import MemoryService
from .models import (
    CapturePlan,
    CaptureRun,
    MemoryAgentDraft,
    MemoryItem,
    RunError,
    TaskItem,
    WorkflowCheckpoint,
)
from .workflow import PhaseTwoWorkflow


class MemoryGraphState(TypedDict, total=False):
    run_id: str
    capture_id: str
    source_text: str
    retrieved: list[MemoryItem]
    draft: MemoryAgentDraft
    error: str


class MemoryAgentWorkflow:
    """Bounded LangGraph subgraph for retrieval and proposal preparation."""

    def __init__(self, model: MemoryAgentModel, service: MemoryService) -> None:
        self.model = model
        self.service = service
        graph = StateGraph(MemoryGraphState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("propose", self._propose)
        graph.add_edge(START, "retrieve")
        graph.add_conditional_edges("retrieve", self._after_retrieve, {"propose": "propose", "end": END})
        graph.add_edge("propose", END)
        self.graph = graph.compile()

    def run(self, run_id: str, capture_id: str, source_text: str) -> MemoryGraphState:
        return self.graph.invoke(
            {"run_id": run_id, "capture_id": capture_id, "source_text": source_text}
        )

    def _retrieve(self, state: MemoryGraphState) -> MemoryGraphState:
        if contains_credential_disclosure(state["source_text"]):
            return {"retrieved": [], "draft": MemoryAgentDraft()}
        return {"retrieved": self.service.retrieve(state["source_text"])}

    @staticmethod
    def _after_retrieve(state: MemoryGraphState) -> str:
        return "propose" if state["source_text"].strip() and "draft" not in state else "end"

    def _propose(self, state: MemoryGraphState) -> MemoryGraphState:
        try:
            return {"draft": self.model.propose_memories(state["source_text"], state["retrieved"])}
        except (ModelError, ValueError) as error:
            return {"error": str(error)}

class PhaseThreeWorkflow:
    """Runs Phase 2 only as needed, then enters the memory subgraph when planned."""

    def __init__(self, model: PhaseThreeModel, service: MemoryService) -> None:
        self.phase_two = PhaseTwoWorkflow(model)
        self.memory_graph = MemoryAgentWorkflow(model, service)
        self.service = service

    def run(self, **kwargs) -> CaptureRun:
        result = self.phase_two.run(**kwargs)
        if result.status in {"failed", "grounding_failed"} or not result.plan:
            return result
        draft: MemoryAgentDraft | None = None
        retrieved: list[MemoryItem] = []
        if any(step.action == "inspect_memory" for step in result.plan.steps):
            if result.model_calls >= 6:
                status = "completed_with_omissions" if result.status == "completed" else result.status
                result = result.model_copy(
                    update={
                        "status": status,
                        "warnings": [
                            *result.warnings,
                            "Memory inspection was skipped because the six-call run limit was reached.",
                        ],
                    }
                )
            else:
                state = self.memory_graph.run(result.id, result.capture_id, kwargs["source_text"])
                retrieved = state.get("retrieved", [])
                if state.get("error"):
                    result = result.model_copy(
                        update={
                            "status": "completed_with_omissions",
                            "warnings": [*result.warnings, "Memory proposal failed; no memory was written."],
                            "errors": [*result.errors, RunError(stage="memory", message=state["error"])],
                            "model_calls": result.model_calls + 1,
                            "retrieved_memory_ids": [item.id for item in retrieved],
                        }
                    )
                else:
                    draft = self._remove_task_based_proposals(state["draft"], result.plan)
                    result = result.model_copy(
                        update={
                            "model_calls": result.model_calls
                            + (0 if contains_credential_disclosure(kwargs["source_text"]) else 1),
                            "retrieved_memory_ids": [item.id for item in retrieved],
                        }
                    )
        return self._prepare_and_commit(result, kwargs["source_text"], draft, retrieved)

    @staticmethod
    def _remove_task_based_proposals(
        draft: MemoryAgentDraft,
        plan: CapturePlan,
    ) -> MemoryAgentDraft:
        task_spans = [
            _normalize(step.source_span)
            for step in plan.steps
            if step.action == "extract_task"
        ]
        if not task_spans:
            return draft
        return draft.model_copy(
            update={
                "proposals": [
                    proposal
                    for proposal in draft.proposals
                    if not any(
                        _normalize(proposal.source_quote) in task_span
                        or task_span in _normalize(proposal.source_quote)
                        for task_span in task_spans
                    )
                ]
            }
        )

    def resume(self, run_id: str) -> CaptureRun:
        """Resume a prepared Phase 3 commit without calling the model again."""

        checkpoint = self.service.store.get_checkpoint(run_id)
        if checkpoint.state == "committed":
            return CaptureRun.model_validate(checkpoint.payload["result"])
        return self._commit_prepared(checkpoint)

    def _prepare_and_commit(
        self,
        result: CaptureRun,
        source_text: str,
        draft: MemoryAgentDraft | None,
        retrieved: list[MemoryItem],
    ) -> CaptureRun:
        tasks = []
        if result.output and result.status in {"completed", "completed_with_omissions"}:
            tasks = [
                TaskItem(
                    capture_id=result.capture_id,
                    run_id=result.id,
                    text=task.text,
                    source_quote=task.source_quote,
                    due_expression=task.due_expression,
                    resolved_due_date=task.resolved_due_date,
                )
                for task in result.output.tasks
            ]
        checkpoint = WorkflowCheckpoint(
            run_id=result.id,
            capture_id=result.capture_id,
            state="prepared",
            payload={
                "source_text": source_text,
                "result": result.model_dump(mode="json"),
                "tasks": [task.model_dump(mode="json") for task in tasks],
                "memory_draft": None if draft is None else draft.model_dump(mode="json"),
                "retrieved_memories": [item.model_dump(mode="json") for item in retrieved],
            },
        )
        self.service.store.save_checkpoint(checkpoint)
        return self._commit_prepared(checkpoint)

    def _commit_prepared(self, checkpoint: WorkflowCheckpoint) -> CaptureRun:
        payload = checkpoint.payload
        result = CaptureRun.model_validate(payload["result"])
        draft_payload = payload.get("memory_draft")
        if draft_payload is not None:
            decisions = self.service.apply_draft(
                checkpoint.run_id,
                checkpoint.capture_id,
                payload["source_text"],
                MemoryAgentDraft.model_validate(draft_payload),
                [MemoryItem.model_validate(item) for item in payload["retrieved_memories"]],
            )
            result = result.model_copy(
                update={
                    "memory_decisions": [decision.model_dump(mode="json") for decision in decisions]
                }
            )
        self.service.store.save_run(result)
        self.service.commit_tasks(
            checkpoint.run_id,
            checkpoint.capture_id,
            [TaskItem.model_validate(task) for task in payload["tasks"]],
        )
        committed_payload = {**payload, "result": result.model_dump(mode="json")}
        self.service.store.commit_checkpoint(checkpoint.run_id, committed_payload)
        return result


def _normalize(value: str) -> str:
    return " ".join(value.split())
