from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .llm import MemoryAgentModel, ModelError, PhaseThreeModel
from .memory_service import MemoryService
from .models import CaptureRun, MemoryAgentDraft, MemoryDecision, MemoryItem, RunError
from .workflow import PhaseTwoWorkflow


class MemoryGraphState(TypedDict, total=False):
    run_id: str
    capture_id: str
    source_text: str
    retrieved: list[MemoryItem]
    draft: MemoryAgentDraft
    decisions: list[MemoryDecision]
    error: str


class MemoryAgentWorkflow:
    """Bounded LangGraph subgraph for retrieval, proposal, and policy-controlled commit."""

    def __init__(self, model: MemoryAgentModel, service: MemoryService) -> None:
        self.model = model
        self.service = service
        graph = StateGraph(MemoryGraphState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("propose", self._propose)
        graph.add_node("commit", self._commit)
        graph.add_edge(START, "retrieve")
        graph.add_conditional_edges("retrieve", self._after_retrieve, {"propose": "propose", "end": END})
        graph.add_conditional_edges("propose", self._after_propose, {"commit": "commit", "end": END})
        graph.add_edge("commit", END)
        self.graph = graph.compile()

    def run(self, run_id: str, capture_id: str, source_text: str) -> MemoryGraphState:
        return self.graph.invoke(
            {"run_id": run_id, "capture_id": capture_id, "source_text": source_text}
        )

    def _retrieve(self, state: MemoryGraphState) -> MemoryGraphState:
        return {"retrieved": self.service.retrieve(state["source_text"])}

    @staticmethod
    def _after_retrieve(state: MemoryGraphState) -> str:
        return "propose" if state["source_text"].strip() else "end"

    def _propose(self, state: MemoryGraphState) -> MemoryGraphState:
        try:
            return {"draft": self.model.propose_memories(state["source_text"], state["retrieved"])}
        except (ModelError, ValueError) as error:
            return {"error": str(error)}

    @staticmethod
    def _after_propose(state: MemoryGraphState) -> str:
        return "end" if state.get("error") else "commit"

    def _commit(self, state: MemoryGraphState) -> MemoryGraphState:
        return {
            "decisions": self.service.apply_draft(
                state["run_id"],
                state["capture_id"],
                state["source_text"],
                state["draft"],
                state["retrieved"],
            )
        }


class PhaseThreeWorkflow:
    """Runs Phase 2 only as needed, then enters the memory subgraph when planned."""

    def __init__(self, model: PhaseThreeModel, service: MemoryService) -> None:
        self.phase_two = PhaseTwoWorkflow(model)
        self.memory_graph = MemoryAgentWorkflow(model, service)

    def run(self, **kwargs) -> CaptureRun:
        result = self.phase_two.run(**kwargs)
        if result.status in {"failed", "grounding_failed"} or not result.plan:
            return result
        if not any(step.action == "inspect_memory" for step in result.plan.steps):
            return result
        if result.model_calls >= 6:
            status = "completed_with_omissions" if result.status == "completed" else result.status
            return result.model_copy(
                update={
                    "status": status,
                    "warnings": [
                        *result.warnings,
                        "Memory inspection was skipped because the six-call run limit was reached.",
                    ],
                }
            )
        state = self.memory_graph.run(result.id, result.capture_id, kwargs["source_text"])
        if state.get("error"):
            return result.model_copy(
                update={
                    "status": "completed_with_omissions",
                    "warnings": [*result.warnings, "Memory proposal failed; no memory was written."],
                    "errors": [*result.errors, RunError(stage="memory", message=state["error"])],
                    "model_calls": result.model_calls + 1,
                    "retrieved_memory_ids": [item.id for item in state.get("retrieved", [])],
                }
            )
        return result.model_copy(
            update={
                "model_calls": result.model_calls + 1,
                "retrieved_memory_ids": [item.id for item in state.get("retrieved", [])],
                "memory_decisions": [
                    decision.model_dump(mode="json") for decision in state.get("decisions", [])
                ],
            }
        )
