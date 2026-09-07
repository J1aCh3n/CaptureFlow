from uuid import NAMESPACE_URL, uuid5

from .memory import MemoryPolicy
from .models import (
    DEFAULT_WORKSPACE_ID,
    MemoryAgentDraft,
    MemoryDecision,
    MemoryItem,
    PendingMemoryReview,
    TaskItem,
    utc_now,
)
from .storage import CaptureStore


class MemoryService:
    """Applies deterministic Phase 3 persistence policy around model proposals."""

    def __init__(self, store: CaptureStore, policy: MemoryPolicy | None = None) -> None:
        self.store = store
        self.policy = policy or MemoryPolicy()

    def retrieve(self, query: str) -> list[MemoryItem]:
        return self.store.search_memories(query)

    def apply_draft(
        self,
        run_id: str,
        capture_id: str,
        source_text: str,
        draft: MemoryAgentDraft,
        retrieved_memories: list[MemoryItem],
    ) -> list[MemoryDecision]:
        """Persist allowed effects with IDs derived from run ID and proposal index."""

        decisions: list[MemoryDecision] = []
        for index, proposal in enumerate(draft.proposals):
            decision = self.policy.decide(proposal, source_text, retrieved_memories)
            decisions.append(decision)
            if decision.decision == "auto":
                self.store.save_memory(
                    MemoryItem(
                        id=_stable_id(run_id, "memory", index),
                        memory_type=proposal.memory_type,
                        content=proposal.content,
                        source_capture_id=capture_id,
                        source_quote=proposal.source_quote,
                        write_mode="auto",
                        confidence=proposal.confidence,
                    ),
                    event_action="created_auto",
                )
            elif decision.decision == "review":
                self.store.save_pending_review(
                    PendingMemoryReview(
                        id=_stable_id(run_id, "review", index),
                        capture_id=capture_id,
                        proposal=proposal,
                        decision_reason=decision.reason,
                        conflicting_memory_ids=decision.conflicting_memory_ids,
                    )
                )
        return decisions

    def commit_tasks(self, run_id: str, capture_id: str, tasks: list[TaskItem]) -> list[TaskItem]:
        """Save generated tasks once, even if a checkpointed run is resumed."""

        committed: list[TaskItem] = []
        for index, task in enumerate(tasks):
            committed.append(
                self.store.save_task(
                    task.model_copy(
                        update={
                            "id": _stable_id(run_id, "task", index),
                            "run_id": run_id,
                            "capture_id": capture_id,
                        }
                    )
                )
            )
        return committed

    def approve_review(
        self,
        review_id: str,
        content: str | None = None,
        supersede_conflicts: bool = False,
    ) -> MemoryItem:
        review = self.store.get_review(review_id)
        if review.status != "pending":
            raise ValueError("only a pending review can be approved")
        proposal = review.proposal.model_copy(update={"content": content or review.proposal.content})
        item = self.store.save_memory(
            MemoryItem(
                id=_stable_id(review.id, "approved-memory", 0),
                memory_type=proposal.memory_type,
                content=proposal.content,
                source_capture_id=review.capture_id,
                source_quote=proposal.source_quote,
                write_mode="reviewed",
                confidence=proposal.confidence,
            ),
            event_action="created_reviewed",
        )
        if supersede_conflicts:
            for memory_id in review.conflicting_memory_ids:
                active = self.store.get_memory(memory_id)
                if active.status == "active":
                    self.store.set_memory_status(memory_id, "superseded", "superseded_by_review")
        self.store.resolve_review(review_id, "approved")
        return item

    def reject_review(self, review_id: str) -> None:
        review = self.store.get_review(review_id)
        if review.status != "pending":
            raise ValueError("only a pending review can be rejected")
        self.store.resolve_review(review_id, "rejected")

    def undo_auto_memory(self, memory_id: str) -> MemoryItem:
        item = self.store.get_memory(memory_id)
        if item.write_mode != "auto" or item.status != "active":
            raise ValueError("only an active automatic memory can be undone")
        return self.store.set_memory_status(memory_id, "deleted", "undone_auto_write")

    def delete_memory(self, memory_id: str) -> MemoryItem:
        item = self.store.get_memory(memory_id)
        if item.status == "deleted":
            return item
        return self.store.set_memory_status(memory_id, "deleted", "deleted")

    def restore_memory(self, memory_id: str) -> MemoryItem:
        item = self.store.get_memory(memory_id)
        if item.status != "deleted":
            raise ValueError("only a deleted memory can be restored")
        return self.store.set_memory_status(memory_id, "active", "restored")

    def edit_memory(self, memory_id: str, content: str) -> MemoryItem:
        item = self.store.get_memory(memory_id)
        if item.status != "active":
            raise ValueError("only an active memory can be edited")
        replacement = self.store.save_memory(
            MemoryItem(
                id=_stable_id(memory_id, "edit", 0),
                workspace_id=item.workspace_id,
                memory_type=item.memory_type,
                content=content,
                source_capture_id=item.source_capture_id,
                source_quote=item.source_quote,
                write_mode="reviewed",
                confidence=item.confidence,
                created_at=utc_now(),
                updated_at=utc_now(),
            ),
            event_action="edited_replacement",
        )
        self.store.set_memory_status(memory_id, "superseded", "superseded_by_edit")
        return replacement


def _stable_id(scope: str, kind: str, index: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"captureflow:{scope}:{kind}:{index}"))
