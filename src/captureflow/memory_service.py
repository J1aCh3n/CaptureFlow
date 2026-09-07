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
        conflict_ids = (
            self._current_conflict_ids(review.conflicting_memory_ids)
            if supersede_conflicts
            else []
        )
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
        for memory_id in conflict_ids:
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
        if content == item.content:
            return item
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
            event_details={"replaces_memory_id": memory_id},
        )
        self.store.set_memory_status(
            memory_id,
            "superseded",
            "superseded_by_edit",
            event_details={"replacement_memory_id": replacement.id},
        )
        return replacement

    def _current_conflict_ids(self, stored_ids: list[str]) -> list[str]:
        """Resolve reviewed conflicts through any edits made while review was pending."""

        active = {item.id: item for item in self.store.list_memories(statuses=("active",))}
        resolved: set[str] = set()
        for stored_id in stored_ids:
            current_id = stored_id
            seen: set[str] = set()
            while current_id not in seen:
                seen.add(current_id)
                if current_id in active:
                    resolved.add(current_id)
                    break
                events = self.store.list_memory_events(current_id)
                replacement_id = next(
                    (
                        event.details.get("replacement_memory_id")
                        for event in events
                        if event.details.get("replacement_memory_id")
                    ),
                    None,
                )
                if replacement_id is None:
                    break
                current_id = replacement_id

            if resolved.intersection(seen):
                continue
            original = self.store.get_memory(stored_id)
            for candidate in active.values():
                if (
                    candidate.memory_type == original.memory_type
                    and candidate.source_capture_id == original.source_capture_id
                    and candidate.source_quote == original.source_quote
                ):
                    resolved.add(candidate.id)
        return sorted(resolved)


def _stable_id(scope: str, kind: str, index: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"captureflow:{scope}:{kind}:{index}"))
