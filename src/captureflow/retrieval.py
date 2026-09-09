"""Bounded, deterministic Phase 7 retrieval over local project records."""

from collections.abc import Callable
from dataclasses import dataclass
import re
from typing import Literal

from .models import ProjectRecord
from .storage import CaptureStore


RetrievalMode = Literal["current", "history"]
MAX_RETRIEVED_PROJECT_RECORDS = 6


@dataclass(frozen=True)
class FrozenRetrievalCase:
    name: str
    query: str
    mode: RetrievalMode
    expected_record_ids: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalEvaluation:
    recall_at_k: float
    failures: tuple[str, ...]


def classify_project_query(query: str) -> RetrievalMode | None:
    """Return a narrow read-query mode; other turns remain Phase 6 write proposals."""

    normalized = query.casefold()
    direct_history_questions = ("why", "为什么", "为何")
    history_topics = ("history", "earlier", "previous", "历史", "之前")
    history_question_markers = (
        "what",
        "how",
        "which",
        "show",
        "tell me",
        "什么",
        "怎么",
        "如何",
        "哪些",
        "查看",
        "告诉我",
        "回顾",
    )
    if any(marker in normalized for marker in direct_history_questions) or (
        any(marker in normalized for marker in history_topics)
        and any(marker in normalized for marker in history_question_markers)
    ):
        return "history"
    current_topics = (
        "status",
        "progress",
        "where are we",
        "what remains",
        "现在",
        "目前",
        "进度",
        "状态",
        "做到哪里",
        "还有什么",
    )
    question_markers = (
        "?",
        "？",
        "what",
        "where",
        "how",
        "which",
        "什么",
        "哪里",
        "如何",
        "怎么",
        "怎样",
        "哪些",
        "吗",
        "呢",
    )
    return (
        "current"
        if any(marker in normalized for marker in current_topics)
        and any(marker in normalized for marker in question_markers)
        else None
    )


def retrieve_project_records(
    store: CaptureStore,
    project_id: str,
    query: str,
    mode: RetrievalMode,
    *,
    limit: int = MAX_RETRIEVED_PROJECT_RECORDS,
) -> list[ProjectRecord]:
    """Retrieve only same-project records with explicit state filtering and stable ties."""

    if not 1 <= limit <= MAX_RETRIEVED_PROJECT_RECORDS:
        raise ValueError(f"retrieval limit must be between 1 and {MAX_RETRIEVED_PROJECT_RECORDS}")
    records = store.list_project_records(
        project_id, states=("active", "completed", "resolved", "superseded")
    )
    if mode == "current":
        records = [
            record
            for record in records
            if record.state == "active" or (record.kind == "milestone" and record.state == "completed")
        ]
    ranked = sorted(
        records,
        key=lambda record: (
            _lexical_score(query, record),
            _state_priority(record, mode),
            record.updated_at.isoformat(),
            record.id,
        ),
        reverse=True,
    )
    return ranked[:limit]


def evaluate_retrieval(
    cases: list[FrozenRetrievalCase],
    retrieve: Callable[[str, RetrievalMode], list[str]],
) -> RetrievalEvaluation:
    """Report Recall@K and failing case names without treating tests as live quality evidence."""

    expected_total = sum(len(case.expected_record_ids) for case in cases)
    if expected_total == 0:
        raise ValueError("retrieval evaluation requires expected record IDs")
    found_total = 0
    failures: list[str] = []
    for case in cases:
        returned_ids = set(retrieve(case.query, case.mode))
        expected_ids = set(case.expected_record_ids)
        found_total += len(returned_ids & expected_ids)
        if not expected_ids.issubset(returned_ids):
            failures.append(case.name)
    return RetrievalEvaluation(
        recall_at_k=found_total / expected_total,
        failures=tuple(failures),
    )


def _lexical_score(query: str, record: ProjectRecord) -> int:
    query_terms = set(_tokens(query))
    if not query_terms:
        return 0
    record_terms = set(_tokens(f"{record.content} {record.rationale or ''} {record.source_quote}"))
    return len(query_terms & record_terms)


def _state_priority(record: ProjectRecord, mode: RetrievalMode) -> int:
    if mode == "history":
        return {
            "decision": 5,
            "constraint": 4,
            "milestone": 3,
            "task": 2,
            "open_question": 1,
            "blocker": 1,
        }[record.kind]
    return {
        "blocker": 5,
        "open_question": 4,
        "task": 3,
        "decision": 2,
        "constraint": 2,
        "milestone": 1,
    }[record.kind]


def _tokens(value: str) -> list[str]:
    return re.findall(r"[\w'-]+", value.casefold())
