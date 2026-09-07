import re

from .models import MemoryDecision, MemoryItem, MemoryProposal


AUTO_CONFIDENCE = 0.85
MAX_RETRIEVED_MEMORIES = 5
_SENSITIVE_PATTERN = re.compile(
    r"\b(?:api[ _-]?key|password|secret|access[ _-]?token|credential|private[ _-]?key)\b",
    re.IGNORECASE,
)
_FIRST_PERSON_PATTERN = re.compile(r"\b(?:i|i'm|i’m|my|mine)\b", re.IGNORECASE)


class MemoryPolicy:
    """Deterministic risk classification for untrusted memory proposals."""

    def decide(
        self,
        proposal: MemoryProposal,
        source_text: str,
        retrieved_memories: list[MemoryItem],
    ) -> MemoryDecision:
        if _normalize(proposal.source_quote) not in _normalize(source_text):
            return MemoryDecision(
                proposal=proposal,
                decision="blocked",
                reason="The proposed memory quote does not occur in the capture.",
            )
        if _SENSITIVE_PATTERN.search(f"{proposal.content}\n{proposal.source_quote}"):
            return MemoryDecision(
                proposal=proposal,
                decision="blocked",
                reason="The proposal appears to contain a credential or secret.",
            )

        retrieved_by_id = {item.id: item for item in retrieved_memories if item.status == "active"}
        conflicts = [
            memory_id
            for memory_id in proposal.related_memory_ids
            if memory_id in retrieved_by_id
            and retrieved_by_id[memory_id].content.casefold() != proposal.content.casefold()
        ]
        if any(item.content.casefold() == proposal.content.casefold() for item in retrieved_by_id.values()):
            return MemoryDecision(
                proposal=proposal,
                decision="duplicate",
                reason="An active retrieved memory already has the same content.",
            )
        if conflicts:
            return MemoryDecision(
                proposal=proposal,
                decision="review",
                reason="The proposal conflicts with an active retrieved memory.",
                conflicting_memory_ids=conflicts,
            )
        if proposal.memory_type in {"decision", "idea"}:
            return MemoryDecision(
                proposal=proposal,
                decision="review",
                reason="Decisions and ideas require explicit user confirmation.",
            )
        if not proposal.is_explicit:
            return MemoryDecision(
                proposal=proposal,
                decision="review",
                reason="Inferred memories require explicit user confirmation.",
            )
        if proposal.confidence < AUTO_CONFIDENCE:
            return MemoryDecision(
                proposal=proposal,
                decision="review",
                reason="Low-confidence memories require explicit user confirmation.",
            )
        if not _FIRST_PERSON_PATTERN.search(proposal.source_quote):
            return MemoryDecision(
                proposal=proposal,
                decision="review",
                reason="Automatic writes require an explicit first-person source statement.",
            )
        return MemoryDecision(
            proposal=proposal,
            decision="auto",
            reason="Explicit, low-risk, high-confidence first-person memory.",
        )


def lexical_match_score(query: str, item: MemoryItem) -> int:
    query_terms = set(_tokens(query))
    if not query_terms:
        return 0
    content_terms = set(_tokens(f"{item.content} {item.source_quote}"))
    return len(query_terms & content_terms)


def _tokens(value: str) -> list[str]:
    return re.findall(r"[\w'-]+", value.casefold())


def _normalize(value: str) -> str:
    return " ".join(value.split())
