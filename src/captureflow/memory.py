import re

from .models import MemoryDecision, MemoryItem, MemoryProposal


AUTO_CONFIDENCE = 0.85
MAX_RETRIEVED_MEMORIES = 5
# CJK text has no spaces, and Python treats CJK characters as word characters, so
# \b would not fire between 的 and 密码 or around an embedded Latin noun. Bound the
# nouns on ASCII letters and digits instead.
_CREDENTIAL_NOUNS = (
    r"(?<![A-Za-z0-9])(?:"
    r"api[ _-]?key|access[ _-]?token|auth[ _-]?token|bearer[ _-]?token|refresh[ _-]?token"
    r"|private[ _-]?key|secret[ _-]?key|passphrase|passcode|password|credential|secret|token"
    r")(?![A-Za-z0-9])"
    r"|密码|口令|密钥|秘钥|私钥|凭据|凭证|访问令牌|令牌"
)
# Separators include the Chinese copulas and the full-width colon.
_CREDENTIAL_ASSIGNMENT = r"\s*(?:is|are|=|:|：|是|为)\s*\S+"
# Well-known credential shapes, which are disclosures even with no noun in the text.
_CREDENTIAL_VALUES = (
    r"(?<![A-Za-z0-9])(?:"
    r"sk-[A-Za-z0-9_\-]{16,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|xox[baprs]-[A-Za-z0-9\-]{10,}"
    r"|AIza[0-9A-Za-z_\-]{30,}"
    r"|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}"
    r")"
)
_SENSITIVE_PATTERN = re.compile(
    rf"(?:{_CREDENTIAL_NOUNS})|(?:{_CREDENTIAL_VALUES})", re.IGNORECASE
)
_CREDENTIAL_DISCLOSURE_PATTERN = re.compile(
    rf"(?:(?:{_CREDENTIAL_NOUNS}){_CREDENTIAL_ASSIGNMENT})|(?:{_CREDENTIAL_VALUES})",
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


def contains_credential_disclosure(value: str) -> bool:
    """Return whether a capture appears to disclose a credential value."""

    return bool(_CREDENTIAL_DISCLOSURE_PATTERN.search(value))
