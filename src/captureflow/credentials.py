"""A deterministic check that keeps disclosed credentials out of the model and SQLite."""

import re

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
_CREDENTIAL_DISCLOSURE_PATTERN = re.compile(
    rf"(?:(?:{_CREDENTIAL_NOUNS}){_CREDENTIAL_ASSIGNMENT})|(?:{_CREDENTIAL_VALUES})",
    re.IGNORECASE,
)


def contains_credential_disclosure(value: str) -> bool:
    """Return whether text appears to disclose a credential value."""

    return bool(_CREDENTIAL_DISCLOSURE_PATTERN.search(value))
