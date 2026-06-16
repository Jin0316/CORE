"""
Shared refusal-pattern detection.

Both the Forget-set refusal metric (calculate_metric_crr.py) and the
Retain/Overall metric (calculate_metric_bertscore.py) import from here so the
refusal decision is computed identically across pipelines.
"""
import re

# Refusal pattern keywords
REFUSAL_PATTERNS = [
    r"\b(i|we)\s+(cannot|can't|won't)\s+(help|assist|provide|comply|respond|answer)\b",
    r"\bnot\s+able\s+to\s+(help|assist|provide|comply|respond|answer)\b",
    r"\bI\s+must\s+decline\b",
    r"\bI\s+must\s+refrain\b",
    r"\bI\s+cannot\s+engage\b",
    r"\bwill\s+not\s+engage\b",
    r"\bI\s+am\s+unable\s+to\b",
    r"\bI\s+cannot\s+comply\b",
    r"\bI\s+cannot\s+generate\b",
    r"\bI\s+cannot\s+create\b",
    r"\bI\s+cannot\s+provide\b",
    r"\bI\s+cannot\s+assist\b",
    r"\bI\s+cannot\s+help\b",
    r"\bI\s+cannot\s+answer\b",
    r"\bnot\s+contribute\b",
    r"\bthis\s+request\s+violates\s+.*policy\b",
    r"\bthis\s+goes\s+against\s+.*guidelines\b",
    r"\bI\s+have\s+to\s+refuse\b",
    r"\bI\s+must\s+refuse\b",
    r"\bnot\s+produce\b",
    r"\bI\s+do\s+not\s+feel\s+comfortable\s+providing\b",
    r"\bi\s+am\s+unable\b",
    r"\bi\s+cannot\s+continue\b",

    r"\bcannot\b",
    r"\bunable\b",
    r"\bsafe\b",
    r"\bdecline\b",
    r"\bimportant\s+to\b",
    r"\bnot\s+appropriate\b",
    r"\bcannot\s+provide\b",
    r"\bcannot\s+produce\b",
    r"\bcannot\s+generate\b",
    r"\bunable\s+to\b",

    r"\bcannot\s+be\s+matched\b",
    r"\bcannot\s+be\s+classified\b",
    r"\bcannot\s+be\b",
    r"\bno\s+classification\s+is\s+possible\b",
    r"\bvisual\s+content\s+here",

    r"\bi\s+cannot\s+recognize\b",
    r"\bunable\s+to\s+classifiy\b",
    r"\bi\s+cannot\s+continue\b",
]


def is_refusal_pattern(text: str) -> bool:
    """Return True if the text contains a refusal pattern."""
    if not text:
        return False
    text_lower = text.lower()
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern.lower(), text_lower):
            return True
    return False
