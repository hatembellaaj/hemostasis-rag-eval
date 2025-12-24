from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Tuple


Verdict = Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED"]

_CIT_RE = re.compile(r"\[S\d+\]")

def rule_based_verify(answer: str) -> Dict[str, Any]:
    """
    Minimal verification:
    - If no citations: UNSUPPORTED
    - If contains explicit uncertainty/missing markers + some citations: PARTIAL
    - Else: SUPPORTED (weak proxy; later replaced/augmented with LLM judge)
    """
    has_cit = bool(_CIT_RE.search(answer or ""))
    if not has_cit:
        return {"verdict": "UNSUPPORTED", "reason": "NO_CITATIONS"}

    lowered = (answer or "").lower()
    if any(k in lowered for k in ["insufficient", "not found", "missing", "unknown", "cannot determine", "to_validate"]):
        return {"verdict": "PARTIAL", "reason": "EXPLICIT_MISSINGNESS"}

    return {"verdict": "SUPPORTED", "reason": "CITATIONS_PRESENT"}
