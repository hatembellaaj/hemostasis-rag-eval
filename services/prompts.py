from __future__ import annotations


SYSTEM_RAG_STRICT = """You are an assistant that drafts clinical protocol text ONLY from the provided evidence.
Rules:
- Use ONLY the evidence snippets provided in CONTEXT.
- Every key clinical claim MUST include a citation like [S1], [S2].
- If evidence is missing or insufficient, do NOT guess. Say what is missing and add it under `to_validate`.
- Keep outputs structured as requested. Be conservative.
"""

PROTOCOL_TEMPLATE = """Produce a protocol draft with the following sections:
1) Indication and scope
2) Target population (inclusion/exclusion, subgroups)
3) Treatment regimen (drug, dose, frequency, duration, escalation)
4) Monitoring and follow-up
5) Adverse events and contraindications
6) Alternatives / special situations
7) Country-specific notes
8) to_validate (bulleted list of missing or uncertain points)
"""

def build_user_prompt(question: str, country: str | None, disease: str | None) -> str:
    parts = []
    parts.append(f"Task: {question.strip()}")
    if disease:
        parts.append(f"Disease: {disease}")
    if country:
        parts.append(f"Country constraint: {country}")
    parts.append("Output must follow the protocol template.")
    return "\n".join(parts)
