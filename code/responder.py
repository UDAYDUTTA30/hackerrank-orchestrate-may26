"""
Responder: Generates user-facing responses grounded in the support corpus.
Uses LLM (Gemini) when available, with a deterministic corpus-extraction fallback.
"""
import re
from typing import List, Tuple

from models import DecisionOutput, RetrievalDoc, RiskOutput


def _score_actionability(content: str) -> Tuple[float, List[str]]:
    """Score a content snippet for actionability and extract relevant lines."""
    lines = content.split("\n")
    score = 0.0
    actionable_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# ") and len(stripped) < 80:
            continue

        if re.match(r"^\d+[\.\)]\s", stripped):
            score += 3
            actionable_lines.append(stripped)
        elif stripped.startswith("- ") or stripped.startswith("* "):
            score += 2
            actionable_lines.append(stripped)
        elif any(verb in stripped.lower() for verb in ["click", "go to", "navigate", "select", "enter", "open", "log in", "call", "contact", "visit"]):
            score += 2
            actionable_lines.append(stripped)
        elif len(stripped) > 20:
            score += 0.5
            actionable_lines.append(stripped)

    return score, actionable_lines


def _extract_actionable_content(docs: List[RetrievalDoc], query: str) -> str:
    """
    Extract the most actionable content from retrieved docs.
    Strictly prioritizes the top-ranked document (Retriever's choice).
    Only looks at doc 2/3 if doc 1 has zero actionable content.
    """
    if not docs:
        return ""

    # Start with the top doc
    top_score, top_lines = _score_actionability(docs[0].snippet)
    if top_score > 2.0:  # If top doc has decent instructions, stick with it
        return "\n".join(top_lines)

    # If top doc is purely informational, check if 2nd or 3rd have steps
    for doc in docs[1:3]:
        score, lines = _score_actionability(doc.snippet)
        if score > 5.0:  # Only switch if subsequent doc is highly actionable (e.g. a tutorial)
            return "\n".join(lines)

    # Default back to top doc lines (even if informational)
    return "\n".join(top_lines) if top_lines else docs[0].snippet


def _build_deterministic_response(decision: DecisionOutput, docs: List[RetrievalDoc], product_area: str, ticket_text: str) -> str:
    """Build a helpful response from corpus content without LLM."""

    if not docs:
        return (
            f"Thank you for reaching out about {product_area.replace('_', ' ')}. "
            "Unfortunately, I could not find a specific answer in our support documentation for your query. "
            "Please contact our support team directly for personalized assistance."
        )

    # Extract the most relevant content
    actionable = _extract_actionable_content(docs, ticket_text)
    top_doc = docs[0]

    # Clean up formatting artifacts
    response = actionable if actionable else top_doc.snippet
    response = re.sub(r"!\[.*?\]\(.*?\)", "", response)  # Remove images
    response = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r"\1 (\2)", response)  # Links
    response = re.sub(r"#{1,6}\s*", "", response)  # Headings
    response = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", response)  # Bold/Italic
    response = re.sub(r"\n{3,}", "\n\n", response)

    # Trim to reasonable length
    if len(response) > 1200:
        sentences = re.split(r"(?<=[.!?])\s+", response[:1200])
        response = " ".join(sentences[:-1]) if len(sentences) > 1 else response[:1200]

    if decision.decision_mode == "reply_with_disclaimer":
        response += (
            "\n\nNote: This guidance is based on available documentation and may not cover your exact situation. "
            "Please verify the details or contact support for further assistance."
        )

    return response.strip()


def compose_response(decision: DecisionOutput, risk: RiskOutput, docs: List[RetrievalDoc], product_area: str, ticket_text: str = "") -> str:
    """Generate final response."""
    if decision.status == "escalated":
        area_label = product_area.replace("_", " ").replace("-", " ")
        reasons = []
        if risk.escalation_required:
            reasons.append("this issue requires specialized handling")
        if risk.risk_flags:
            flag_map = {
                "account_compromise": "potential account security issue",
                "fraud_payment": "potential fraud or unauthorized transaction",
                "legal_threat": "legal matter",
                "platform_outage": "platform availability issue",
                "security_inquiry": "security-sensitive inquiry",
                "invalid_request": "request outside our support scope",
            }
            for flag in risk.risk_flags:
                if flag in flag_map:
                    reasons.append(flag_map[flag])
        
        reason_text = "; ".join(reasons) if reasons else "this requires human review"
        return (
            f"Your request regarding {area_label} has been escalated to our specialist team because {reason_text}. "
            "A support agent will review your case and respond as soon as possible."
        )

    # Try LLM
    try:
        from llm_client import generate_text, API_KEY
        if API_KEY and docs:
            context = "\n\n---\n\n".join([f"Source: {d.path}\n\n{d.snippet}" for d in docs[:3]])
            prompt = f"""You are a helpful support agent. Customer issue: {ticket_text} in area: {product_area}.
Using ONLY the docs below, write a clear, actionable response. No greetings. Ground strictly in context.

Docs:
{context}"""
            llm_response = generate_text(prompt)
            if llm_response and not llm_response.startswith(("LLM integration", "An error")):
                return llm_response
    except ImportError:
        pass

    return _build_deterministic_response(decision, docs, product_area, ticket_text)
