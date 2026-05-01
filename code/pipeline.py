"""
Pipeline: End-to-end orchestration for processing a single support ticket.
Coordinates classifier -> retriever -> risk engine -> decision engine ->
responder -> grounding validator.
"""
from typing import Dict

from classifier import classify
from config import TOP_K
from decision_engine import decide
from grounding_validator import validate_grounding
from models import TicketInput
from responder import compose_response
from retriever import Retriever
from risk_engine import assess_risk
from taxonomy import ProductTaxonomy


def run_ticket(
    ticket: TicketInput, taxonomy: ProductTaxonomy, retriever: Retriever, explain: bool = False
) -> Dict[str, str]:
    """
    Process a single support ticket through the full triage pipeline.

    Returns a dict with: status, product_area, response, justification, request_type.
    If explain=True, also includes a _debug dict with internal scores.
    """
    text = f"{ticket.subject} {ticket.issue}"

    # Step 1: Classify domain, request type, and product area
    classifier_output = classify(ticket, taxonomy)

    # Step 2: Retrieve relevant corpus documents
    docs = retriever.search(text, classifier_output.domain, k=TOP_K)
    retrieval_relevance = docs[0].score if docs else 0.0

    # Step 3: Assess risk
    risk = assess_risk(ticket, classifier_output)

    # Step 4: Make the reply/escalate decision
    decision = decide(
        classifier_score=classifier_output.classifier_score,
        retrieval_relevance=retrieval_relevance,
        risk_score=risk.risk_score,
        escalation_required=risk.escalation_required,
    )

    # Step 5: Generate response
    response = compose_response(decision, risk, docs, classifier_output.product_area, text)

    # Step 6: Validate grounding of the response
    grounding = validate_grounding(response, docs)
    if grounding.grounding_score < 0.12 and decision.status == "replied" and docs:
        # Response is poorly grounded — escalate for safety
        decision.status = "escalated"
        response = (
            "I could not find sufficiently grounded evidence in the support documentation to provide a reliable answer. "
            "Escalating this ticket to a specialist for accurate assistance."
        )

    # Step 7: Build justification
    top_source = docs[0].path if docs else "no_match"
    justification = (
        f"domain={classifier_output.domain}; "
        f"product_area={classifier_output.product_area}; "
        f"classifier_confidence={classifier_output.classifier_score:.2f}; "
        f"retrieval_relevance={retrieval_relevance:.2f}; "
        f"risk={risk.risk_score:.2f}; "
        f"grounding={grounding.grounding_score:.2f}; "
        f"decision={decision.decision_mode}; "
        f"top_source={top_source}"
    )
    if risk.risk_flags:
        justification += f"; risk_flags={','.join(risk.risk_flags)}"
    if classifier_output.reasons:
        justification += f"; reasons={'; '.join(classifier_output.reasons[:2])}"

    result = {
        "status": decision.status,
        "product_area": classifier_output.product_area,
        "response": response,
        "justification": justification,
        "request_type": classifier_output.request_type,
    }

    if explain:
        result["_debug"] = {
            "domain": classifier_output.domain,
            "classifier_score": round(classifier_output.classifier_score, 4),
            "retrieval_relevance": round(retrieval_relevance, 4),
            "risk_score": round(risk.risk_score, 4),
            "decision_mode": decision.decision_mode,
            "confidence": round(decision.confidence, 4),
            "grounding_score": round(grounding.grounding_score, 4),
            "risk_flags": risk.risk_flags,
            "top_doc": docs[0].path if docs else "",
        }

    return result
