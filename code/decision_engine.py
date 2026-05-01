from config import (
    ALPHA,
    BETA,
    DISCLAIMER_CONFIDENCE_THRESHOLD,
    ESCALATE_CONFIDENCE_THRESHOLD,
    GAMMA,
    RISK_OVERRIDE_THRESHOLD,
)
from models import DecisionOutput


def decide(classifier_score: float, retrieval_relevance: float, risk_score: float, escalation_required: bool) -> DecisionOutput:
    confidence = ALPHA * classifier_score + BETA * retrieval_relevance + GAMMA * (1 - risk_score)
    confidence = max(0.0, min(1.0, confidence))

    if escalation_required or risk_score > RISK_OVERRIDE_THRESHOLD:
        return DecisionOutput(confidence=confidence, status="escalated", decision_mode="risk_override")
    if confidence < ESCALATE_CONFIDENCE_THRESHOLD:
        return DecisionOutput(confidence=confidence, status="escalated", decision_mode="low_confidence")
    if confidence < DISCLAIMER_CONFIDENCE_THRESHOLD:
        return DecisionOutput(confidence=confidence, status="replied", decision_mode="reply_with_disclaimer")
    return DecisionOutput(confidence=confidence, status="replied", decision_mode="full_resolution")
