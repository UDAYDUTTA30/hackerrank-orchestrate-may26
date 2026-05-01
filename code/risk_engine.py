"""
Risk Engine: Assesses risk level and determines whether a ticket should
be force-escalated based on safety, fraud, legal, or platform concerns.
"""
from typing import List

from models import ClassifierOutput, RiskOutput, TicketInput


# High-risk categories that force escalation
IMMEDIATE_ESCALATE = {
    "account_compromise": [
        "account hacked", "account compromise", "stolen identity", "identity theft",
        "someone accessed my account", "unauthorized access", "password stolen"
    ],
    "fraud_payment": [
        "unauthorized transaction", "fraud", "stolen card", "didn't make this purchase",
        "fraudulent charge", "identity stolen", "phishing"
    ],
    "legal_threat": [
        "lawsuit", "legal notice", "attorney", "sue", "legal action",
        "regulatory complaint", "compliance violation"
    ],
    "platform_outage": [
        "site is down", "none of the submissions", "platform is down",
        "pages are accessible", "system is down", "cannot access any"
    ],
}

# Medium-risk categories that raise the risk score but don't force escalation
SOFT_RISK = {
    "security_inquiry": [
        "vulnerability", "security bug", "jailbreak", "internal policy",
        "exact logic", "bypass", "exploit", "reverse engineer"
    ],
    "data_sensitivity": [
        "delete my data", "personal information", "gdpr", "data breach",
        "export my data", "right to be forgotten"
    ],
    "financial_dispute": [
        "refund", "overcharged", "billing error", "wrong amount",
        "double charged", "cancel subscription"
    ],
}


def assess_risk(ticket: TicketInput, classifier_output: ClassifierOutput) -> RiskOutput:
    """
    Assess the risk level of a ticket and determine if forced escalation is needed.
    Returns risk score (0-1), risk flags, and escalation_required boolean.
    """
    text = f"{ticket.subject} {ticket.issue}".lower()
    flags: List[str] = []
    risk = 0.1
    escalation_required = False

    # Check immediate escalation triggers
    for name, keys in IMMEDIATE_ESCALATE.items():
        if any(k in text for k in keys):
            flags.append(name)
            risk = max(risk, 0.9)
            escalation_required = True

    # Check soft risk factors
    for name, keys in SOFT_RISK.items():
        if any(k in text for k in keys):
            flags.append(name)
            risk = max(risk, 0.55)

    # Invalid/out-of-scope requests — lower risk, agent can handle
    if classifier_output.request_type == "invalid":
        flags.append("invalid_request")
        risk = max(risk, 0.35)

    # Feature requests are low risk
    if classifier_output.request_type == "feature_request":
        risk = min(risk, 0.2)

    # Domain unknown + complex query = moderate risk
    if classifier_output.domain == "none" and len(text.split()) > 30:
        flags.append("ambiguous_domain")
        risk = max(risk, 0.45)

    # Bug reports about core platform should escalate
    if classifier_output.request_type == "bug" and any(s in text for s in ["site is down", "cannot access", "all pages"]):
        flags.append("platform_outage")
        risk = max(risk, 0.8)
        escalation_required = True

    return RiskOutput(
        risk_score=min(1.0, risk),
        risk_flags=sorted(set(flags)),
        escalation_required=escalation_required,
    )
