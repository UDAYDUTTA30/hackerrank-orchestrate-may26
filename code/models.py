from dataclasses import dataclass, field
from typing import List


@dataclass
class TicketInput:
    issue: str
    subject: str
    company: str
    row_id: int


@dataclass
class ClassifierOutput:
    domain: str
    product_area: str
    request_type: str
    classifier_score: float
    reasons: List[str] = field(default_factory=list)


@dataclass
class RetrievalDoc:
    path: str
    score: float
    snippet: str


@dataclass
class RiskOutput:
    risk_score: float
    risk_flags: List[str]
    escalation_required: bool


@dataclass
class DecisionOutput:
    confidence: float
    status: str
    decision_mode: str


@dataclass
class GroundingOutput:
    grounding_score: float
    unverified_claims: List[str]

