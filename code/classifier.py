"""
Classifier: Determines domain, request type, and product area for a support ticket.
Uses LLM (Gemini) when available, with a robust deterministic fallback.
"""
from typing import List, Optional

from pydantic import BaseModel, Field

from models import ClassifierOutput, TicketInput
from taxonomy import ProductTaxonomy


# --- LLM schema (used when API key is available) ---

class LLMClassification(BaseModel):
    domain: str = Field(default="none", description="The company domain: 'hackerrank', 'claude', 'visa', or 'none'")
    request_type: str = Field(default="product_issue", description="The type of request: 'product_issue', 'feature_request', 'bug', 'invalid'")
    product_area: str = Field(default="general_support", description="The specific product area/category the ticket belongs to")
    confidence: float = Field(default=0.5, description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(default="", description="Brief reason for the classification")


# --- Domain inference ---

def _infer_domain(company: str, text: str) -> str:
    """Infer domain from company field, then from ticket text keywords."""
    c = (company or "").strip().lower()
    # Direct company match
    if c in {"hackerrank", "claude", "visa"}:
        return c

    t = text.lower()

    # HackerRank signals
    hr_signals = ["hackerrank", "hacker rank", "codepair", "hackerrank for work", "hackerrank community"]
    if any(s in t for s in hr_signals):
        return "hackerrank"

    # Claude/Anthropic signals
    claude_signals = ["claude", "anthropic", "bedrock", "artifacts", "claude.ai", "claude pro", "claude code"]
    if any(s in t for s in claude_signals):
        return "claude"

    # Visa signals
    visa_signals = ["visa", "visa card", "visa debit", "visa credit", "traveller's cheque",
                    "travelers cheque", "merchant", "chargeback"]
    if any(s in t for s in visa_signals):
        return "visa"

    # Context clues for specific domains
    if any(s in t for s in ["assessment", "candidate", "test invite", "codepair"]):
        return "hackerrank"
    if any(s in t for s in ["conversation", "ai assistant", "prompt"]):
        return "claude"
    if any(s in t for s in ["card", "payment", "transaction", "bank"]):
        return "visa"

    return "none"


# --- Request type inference ---

def _infer_request_type(text: str, domain: str) -> str:
    """Classify the request type from ticket text."""
    t = text.lower()

    # Invalid / out-of-scope detection
    invalid_signals = [
        "actor in iron man", "recipe for", "what is the capital of",
        "tell me a joke", "write me a poem", "help me hack",
        "do my homework", "solve this coding problem for me",
        "thank you", "thanks for", "happy to help", "awesome service"
    ]
    if any(s in t for s in invalid_signals):
        return "invalid"

    # Feature request signals
    feature_signals = [
        "feature request", "please add", "would like to have", "would be great if",
        "can you add", "it would be nice", "suggestion", "wish list", "wishlist",
        "requesting a feature", "new feature", "consider adding", "we need support for"
    ]
    if any(s in t for s in feature_signals):
        return "feature_request"

    # Bug signals
    bug_signals = [
        "bug", "site is down", "not working", "failing", "broken", "error",
        "crash", "glitch", "doesn't load", "cannot access", "502", "500",
        "page not found", "timeout", "unresponsive"
    ]
    if any(s in t for s in bug_signals):
        return "bug"

    # Out-of-scope but not necessarily "invalid" — might just be generic
    if domain == "none":
        # Check if it's a generic/meaningless message
        if len(t.split()) < 5 and not any(s in t for s in ["help", "issue", "problem", "question"]):
            return "invalid"

    return "product_issue"


# --- Main classifier ---

def classify(ticket: TicketInput, taxonomy: ProductTaxonomy) -> ClassifierOutput:
    """
    Classify a support ticket. Attempts LLM classification first,
    falls back to deterministic rules.
    """
    text = f"{ticket.subject} {ticket.issue}".strip()

    # Try LLM classification if available
    try:
        from llm_client import generate_structured, API_KEY
        if API_KEY:
            areas_for_prompt = ""
            # We'll build the prompt after domain inference for better context
            domain_hint = _infer_domain(ticket.company, text)
            if domain_hint != "none":
                area_descs = taxonomy.get_area_descriptions(domain_hint)
                areas_for_prompt = "\n".join(f"- {area}: {desc}" for area, desc in area_descs.items())

            prompt = f"""You are an expert support triage agent for three companies: HackerRank, Claude (by Anthropic), and Visa.

Analyze this support ticket and classify it.

Company field: {ticket.company}
Subject: {ticket.subject}
Issue: {ticket.issue}

Rules:
- domain must be one of: hackerrank, claude, visa, none
- request_type must be one of: product_issue, feature_request, bug, invalid
- product_area should be one of the areas listed below for the detected domain
- If the ticket is out-of-scope (e.g., asking about actors, recipes, unrelated topics), set request_type to 'invalid'
- If the ticket mentions a site being down or a platform outage, set request_type to 'bug'
- 'product_issue' is the default for general support questions

{f"Available product areas for {domain_hint}:" if areas_for_prompt else ""}
{areas_for_prompt}

Return your classification with confidence score and brief reasoning."""

            try:
                llm_result = generate_structured(prompt, LLMClassification)
                if llm_result and llm_result.domain:
                    # Validate domain
                    llm_domain = llm_result.domain.lower().strip()
                    if llm_domain not in {"hackerrank", "claude", "visa", "none"}:
                        llm_domain = domain_hint

                    # Validate request_type
                    llm_rt = llm_result.request_type.lower().strip()
                    if llm_rt not in {"product_issue", "feature_request", "bug", "invalid"}:
                        llm_rt = "product_issue"

                    return ClassifierOutput(
                        domain=llm_domain,
                        product_area=llm_result.product_area.lower().replace(" ", "_").replace("-", "_"),
                        request_type=llm_rt,
                        classifier_score=max(0.3, min(1.0, llm_result.confidence)),
                        reasons=[f"llm_reason={llm_result.reasoning}"],
                    )
            except Exception as e:
                print(f"[Classifier] LLM error, falling back to deterministic: {e}")
    except ImportError:
        pass

    # --- Deterministic fallback ---
    domain = _infer_domain(ticket.company, text)
    request_type = _infer_request_type(text, domain)
    product_area, tax_score, hits = taxonomy.infer_area(domain, text)
    reasons: List[str] = [f"domain={domain}", f"request_type={request_type}"]
    if hits:
        reasons.append(f"taxonomy_hits={','.join(hits)}")

    # Composite classifier score
    base = 0.5 * tax_score
    if request_type != "invalid":
        base += 0.35
    if domain != "none":
        base += 0.15
    score = max(0.2, min(1.0, base))

    return ClassifierOutput(
        domain=domain,
        product_area=product_area,
        request_type=request_type,
        classifier_score=score,
        reasons=reasons,
    )
