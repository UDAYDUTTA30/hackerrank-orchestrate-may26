import re
from typing import List

from models import GroundingOutput, RetrievalDoc


def validate_grounding(response: str, retrieved_docs: List[RetrievalDoc]) -> GroundingOutput:
    response_tokens = set(re.findall(r"[a-z0-9']+", response.lower()))
    kb_tokens = set()
    for doc in retrieved_docs:
        kb_tokens.update(re.findall(r"[a-z0-9']+", doc.snippet.lower()))

    if not response_tokens:
        return GroundingOutput(grounding_score=0.0, unverified_claims=["empty_response"])

    uncommon = [t for t in response_tokens if len(t) > 6 and t not in kb_tokens]
    token_overlap = len(response_tokens.intersection(kb_tokens)) / max(1, len(response_tokens))

    # Sentence-level validation: each sentence should contain at least one KB-backed token.
    sentences = [s.strip() for s in re.split(r"[.!?]+", response.lower()) if s.strip()]
    covered = 0
    for sentence in sentences:
        sentence_tokens = set(re.findall(r"[a-z0-9']+", sentence))
        if sentence_tokens.intersection(kb_tokens):
            covered += 1
    sentence_coverage = covered / max(1, len(sentences))

    score = max(0.0, min(1.0, 0.7 * token_overlap + 0.3 * sentence_coverage))
    return GroundingOutput(grounding_score=score, unverified_claims=sorted(uncommon)[:8])
