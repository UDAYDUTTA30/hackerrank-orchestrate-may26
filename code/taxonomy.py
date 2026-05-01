"""
Taxonomy: Data-driven product area classification using keyword patterns.
Maps ticket text to the most relevant product area within a domain.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple


class ProductTaxonomy:
    def __init__(self, taxonomy_path: Path) -> None:
        with taxonomy_path.open("r", encoding="utf-8") as f:
            self.data: Dict[str, dict] = json.load(f)

    def get_areas(self, domain: str) -> List[str]:
        """Return all known product area names for a domain."""
        domain_key = (domain or "").lower()
        if domain_key not in self.data:
            return []
        return list(self.data[domain_key].get("areas", {}).keys())

    def get_area_descriptions(self, domain: str) -> Dict[str, str]:
        """Return {area: description} for a domain."""
        domain_key = (domain or "").lower()
        if domain_key not in self.data:
            return {}
        areas = self.data[domain_key].get("areas", {})
        return {area: info.get("description", area) for area, info in areas.items()}

    def infer_area(self, domain: str, text: str) -> Tuple[str, float, List[str]]:
        """
        Infer the best product area for a given domain and ticket text.
        Returns (area_name, confidence_score, matched_keywords).
        """
        domain_key = (domain or "").lower()
        if domain_key not in self.data:
            return "general_support", 0.2, ["unknown_domain"]

        areas = self.data[domain_key].get("areas", {})
        text_l = text.lower()
        best_area = "general_support"
        best_hits: List[str] = []
        best_weighted_score = 0.0

        for area, info in areas.items():
            keywords = info.get("keywords", [])
            hits = [k for k in keywords if k in text_l]
            # Weight by number of hits and keyword specificity (longer keywords = more specific)
            weighted = sum(len(k.split()) for k in hits)  # multi-word keywords count more
            if weighted > best_weighted_score:
                best_weighted_score = weighted
                best_hits = hits
                best_area = area

        # Confidence based on match quality
        if best_hits:
            score = min(1.0, 0.4 + 0.12 * best_weighted_score)
        else:
            score = 0.25
        return best_area, score, best_hits
