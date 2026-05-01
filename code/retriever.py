"""
Retriever: Loads and indexes the support corpus, performs TF-IDF-style
lexical retrieval with path-based domain filtering and smart snippet extraction.
"""
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from models import RetrievalDoc


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


class CorpusDocument:
    """A single indexed document from the support corpus."""

    __slots__ = ("path", "rel_path", "domain", "category", "content", "tokens", "token_counts", "title")

    def __init__(self, path: str, rel_path: str, domain: str, category: str, content: str, title: str) -> None:
        self.path = path
        self.rel_path = rel_path
        self.domain = domain
        self.category = category
        self.content = content
        self.title = title
        
        # Tokenize content
        content_tokens = _tokenize(content)
        title_tokens = _tokenize(title)
        
        # Boost title tokens (effectively making them appear multiple times)
        self.token_counts = Counter(content_tokens)
        for t in title_tokens:
            self.token_counts[t] += 4  # Title weight = 5x (1 base + 4 bonus)
            
        # Boost headings in content
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                heading_tokens = _tokenize(stripped.lstrip("#"))
                for t in heading_tokens:
                    self.token_counts[t] += 1  # Heading weight = 2x
        
        self.tokens = list(self.token_counts.keys())


class Retriever:
    """TF-IDF lexical retriever with domain scoping and smart snippet extraction."""

    def __init__(self, data_root: Path) -> None:
        self.corpus: List[CorpusDocument] = []
        self.idf: Dict[str, float] = {}
        self._load_corpus(data_root)
        self._build_idf()

    def _load_corpus(self, data_root: Path) -> None:
        for md in data_root.rglob("*.md"):
            try:
                content = md.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            # Strip YAML frontmatter
            content = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL | re.MULTILINE)

            rel = md.relative_to(data_root)
            parts = rel.parts
            domain = parts[0] if len(parts) >= 1 else "unknown"
            category = parts[1] if len(parts) >= 2 else "general"

            # Extract title from first heading
            title = ""
            for line in content.split("\n")[:10]:
                stripped = line.strip()
                if stripped.startswith("#"):
                    title = stripped.lstrip("#").strip()
                    break
            if not title:
                title = md.stem.replace("-", " ").replace("_", " ")

            self.corpus.append(
                CorpusDocument(
                    path=str(md),
                    rel_path=str(rel),
                    domain=domain,
                    category=category,
                    content=content,
                    title=title,
                )
            )

    def _build_idf(self) -> None:
        """Build inverse document frequency map across the corpus."""
        n = len(self.corpus)
        if n == 0:
            return
        doc_freq: Counter = Counter()
        for doc in self.corpus:
            for token in doc.token_counts:
                doc_freq[token] += 1
        self.idf = {token: math.log((n + 1) / (df + 1)) + 1 for token, df in doc_freq.items()}

    def _extract_best_snippet(self, content: str, query_tokens: set, max_length: int = 1000) -> str:
        """Extract the most relevant snippet from the document based on query overlap."""
        lines = content.split("\n")
        if not lines:
            return content[:max_length]

        scored_lines: List[Tuple[int, float, str]] = []
        for i, line in enumerate(lines):
            line_tokens = set(_tokenize(line))
            overlap = len(line_tokens.intersection(query_tokens))
            if line.strip().startswith("#") or re.match(r"^\s*\d+[\.\)]\s", line):
                overlap += 1.0
            scored_lines.append((i, overlap, line))

        best_start = 0
        best_score = 0.0
        window = 12
        for i in range(len(scored_lines)):
            window_score = sum(s for _, s, _ in scored_lines[i : i + window])
            if window_score > best_score:
                best_score = window_score
                best_start = i

        start = max(0, best_start - 1)
        snippet_lines = [line for _, _, line in scored_lines[start : start + window + 4]]
        snippet = "\n".join(snippet_lines).strip()
        if len(snippet) > max_length:
            snippet = snippet[:max_length]
        return snippet

    def search(self, query: str, domain: Optional[str] = None, k: int = 5) -> List[RetrievalDoc]:
        """Search the corpus using TF-IDF scoring with domain filtering."""
        
        # Synonym expansion for common support intents
        expanded_query = query.lower()
        synonyms = {
            "lost access": "removed seat workspace admin permission join",
            "down": "outage failure error broken unresponsive",
            "refund": "billing payment money charge back",
            "identity stolen": "fraud compromise hacked security",
        }
        for trigger, expansion in synonyms.items():
            if trigger in expanded_query:
                expanded_query += " " + expansion

        query_tokens = _tokenize(expanded_query)
        query_set = set(query_tokens)
        query_counts = Counter(query_tokens)

        if not query_tokens:
            return []

        scored: List[Tuple[float, CorpusDocument]] = []
        for doc in self.corpus:
            if domain and domain in {"hackerrank", "claude", "visa"}:
                if doc.domain != domain:
                    continue

            score = 0.0
            overlap_count = 0
            for token in query_set:
                if token in doc.token_counts:
                    overlap_count += 1
                    tf = 1 + math.log(doc.token_counts[token])
                    idf = self.idf.get(token, 1.0)
                    qtf = query_counts[token]
                    score += tf * idf * qtf

            if overlap_count == 0:
                continue

            # Length normalization (BM25-lite)
            doc_len = sum(doc.token_counts.values())
            score = score / (1.0 + 0.5 * (doc_len / 500.0))

            # Coverage & Structural Boost
            coverage = overlap_count / len(query_set)
            score *= (0.6 + 0.4 * coverage)
            
            # Title overlap bonus
            title_tokens = set(_tokenize(doc.title))
            title_overlap = len(query_set.intersection(title_tokens))
            if title_overlap > 0:
                score *= (1.0 + 0.25 * title_overlap)

            scored.append((score, doc))

        scored.sort(key=lambda x: -x[0])

        results: List[RetrievalDoc] = []
        for raw_score, doc in scored[:k]:
            normalized = min(1.0, raw_score / max(scored[0][0], 1e-6)) if scored else 0.0
            snippet = self._extract_best_snippet(doc.content, query_set)
            results.append(
                RetrievalDoc(
                    path=doc.rel_path,
                    score=normalized,
                    snippet=snippet,
                )
            )
        return results
