"""
Base types for search clients used in external factuality verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, runtime_checkable


@dataclass
class SearchHit:
    """A single search result normalized across backends.

    Attributes
    ----------
    url : str
        Canonical URL of the result page.
    domain : str
        Host component of the URL.
    title : str
        Page title, with inline highlight markers flattened to plain text.
    headline : str
        Short summary/description supplied by the search engine.
    passages : list[str]
        Passages/snippets extracted by the search engine.
    snippet : str
        Convenience field equal to ``" ".join(passages)`` if passages are
        present, otherwise the ``headline``. Suitable as a compact evidence
        string for factuality verification.
    saved_copy_url : str, optional
        URL of a cached/saved copy if provided by the backend.
    rank : int
        Zero-based rank of the hit in the original result list.
    """

    url: str
    domain: str
    title: str
    headline: str
    passages: List[str] = field(default_factory=list)
    snippet: str = ""
    saved_copy_url: Optional[str] = None
    rank: int = 0

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "domain": self.domain,
            "title": self.title,
            "headline": self.headline,
            "passages": list(self.passages),
            "snippet": self.snippet,
            "saved_copy_url": self.saved_copy_url,
            "rank": self.rank,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SearchHit":
        return cls(
            url=str(data.get("url", "")),
            domain=str(data.get("domain", "")),
            title=str(data.get("title", "")),
            headline=str(data.get("headline", "")),
            passages=[str(p) for p in (data.get("passages") or [])],
            snippet=str(data.get("snippet", "")),
            saved_copy_url=data.get("saved_copy_url"),
            rank=int(data.get("rank", 0)),
        )


@runtime_checkable
class SearchClient(Protocol):
    """Protocol implemented by every search backend."""

    async def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        """Return up to ``top_k`` hits for ``query``, ordered by relevance."""
        ...
