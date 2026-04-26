"""
In-memory stub :class:`SearchClient` used in unit tests.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List

from uqlm.utils.search.base import SearchClient, SearchHit

logger = logging.getLogger(__name__)


class StubSearchClient(SearchClient):
    """Returns pre-registered hits for exact query matches.

    Useful for deterministic tests of ``ExternalVerifier`` without touching the
    network. Queries not present in ``fixtures`` resolve to an empty list.

    Parameters
    ----------
    fixtures : dict[str, list[SearchHit]] | None
        Mapping from query string to the list of hits to return. The list is
        truncated to ``top_k`` at lookup time.
    default : list[SearchHit] | None
        Hits returned for queries not present in ``fixtures``. Defaults to
        an empty list.
    """

    def __init__(
        self,
        fixtures: Dict[str, Iterable[SearchHit]] | None = None,
        default: Iterable[SearchHit] | None = None,
    ) -> None:
        self.fixtures: Dict[str, List[SearchHit]] = {
            q: list(hits) for q, hits in (fixtures or {}).items()
        }
        self.default: List[SearchHit] = list(default or [])
        self.calls: List[tuple[str, int]] = []

    def register(self, query: str, hits: Iterable[SearchHit]) -> None:
        self.fixtures[query] = list(hits)

    async def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        self.calls.append((query, top_k))
        hits = self.fixtures.get(query, self.default)
        logger.debug(
            "StubSearchClient.search q=%r top_k=%d -> %d hits", query, top_k, min(len(hits), top_k)
        )
        return list(hits[:top_k])
