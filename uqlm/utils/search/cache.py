"""
On-disk JSON cache wrapper for :class:`SearchClient`.

Used by the sanity-check notebook and by observability tests so that repeated
runs do not hit the live Yandex API and give deterministic results.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import List

from uqlm.utils.search.base import SearchClient, SearchHit

logger = logging.getLogger(__name__)


class CachedSearchClient(SearchClient):
    """Wrap another :class:`SearchClient` with a per-query JSON-on-disk cache.

    Each ``(query, top_k)`` pair is keyed by a SHA-1 digest and stored under
    ``cache_dir`` as a standalone ``.json`` file containing the serialized
    list of :class:`SearchHit` objects.
    """

    def __init__(self, inner: SearchClient, cache_dir: str | Path) -> None:
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _key_path(self, query: str, top_k: int) -> Path:
        raw = f"{query}\x00{top_k}".encode("utf-8")
        digest = hashlib.sha1(raw).hexdigest()
        return self.cache_dir / f"{digest}.json"

    async def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        path = self._key_path(query, top_k)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                hits = [SearchHit.from_dict(h) for h in payload.get("hits", [])]
                logger.debug(
                    "CachedSearchClient cache HIT q=%r top_k=%d hits=%d",
                    query,
                    top_k,
                    len(hits),
                )
                return hits
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("CachedSearchClient failed to read %s: %s", path, exc)

        hits = await self.inner.search(query=query, top_k=top_k)

        async with self._lock:
            try:
                payload = {
                    "query": query,
                    "top_k": top_k,
                    "hits": [h.to_dict() for h in hits],
                }
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.debug(
                    "CachedSearchClient cache MISS q=%r top_k=%d hits=%d stored=%s",
                    query,
                    top_k,
                    len(hits),
                    path.name,
                )
            except OSError as exc:
                logger.warning("CachedSearchClient failed to write %s: %s", path, exc)

        return hits
