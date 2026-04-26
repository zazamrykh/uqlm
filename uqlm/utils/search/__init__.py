"""
Search clients for external factuality verification.

This subpackage provides a light abstraction over web search backends used by
``ExternalVerifier`` to gather evidence snippets for baseless/overclaim claims.

The central type is :class:`SearchClient`, a :class:`typing.Protocol` with a
single ``async def search(query, top_k)`` method. Concrete implementations:

- :class:`YandexXmlSearchClient` — real Yandex XML Search API client.
- :class:`CachedSearchClient` — on-disk JSON cache wrapping another client,
  used by the sanity-check notebook to make reruns cheap and deterministic.
- :class:`StubSearchClient` — returns registered fixtures; used in tests.
"""

from uqlm.utils.search.base import SearchClient, SearchHit
from uqlm.utils.search.cache import CachedSearchClient
from uqlm.utils.search.stub import StubSearchClient
from uqlm.utils.search.yandex_xml import YandexXmlSearchClient

__all__ = [
    "SearchClient",
    "SearchHit",
    "YandexXmlSearchClient",
    "CachedSearchClient",
    "StubSearchClient",
]
