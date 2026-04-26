"""
Yandex Cloud Search API v2 client (XML response).

The legacy v1 GET endpoint (``yandex.ru/search/xml?folderid=...``) was
blocked in 2024. The current supported path is:

    POST https://searchapi.api.cloud.yandex.net/v2/web/search
    Authorization: Api-Key <YANDEX_CLOUD_API_KEY>
    Content-Type: application/json

    {
        "folderId": "<YANDEX_CLOUD_FOLDER_ID>",
        "query": {
            "searchType": "SEARCH_TYPE_RU",
            "queryText": "<query>"
        },
        "responseFormat": "FORMAT_XML"
    }

The response is JSON with a ``rawData`` field containing a base64-encoded
Yandex XML document (same ``<yandexsearch>`` schema as v1). We decode it
and parse it with the same ``parse_xml`` static method.

Env vars (see ``.env.example``):
    YANDEX_CLOUD_FOLDER_ID   — Yandex Cloud folder id (b1g...)
    YANDEX_CLOUD_API_KEY     — API key (AQV...) with search-api.executor role
    YANDEX_CLOUD_LR          — optional region code (213 = Moscow)
"""

from __future__ import annotations

import base64
import logging
import os
from typing import List, Optional
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import httpx

from uqlm.utils.search.base import SearchClient, SearchHit

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://searchapi.api.cloud.yandex.net/v2/web/search"
DEFAULT_TIMEOUT = 20.0
MAX_QUERY_CHARS = 400
MAX_QUERY_WORDS = 40

# Yandex Search API v2 search type constants.
SEARCH_TYPE_RU = "SEARCH_TYPE_RU"
SEARCH_TYPE_COM = "SEARCH_TYPE_COM"


class YandexXmlSearchError(RuntimeError):
    """Raised when the Yandex Search API returns an error response."""


def _flatten_text(element: Optional[ET.Element]) -> str:
    """Return the concatenation of all text nodes in ``element``.

    Using ``itertext`` correctly handles inline ``<hlword>`` highlight nodes
    inside ``title`` / ``headline`` / ``passage`` without dropping text.
    Multiple consecutive whitespace characters are collapsed.
    """
    if element is None:
        return ""
    parts = [t for t in element.itertext()]
    return " ".join("".join(parts).split()).strip()


def _trim_query(query: str) -> str:
    """Enforce Yandex query length limits (≤ 400 chars / ≤ 40 words)."""
    words = query.split()
    if len(words) > MAX_QUERY_WORDS:
        words = words[:MAX_QUERY_WORDS]
    trimmed = " ".join(words)
    if len(trimmed) > MAX_QUERY_CHARS:
        trimmed = trimmed[:MAX_QUERY_CHARS].rsplit(" ", 1)[0]
    return trimmed


class YandexXmlSearchClient(SearchClient):
    """Async Yandex Cloud Search API v2 client.

    Parameters
    ----------
    folder_id : str
        Yandex Cloud folder id (``b1g...``).
    api_key : str
        Yandex Cloud API key (``AQV...``) with ``search-api.executor`` role.
    search_type : str, default="SEARCH_TYPE_RU"
        Yandex search type. Use ``"SEARCH_TYPE_COM"`` for global English search.
    endpoint : str, default=DEFAULT_ENDPOINT
        API endpoint. Override for tests.
    timeout : float, default=20.0
        HTTP timeout, seconds.
    http_client : httpx.AsyncClient, optional
        Custom client (e.g., session-scoped). A private client is created
        if not provided and is closed by :py:meth:`aclose`.
    """

    def __init__(
        self,
        folder_id: str,
        api_key: str,
        search_type: str = SEARCH_TYPE_RU,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: float = DEFAULT_TIMEOUT,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        if not folder_id or not api_key:
            raise ValueError(
                "YandexXmlSearchClient requires non-empty folder_id and api_key"
            )
        self.folder_id = folder_id
        self.api_key = api_key
        self.search_type = search_type
        self.endpoint = endpoint
        self.timeout = timeout
        self._owned_client = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=timeout)

    @classmethod
    def from_env(
        cls,
        folder_env: str = "YANDEX_CLOUD_FOLDER_ID",
        key_env: str = "YANDEX_CLOUD_API_KEY",
        **kwargs,
    ) -> "YandexXmlSearchClient":
        """Instantiate from environment variables.

        Raises :class:`RuntimeError` if required variables are missing, so the
        caller can skip live tests gracefully.
        """
        folder_id = os.environ.get(folder_env)
        api_key = os.environ.get(key_env)
        if not folder_id or not api_key:
            raise RuntimeError(
                f"Missing Yandex Cloud credentials: set {folder_env} and {key_env} "
                f"env vars. See .env.example for instructions."
            )
        return cls(folder_id=folder_id, api_key=api_key, **kwargs)

    async def aclose(self) -> None:
        if self._owned_client:
            await self._http.aclose()

    async def __aenter__(self) -> "YandexXmlSearchClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        """Run one search query and return normalized hits.

        Uses Yandex Cloud Search API v2 (POST, JSON body, base64-encoded XML
        response). The ``rawData`` field in the JSON response is base64-decoded
        to obtain the standard ``<yandexsearch>`` XML document.
        """
        trimmed_query = _trim_query(query)
        if not trimmed_query:
            logger.debug("YandexXmlSearchClient.search received empty query")
            return []

        body = {
            "folderId": self.folder_id,
            "query": {
                "searchType": self.search_type,
                "queryText": trimmed_query,
            },
            "responseFormat": "FORMAT_XML",
        }

        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.debug(
            "YandexXmlSearchClient.search q=%r top_k=%d", trimmed_query, top_k
        )

        response = await self._http.post(self.endpoint, json=body, headers=headers)

        if response.status_code in (401, 403):
            try:
                err_json = response.json()
                msg = err_json.get("message") or str(err_json)
            except Exception:
                msg = response.text[:300]
            raise YandexXmlSearchError(
                f"Yandex Search API returned {response.status_code}: {msg}"
            )

        response.raise_for_status()

        try:
            resp_json = response.json()
        except Exception as exc:
            raise YandexXmlSearchError(
                f"Yandex Search API returned non-JSON response: {exc}"
            ) from exc

        raw_data = resp_json.get("rawData") or ""
        if not raw_data:
            logger.warning(
                "YandexXmlSearchClient: empty rawData in response for q=%r", trimmed_query
            )
            return []

        try:
            xml_bytes = base64.b64decode(raw_data)
        except Exception as exc:
            raise YandexXmlSearchError(
                f"Failed to base64-decode rawData: {exc}"
            ) from exc

        hits = self.parse_xml(xml_bytes, top_k=top_k)
        logger.debug(
            "YandexXmlSearchClient.search got %d hits for q=%r", len(hits), trimmed_query
        )
        return hits

    @staticmethod
    def parse_xml(xml_bytes: bytes, top_k: int = 5) -> List[SearchHit]:
        """Parse a Yandex XML response body into :class:`SearchHit` objects.

        Stateless — used both by :py:meth:`search` and by unit tests.
        The XML schema is the same ``<yandexsearch>`` format for both v1 and v2.
        """
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as exc:
            raise YandexXmlSearchError(f"Invalid XML response: {exc}") from exc

        # Yandex wraps an error inside <response><error>...</error></response>.
        error_el = root.find(".//response/error")
        if error_el is not None:
            code = error_el.get("code", "?")
            text = _flatten_text(error_el) or "unknown error"
            raise YandexXmlSearchError(f"Yandex XML API error (code={code}): {text}")

        hits: List[SearchHit] = []
        for rank, doc in enumerate(root.iterfind(".//response/results//doc")):
            if len(hits) >= top_k:
                break
            url = _flatten_text(doc.find("url"))
            if not url:
                continue

            domain = _flatten_text(doc.find("domain"))
            if not domain:
                parsed = urlparse(url)
                domain = parsed.hostname or ""

            title = _flatten_text(doc.find("title"))
            headline = _flatten_text(doc.find("headline"))

            passages = [
                _flatten_text(p)
                for p in doc.iterfind("passages/passage")
                if _flatten_text(p)
            ]
            snippet = " ".join(passages) if passages else headline

            saved_copy_el = doc.find("saved-copy-url")
            saved_copy_url = _flatten_text(saved_copy_el) if saved_copy_el is not None else None

            hits.append(
                SearchHit(
                    url=url,
                    domain=domain,
                    title=title,
                    headline=headline,
                    passages=passages,
                    snippet=snippet,
                    saved_copy_url=saved_copy_url or None,
                    rank=rank,
                )
            )
        return hits
