"""
External factuality verification for baseless / overclaim claims.

This module provides:

- :class:`ClaimForExternal` and :class:`ExternalVerdict` dataclasses that
  define the IO contract of Stage 3.
- :class:`BaselessChecker` protocol so different strategies (web search,
  semantic entropy, internal re-retrieval) are interchangeable.
- :class:`SearchBasedChecker` — the MVP implementation that issues search
  queries via a :class:`SearchClient`, feeds the resulting snippets to an
  LLM via the Stage 3 prompt, and returns a structured verdict.
- :class:`ExternalVerifier` — batches a flat list of claims through the
  injected checker concurrently.

The design intentionally keeps the checker stateless w.r.t. which claim is
being verified: the flat list `list[ClaimForExternal]` makes it easy to
pipe claims from multiple answers through a single `gather()` call.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from uqlm.utils.prompts.external_factuality_prompts import (
    EXTERNAL_FACTUALITY_SYSTEM_PROMPT,
    SnippetForPrompt,
    get_external_factuality_prompt,
    parse_external_factuality_response,
)
from uqlm.utils.search.base import SearchClient, SearchHit

logger = logging.getLogger(__name__)


@dataclass
class ClaimForExternal:
    """Input to the external factuality checker.

    Parameters
    ----------
    claim : str
        The atomic claim to verify.
    search_queries : list[str]
        1..N candidate queries emitted by Stage 1. Only the first
        ``max_queries_per_claim`` are actually used.
    context_reasoning : str | None
        Optional hint carrying the Stage-1 reasoning for transparency.
    key : object | None
        Opaque identifier used by the caller to map verdicts back to the
        original ``(answer_idx, claim_idx)``. Not interpreted by the checker.
    """

    claim: str
    search_queries: List[str]
    context_reasoning: Optional[str] = None
    key: object = None


@dataclass
class ExternalVerdict:
    """Output of the external factuality checker."""

    world_verdict: str  # one of supported|contradicted|baseless|overclaim|unknown
    reasoning: str = ""
    used_snippet_indices: List[int] = field(default_factory=list)
    evidence_snippets: List[str] = field(default_factory=list)
    evidence_urls: List[str] = field(default_factory=list)
    raw_response: str = ""
    world_prompt: str = ""   # full Stage 3 user prompt (for observability / debugging)
    key: object = None


class BaselessChecker(Protocol):
    """Strategy for verifying a single baseless / overclaim claim."""

    async def check(self, claim: ClaimForExternal) -> ExternalVerdict: ...


class SearchBasedChecker:
    """Search-based :class:`BaselessChecker` backed by an :class:`SearchClient`.

    Parameters
    ----------
    search_client : SearchClient
        Any implementation of the :class:`SearchClient` protocol.
    llm : BaseChatModel
        LangChain chat model used for Stage 3 reasoning.
    max_snippets_per_claim : int, default=5
        Hard cap on the number of snippets fed to the LLM per claim.
    max_queries_per_claim : int, default=2
        Hard cap on the number of search queries issued per claim.
    snippets_per_query : int, default=5
        ``top_k`` forwarded to :py:meth:`SearchClient.search`.
    """

    def __init__(
        self,
        search_client: SearchClient,
        llm: BaseChatModel,
        max_snippets_per_claim: int = 5,
        max_queries_per_claim: int = 2,
        snippets_per_query: int = 5,
    ) -> None:
        if search_client is None:
            raise ValueError("SearchBasedChecker requires a SearchClient instance")
        if llm is None:
            raise ValueError("SearchBasedChecker requires an LLM instance")
        self.search_client = search_client
        self.llm = llm
        self.max_snippets_per_claim = max_snippets_per_claim
        self.max_queries_per_claim = max_queries_per_claim
        self.snippets_per_query = snippets_per_query

    async def check(self, claim: ClaimForExternal) -> ExternalVerdict:
        queries = [q for q in (claim.search_queries or []) if q and q.strip()][
            : self.max_queries_per_claim
        ]
        logger.debug(
            "SearchBasedChecker.check claim=%r queries=%s", claim.claim[:120], queries
        )

        if not queries:
            return ExternalVerdict(
                world_verdict="unknown",
                reasoning="No search queries supplied by Stage 1.",
                key=claim.key,
            )

        # Gather snippets from all queries concurrently, dedupe by URL.
        hits_per_query = await asyncio.gather(
            *(self.search_client.search(q, top_k=self.snippets_per_query) for q in queries),
            return_exceptions=True,
        )

        deduped: List[SearchHit] = []
        seen_urls: set[str] = set()
        for idx, hits in enumerate(hits_per_query):
            if isinstance(hits, BaseException):
                logger.warning(
                    "SearchBasedChecker: search failed for q=%r: %s", queries[idx], hits
                )
                continue
            for h in hits:
                if not h.url or h.url in seen_urls:
                    continue
                seen_urls.add(h.url)
                deduped.append(h)
                if len(deduped) >= self.max_snippets_per_claim:
                    break
            if len(deduped) >= self.max_snippets_per_claim:
                break

        logger.debug(
            "SearchBasedChecker: claim=%r gathered %d unique hits",
            claim.claim[:120],
            len(deduped),
        )

        if not deduped:
            return ExternalVerdict(
                world_verdict="unknown",
                reasoning="No search results returned by the search backend.",
                key=claim.key,
            )

        snippets = [
            SnippetForPrompt(
                index=i,
                domain=h.domain,
                url=h.url,
                snippet=h.snippet,
            )
            for i, h in enumerate(deduped)
        ]
        prompt = get_external_factuality_prompt(
            claim=claim.claim,
            snippets=snippets,
            context_reasoning=claim.context_reasoning,
        )
        messages = [
            SystemMessage(EXTERNAL_FACTUALITY_SYSTEM_PROMPT),
            HumanMessage(prompt),
        ]

        try:
            generation = await self.llm.ainvoke(messages)
            raw_text = getattr(generation, "content", "") or ""
        except Exception as exc:
            logger.warning("SearchBasedChecker: LLM call failed: %s", exc)
            return ExternalVerdict(
                world_verdict="unknown",
                reasoning=f"LLM call failed: {exc}",
                key=claim.key,
            )

        parsed = parse_external_factuality_response(raw_text)

        used_idx = [i for i in parsed["used_snippet_indices"] if 0 <= i < len(deduped)]
        evidence_snippets = [deduped[i].snippet for i in used_idx]
        evidence_urls = [deduped[i].url for i in used_idx]

        verdict = ExternalVerdict(
            world_verdict=parsed["world_verdict"],
            reasoning=parsed["reasoning"],
            used_snippet_indices=used_idx,
            evidence_snippets=evidence_snippets,
            evidence_urls=evidence_urls,
            raw_response=raw_text,
            world_prompt=prompt,
            key=claim.key,
        )
        logger.debug(
            "SearchBasedChecker: claim=%r -> %s (evidence=%d)",
            claim.claim[:120],
            verdict.world_verdict,
            len(evidence_urls),
        )
        return verdict


class ExternalVerifier:
    """Batch dispatcher that runs a :class:`BaselessChecker` over many claims."""

    def __init__(self, checker: BaselessChecker, max_concurrency: int = 8) -> None:
        self.checker = checker
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def verify(self, claims: List[ClaimForExternal]) -> List[ExternalVerdict]:
        if not claims:
            return []

        async def _run(claim: ClaimForExternal) -> ExternalVerdict:
            async with self._semaphore:
                try:
                    return await self.checker.check(claim)
                except Exception as exc:  # defensive: never let one claim crash the batch
                    logger.exception("ExternalVerifier: checker failed for claim %r", claim.claim[:120])
                    return ExternalVerdict(
                        world_verdict="unknown",
                        reasoning=f"Checker raised: {exc}",
                        key=claim.key,
                    )

        return list(await asyncio.gather(*(_run(c) for c in claims)))
