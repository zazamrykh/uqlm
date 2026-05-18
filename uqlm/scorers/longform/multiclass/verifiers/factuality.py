# Copyright 2025 CVS Health and/or one of its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
``factuality`` violated_support verifier.

This module orchestrates a two-stage pipeline for every input claim:

- **Stage 2a (triage)**: an LLM call per claim that decides whether the
  claim is worth verifying against the open web, and if so, emits 1..N
  short search queries.
- **Stage 2b (external check)**: the existing
  :class:`uqlm.scorers.longform.external_verifier.ExternalVerifier` runs
  for claims that passed triage. Its result is mapped into the canonical
  multi-class verdict vocabulary (``supported | baseless | overclaim |
  contradiction``).

Claims skipped by triage receive a result block with verdict ``supported``
(hallucination_score = 0.0) but with ``_debug_status = "not_checked"`` so
downstream code can distinguish "verified clean" from "we decided not to
check". Claims that were checked but the external verifier returned
``"unknown"`` get verdict ``baseless`` (conservative default) plus
``_debug_status = "checked"``.

Returned dicts also carry the Stage-2a fields ``need_external_verification``
and ``search_queries`` for traceability.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from uqlm.scorers.longform.multiclass._aggregation import verdict_to_score
from uqlm.scorers.longform.multiclass._parsing import parse_verifier_response
from uqlm.utils.prompts.violated_support.factuality import (
    FACTUALITY_TRIAGE_SYSTEM_PROMPT,
    get_factuality_triage_prompt,
)

logger = logging.getLogger(__name__)


# Maps Stage-2b "world_verdict" vocabulary onto the canonical 4-class vocab.
_WORLD_TO_CANONICAL = {
    "supported": "supported",
    "contradicted": "contradiction",
    "overclaim": "overclaim",
    "baseless": "baseless",
    # Stage-2b ``unknown`` means "evidence was inconclusive". We treat that
    # as a soft hallucination signal ("baseless") so it neither hides the
    # claim entirely nor confidently flags a contradiction.
    "unknown": "baseless",
}


async def verify_factuality(
    *,
    claims: List[dict],
    input_text: str,
    answer: str,
    llm: BaseChatModel,
    external_verifier: Any,
    **_: Any,
) -> List[dict]:
    """Two-stage factuality verification.

    Concurrency is governed by the LLM backend (ollama queues internally
    via ``OLLAMA_NUM_PARALLEL``; the service additionally spreads requests
    across replicas via round-robin) and by the search client's own rate
    limit. We do not add an extra semaphore here.

    Parameters
    ----------
    claims:
        Per-claim dicts from :meth:`ResponseDecomposer.decompose_multiclass`.
    input_text, answer:
        Provided so the triage prompt can disambiguate intent and ignore
        claims that merely paraphrase grounding material in ``input_text``.
    llm:
        LangChain chat model used for Stage 2a (triage).
    external_verifier:
        An instance of
        :class:`uqlm.scorers.longform.external_verifier.ExternalVerifier`
        (or a duck-typed substitute exposing
        ``verify(claims: list[ClaimForExternal]) -> list[ExternalVerdict]``).
    """
    if not claims:
        return []
    if llm is None:
        raise ValueError("verify_factuality requires an LLM for the triage stage.")
    if external_verifier is None:
        raise ValueError(
            "verify_factuality requires an external_verifier instance."
        )

    # Stage 2a — triage every claim concurrently. The LLM backend's own
    # queueing keeps this bounded.
    logger.info(
        "verify_factuality: triage dispatching %d concurrent LLM calls",
        len(claims),
    )
    triage_tasks = [
        _triage_one(claim, input_text=input_text, llm=llm)
        for claim in claims
    ]
    triage_results = await asyncio.gather(*triage_tasks)

    # Build per-claim placeholders. ``checked`` lookups use the (idx,) tuple.
    results: List[dict] = []
    pending_for_external: List[Any] = []  # list[ClaimForExternal]
    for i, (claim, triage) in enumerate(zip(claims, triage_results)):
        block = _build_skipped_block(triage)
        results.append(block)
        if triage["need_external_verification"] and triage["search_queries"]:
            # Lazy import to avoid circular dependency with the legacy
            # ``external_verifier`` module that we still rely on.
            from uqlm.scorers.longform.external_verifier import ClaimForExternal

            pending_for_external.append(
                ClaimForExternal(
                    claim=claim["claim"],
                    search_queries=list(triage["search_queries"]),
                    context_reasoning=triage["reasoning"],
                    key=(i,),
                )
            )

    if pending_for_external:
        verdicts = await external_verifier.verify(pending_for_external)
        for v in verdicts:
            i = v.key[0] if isinstance(v.key, tuple) else int(v.key)
            results[i] = _build_checked_block(triage_results[i], v)

    return results


# ---------------------------------------------------------------------------
# Stage 2a — triage
# ---------------------------------------------------------------------------


async def _triage_one(
    claim: dict,
    *,
    input_text: str,
    llm: BaseChatModel,
) -> dict:
    prompt = get_factuality_triage_prompt(
        input_text=input_text,
        claim=claim["claim"],
        anchor_text=claim.get("anchor_text", ""),
    )
    messages = [
        SystemMessage(FACTUALITY_TRIAGE_SYSTEM_PROMPT),
        HumanMessage(prompt),
    ]
    try:
        generation = await llm.ainvoke(messages)
        raw_text = generation.content
    except Exception as exc:  # pragma: no cover - network / model failures
        logger.warning("Factuality triage LLM call failed: %s", exc)
        raw_text = ""

    payload = parse_verifier_response(raw_text) or {}

    flag = bool(payload.get("need_external_verification", False))
    raw_queries = payload.get("search_queries") or []
    if not isinstance(raw_queries, list):
        raw_queries = [raw_queries]
    queries = [str(q).strip() for q in raw_queries if str(q).strip()]

    if not flag:
        queries = []

    return {
        "need_external_verification": flag,
        "search_queries": queries,
        "reasoning": str(payload.get("reasoning", "") or "").strip(),
        "raw_response": raw_text,
        "prompt": prompt,
    }


# ---------------------------------------------------------------------------
# Block builders
# ---------------------------------------------------------------------------


def _build_skipped_block(triage: dict) -> dict:
    """Build the factuality result for a claim that was NOT externally checked."""
    return {
        # Public fields — same shape as a "real" verdict block.
        "reasoning": triage["reasoning"]
        or "Triage decided this claim does not require external verification.",
        "evidence": [],
        "evidence_snippets": [],
        "evidence_urls": [],
        "verdict": "supported",
        "hallucination_score": 0.0,
        # Stage-2a fields (always present so consumers can branch on them).
        "need_external_verification": triage["need_external_verification"],
        "search_queries": triage["search_queries"],
        # Observability.
        "triage_raw_response": triage["raw_response"],
        "triage_prompt": triage["prompt"],
        "raw_response": "",
        "prompt": "",
        # Internal debug flag.
        "_debug_status": "not_checked",
    }


def _build_checked_block(triage: dict, external_verdict: Any) -> dict:
    """Build the factuality result for a claim that was externally verified."""
    canonical = _WORLD_TO_CANONICAL.get(
        getattr(external_verdict, "world_verdict", "") or "",
        "baseless",
    )
    return {
        "reasoning": getattr(external_verdict, "reasoning", "") or "",
        "evidence": list(getattr(external_verdict, "evidence_snippets", []) or []),
        "evidence_snippets": list(
            getattr(external_verdict, "evidence_snippets", []) or []
        ),
        "evidence_urls": list(getattr(external_verdict, "evidence_urls", []) or []),
        "verdict": canonical,
        "hallucination_score": verdict_to_score(canonical),
        "need_external_verification": triage["need_external_verification"],
        "search_queries": triage["search_queries"],
        "triage_raw_response": triage["raw_response"],
        "triage_prompt": triage["prompt"],
        "raw_response": getattr(external_verdict, "raw_response", "") or "",
        "prompt": getattr(external_verdict, "world_prompt", "") or "",
        "_debug_status": "checked",
        # Preserve the original Stage-2b verdict for debugging.
        "_world_verdict": getattr(external_verdict, "world_verdict", ""),
    }
