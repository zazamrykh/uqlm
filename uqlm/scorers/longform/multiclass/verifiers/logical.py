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

"""``logical`` violated_support verifier.

Each claim is judged against the claims that were emitted earlier in the
same answer (ordered by ``start_offset``). The very first claim trivially
gets ``supported`` without an LLM call — there are no prior claims to
contradict.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from uqlm.scorers.longform.multiclass._aggregation import verdict_to_score
from uqlm.scorers.longform.multiclass._parsing import (
    build_axis_verdict,
    parse_verifier_response,
)
from uqlm.utils.prompts.violated_support.logical import (
    LOGICAL_SYSTEM_PROMPT,
    get_logical_prompt,
)

logger = logging.getLogger(__name__)


async def verify_logical(
    *,
    claims: List[dict],
    input_text: str,
    answer: str,
    llm: BaseChatModel,
    **_: Any,
) -> List[dict]:
    """Evaluate every claim against the claims emitted earlier in the answer."""
    if not claims:
        return []
    if llm is None:
        raise ValueError("verify_logical requires a non-None LLM instance.")

    # Sort claims by start_offset so "prior" is well-defined. Claims with
    # offset -1 (anchor not found) fall back to their original order; we
    # use stable sort to preserve input ordering for ties.
    indexed: List[Tuple[int, dict]] = list(enumerate(claims))
    indexed.sort(key=lambda x: (x[1].get("start_offset", -1), x[0]))
    sorted_order = [orig_idx for orig_idx, _ in indexed]

    # Build prior_claims for each position in sorted order.
    prior_lists: List[List[dict]] = []
    for k in range(len(indexed)):
        prior_lists.append([c for _, c in indexed[:k]])

    async def _evaluate(k: int) -> dict:
        orig_idx, claim = indexed[k]
        prior = prior_lists[k]
        if not prior:
            # First claim — no prior context to contradict.
            return _supported_block(reason="First claim in the answer; no prior claims to contradict.")
        return await _verify_single(
            claim,
            input_text=input_text,
            answer=answer,
            prior_claims=prior,
            llm=llm,
        )

    logger.info(
        "verify_logical: dispatching %d evaluations (1 shortcut + %d LLM calls)",
        len(indexed),
        max(0, len(indexed) - 1),
    )
    sorted_results = await asyncio.gather(*(_evaluate(k) for k in range(len(indexed))))

    # Restore original claim order.
    out: List[Optional[dict]] = [None] * len(claims)
    for sorted_pos, orig_idx in enumerate(sorted_order):
        out[orig_idx] = sorted_results[sorted_pos]
    return [r for r in out if r is not None]  # type: ignore[return-value]


async def _verify_single(
    claim: dict,
    *,
    input_text: str,
    answer: str,
    prior_claims: List[dict],
    llm: BaseChatModel,
) -> dict:
    prompt = get_logical_prompt(
        input_text=input_text,
        answer=answer,
        claim=claim["claim"],
        anchor_text=claim.get("anchor_text", ""),
        prior_claims=prior_claims,
    )
    messages = [
        SystemMessage(LOGICAL_SYSTEM_PROMPT),
        HumanMessage(prompt),
    ]
    try:
        generation = await llm.ainvoke(messages)
        raw_text = generation.content
    except Exception as exc:  # pragma: no cover - network / model failures
        logger.warning("verify_logical: LLM call failed: %s", exc)
        raw_text = ""

    payload = parse_verifier_response(raw_text)
    return build_axis_verdict(payload, raw_text=raw_text, prompt=prompt)


def _supported_block(reason: str) -> dict:
    return {
        "reasoning": reason,
        "evidence": [],
        "verdict": "supported",
        "hallucination_score": verdict_to_score("supported"),
        "raw_response": "",
        "prompt": "",
    }
