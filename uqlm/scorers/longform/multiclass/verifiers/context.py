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
``context`` violated_support verifier.

Evaluates every input claim against the provided Context independently of
the other supports. Each claim triggers a single LLM call; all claims are
processed concurrently via ``asyncio.gather``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from uqlm.scorers.longform.multiclass._parsing import (
    build_axis_verdict,
    parse_verifier_response,
)
from uqlm.utils.prompts.violated_support.context import (
    CONTEXT_SYSTEM_PROMPT,
    get_context_prompt,
)

logger = logging.getLogger(__name__)


async def verify_context(
    *,
    claims: List[dict],
    input_text: str,
    answer: str,
    llm: BaseChatModel,
    **_: Any,
) -> List[dict]:
    """Evaluate every claim against the grounding material in ``input_text``.

    Returns a list of per-support verdict dicts (one per input claim, same
    order) suitable for assignment into ``claim["violated_supports"]["context"]``.
    """
    if not claims:
        return []
    if llm is None:
        raise ValueError("verify_context requires a non-None LLM instance.")

    logger.info("verify_context: dispatching %d concurrent LLM calls", len(claims))
    tasks = [
        _verify_single(claim, input_text=input_text, answer=answer, llm=llm)
        for claim in claims
    ]
    return await asyncio.gather(*tasks)


async def _verify_single(
    claim: dict,
    *,
    input_text: str,
    answer: str,
    llm: BaseChatModel,
) -> dict:
    prompt = get_context_prompt(
        input_text=input_text,
        answer=answer,
        claim=claim["claim"],
        anchor_text=claim.get("anchor_text", ""),
    )
    messages = [
        SystemMessage(CONTEXT_SYSTEM_PROMPT),
        HumanMessage(prompt),
    ]
    try:
        generation = await llm.ainvoke(messages)
        raw_text = generation.content
    except Exception as exc:  # pragma: no cover - network / model failures
        logger.warning("verify_context: LLM call failed: %s", exc)
        raw_text = ""

    payload = parse_verifier_response(raw_text)
    return build_axis_verdict(payload, raw_text=raw_text, prompt=prompt)
