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
Shared JSON parsing utilities for multi-class verifier responses.

All ``verify_<support>`` functions follow the same protocol:

- They are async.
- They take a list of claim dicts and the (query, context, answer) triple.
- They prompt an LLM once per claim, parallelised via ``asyncio.gather``.
- The LLM is asked to return a single JSON object with at least
  ``reasoning`` and ``verdict`` fields. Some verifiers add ``evidence``,
  ``evidence_snippets`` etc.

This module centralises the parsing so all verifiers behave consistently
when the LLM wraps its response in markdown fences, prepends prose, or
emits a slightly malformed JSON object.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from uqlm.scorers.longform.multiclass._aggregation import (
    normalize_verdict,
    verdict_to_score,
)

logger = logging.getLogger(__name__)


def parse_verifier_response(raw_text: str) -> Optional[Dict[str, Any]]:
    """Parse the LLM JSON object returned by a per-claim verifier.

    Returns ``None`` when no JSON object can be extracted. The verdict and
    score normalisation is left to :func:`build_axis_verdict` so the parser
    stays generic (the same parser is reused by triage prompts that do not
    emit a verdict at all).
    """
    if not raw_text:
        return None

    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        logger.warning(
            "Verifier response parser: no JSON object found "
            "(first 200 chars of raw response): %s",
            raw_text[:200],
        )
        return None

    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError as exc:
        logger.warning("Verifier response JSON parse error: %s", exc)
        return None

    if not isinstance(payload, dict):
        logger.warning(
            "Verifier response parser: expected JSON object, got %s",
            type(payload).__name__,
        )
        return None
    return payload


def build_axis_verdict(
    payload: Optional[Dict[str, Any]],
    *,
    raw_text: str = "",
    prompt: str = "",
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a normalized per-support verdict block from a parsed payload.

    Always returns a dict containing at minimum ``verdict``,
    ``hallucination_score`` and ``reasoning`` so downstream aggregators
    never KeyError. When ``payload`` is ``None`` (parse failure) the verdict
    defaults to ``"baseless"`` — the safest hallucination label that still
    signals "model output was unparseable".
    """
    if payload is None:
        verdict = "baseless"
        reasoning = "Verifier response could not be parsed; defaulting to baseless."
        evidence = []
    else:
        verdict = normalize_verdict(str(payload.get("verdict", "")))
        reasoning = str(payload.get("reasoning", "") or "").strip()
        raw_evidence = payload.get("evidence", payload.get("relevant_context", []))
        if isinstance(raw_evidence, list):
            evidence = [str(e) for e in raw_evidence]
        elif raw_evidence:
            evidence = [str(raw_evidence)]
        else:
            evidence = []

    out: Dict[str, Any] = {
        "reasoning": reasoning,
        "evidence": evidence,
        "verdict": verdict,
        "hallucination_score": verdict_to_score(verdict),
        "raw_response": raw_text,
        "prompt": prompt,
    }
    if extra_fields:
        out.update(extra_fields)
    return out
