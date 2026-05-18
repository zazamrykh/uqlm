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
Constants and aggregation helpers for the multi-class scorer.

Verdict vocabulary
------------------
Exactly four public verdicts are emitted by every support verifier:

- ``supported``     — no hallucination signal on this axis
- ``baseless``      — claim adds information not supported by the relevant axis
- ``overclaim``     — claim is a stronger statement than the axis justifies
- ``contradiction`` — claim directly conflicts with the axis

``hallucination_score`` is a numeric mapping where higher = more hallucination
risk. It is the only score consumed by the orchestrator's aggregators.

For the ``factuality`` violated_support there is an internal debug status
``_debug_status ∈ {"checked", "not_checked"}`` that records whether the
external verifier was actually invoked. The public verdict for a skipped
claim is ``supported`` (i.e. it does not push the response-level
hallucination score up); the debug field is for traceability only.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

SUPPORTED_VERDICTS: Tuple[str, ...] = (
    "supported",
    "baseless",
    "overclaim",
    "contradiction",
)

SUPPORT_NAMES: Tuple[str, ...] = (
    "context",
    "factuality",
    "instruction",
    "logical",
)

# Higher = more hallucination risk. Used by every aggregator.
VERDICT_HALLUCINATION_SCORE = {
    "supported": 0.0,
    "baseless": 0.5,
    "overclaim": 0.5,
    "contradiction": 1.0,
}

_HALLUCINATED_VERDICTS = frozenset({"baseless", "overclaim", "contradiction"})


# ---------------------------------------------------------------------------
# Verdict helpers
# ---------------------------------------------------------------------------


def normalize_verdict(verdict: str) -> str:
    """Normalise a free-form verdict string to the canonical vocabulary.

    Returns one of :data:`SUPPORTED_VERDICTS`. Unrecognised inputs default to
    ``"baseless"`` (the safest non-blocking hallucination label).
    """
    if not isinstance(verdict, str):
        return "baseless"
    v = verdict.strip().lower()
    # Accept a few common synonyms.
    if v in {"contradicted", "contradict"}:
        v = "contradiction"
    if v in SUPPORTED_VERDICTS:
        return v
    return "baseless"


def verdict_to_score(verdict: str) -> float:
    """Map a verdict to its hallucination score (higher = worse)."""
    return VERDICT_HALLUCINATION_SCORE[normalize_verdict(verdict)]


def is_hallucinated_verdict(verdict: str) -> bool:
    """Return True for verdicts that count as a hallucination."""
    return normalize_verdict(verdict) in _HALLUCINATED_VERDICTS


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(scores: Iterable[float], mode: str) -> float:
    """Reduce a list of per-claim scores. NaN entries are ignored.

    Returns NaN when the list is empty or contains only NaN values.
    """
    values: List[float] = [float(s) for s in scores if s is not None and not np.isnan(s)]
    if not values:
        return float("nan")
    if mode == "mean":
        return float(np.mean(values))
    if mode == "min":
        # In "higher = worse" convention, the most-conservative aggregator is
        # ``max`` (the most-hallucinated claim drives the response score).
        # We keep the name ``min`` for API compatibility with the older
        # ``aggregation`` parameter but interpret it as "most pessimistic".
        return float(np.max(values))
    raise ValueError(f"Unknown aggregation mode: {mode!r}. Expected 'mean' or 'min'.")


def aggregate_per_support(claims: List[dict], support: str, mode: str = "mean") -> float:
    """Aggregate a single support's hallucination scores across an answer.

    Parameters
    ----------
    claims:
        List of per-claim dicts as produced by :class:`MultiClassScorer`.
        Each claim is expected to expose ``claim["violated_supports"][support]``
        with a ``"hallucination_score"`` field. Claims missing the support are
        silently skipped.
    support:
        One of :data:`SUPPORT_NAMES`.
    mode:
        ``"mean"`` (default) or ``"min"`` (worst-case).
    """
    scores: List[float] = []
    for claim in claims:
        block = (claim.get("violated_supports") or {}).get(support)
        if block is None:
            continue
        scores.append(block.get("hallucination_score", float("nan")))
    return _aggregate(scores, mode=mode)


def aggregate_overall(
    claims: List[dict],
    supports: Iterable[str],
    mode: str = "mean",
) -> float:
    """Aggregate hallucination scores across *all* requested supports.

    The overall score is the mean of every (claim, support) hallucination
    score actually computed for this answer. Claims with no recorded support
    contribute nothing; supports that were not requested are absent from
    ``claim["violated_supports"]`` and therefore ignored automatically.
    """
    scores: List[float] = []
    supports = list(supports)
    for claim in claims:
        per_support = claim.get("violated_supports") or {}
        for sup in supports:
            block = per_support.get(sup)
            if block is None:
                continue
            scores.append(block.get("hallucination_score", float("nan")))
    return _aggregate(scores, mode=mode)
