"""
Prompt template for Stage 3 — external factuality verification.

Given a single claim and a numbered list of search-result snippets, the LLM
must decide whether real-world evidence supports / contradicts / overclaims /
is unable to verify the claim. The output schema forces ``reasoning`` before
the final ``world_verdict`` so the model thinks step by step.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Sequence

logger = logging.getLogger(__name__)


EXTERNAL_FACTUALITY_SYSTEM_PROMPT = """\
You are a rigorous fact-checking assistant. Given a single atomic claim and a numbered list of \
search-result snippets retrieved from the web, you must decide whether the real-world evidence \
supports, contradicts, overclaims, or fails to verify the claim.

You must respond with a single valid JSON object and nothing else — no markdown fences, no \
explanation outside the JSON.\
"""


WORLD_VERDICT_VALUES = ("supported", "contradicted", "baseless", "overclaim", "unknown")


@dataclass
class SnippetForPrompt:
    """Minimal view over a :class:`SearchHit` used to render the prompt."""

    index: int
    domain: str
    url: str
    snippet: str


def format_snippets(snippets: Sequence[SnippetForPrompt]) -> str:
    """Render snippets as a numbered Markdown-like block."""
    if not snippets:
        return "(no snippets available)"
    lines: List[str] = []
    for s in snippets:
        lines.append(f"[{s.index}] ({s.domain}) {s.url}")
        snippet_text = s.snippet.strip() or "(no snippet text)"
        lines.append(f"    {snippet_text}")
    return "\n".join(lines)


def get_external_factuality_prompt(
    claim: str,
    snippets: Sequence[SnippetForPrompt],
    context_reasoning: str | None = None,
) -> str:
    """Build the Stage 3 user prompt.

    Parameters
    ----------
    claim : str
        The atomic claim to verify.
    snippets : sequence of SnippetForPrompt
        Evidence snippets collected from web search, already de-duplicated and
        ranked.
    context_reasoning : str, optional
        Stage-1 reasoning that led the model to mark the claim as baseless /
        overclaim. Passed as a hint so the world verifier can focus on the
        right aspect.
    """
    context_hint = ""
    if context_reasoning:
        context_hint = f"## Prior reasoning (from context verification)\n{context_reasoning}\n\n"

    snippets_block = format_snippets(snippets)

    prompt = f"""\
{context_hint}## Claim to verify against the real world
{claim}

## Evidence snippets
Each snippet is preceded by an index in brackets and the source domain / URL.

{snippets_block}

## Task
Using ONLY the evidence snippets above, decide the relationship between the claim and real-world \
knowledge. Do not rely on memory beyond what the snippets state.

Possible verdicts:
- **"supported"** — one or more snippets clearly confirm the claim.
- **"contradicted"** — one or more snippets clearly contradict the claim.
- **"overclaim"** — the snippets support a weaker form of the statement, but the claim is \
strictly stronger than the evidence justifies (for example, the snippet mentions a contribution \
but the claim asserts sole causation; the snippet says "some" while the claim says "all").
- **"baseless"** — the snippets are off-topic, too vague, or simply do not address the claim.
- **"unknown"** — the snippets are contradictory, missing, or otherwise insufficient to decide \
between the above.

## Output format
Respond with EXACTLY ONE JSON object with the following fields, in this order (world_verdict is \
ALWAYS the last field so you reason before committing):

{{
  "reasoning": "brief explanation grounded in specific snippet indices",
  "used_snippet_indices": [0, 2],
  "world_verdict": "one of: {', '.join(WORLD_VERDICT_VALUES)}"
}}
"""
    return prompt


_CODE_FENCE_OPEN_RE = re.compile(r"```(?:json)?\s*", re.IGNORECASE)
_CODE_FENCE_CLOSE_RE = re.compile(r"```\s*$")


def parse_external_factuality_response(raw_text: str) -> dict:
    """Parse the Stage 3 LLM response into a normalized dict.

    The return dict always contains the keys ``world_verdict``, ``reasoning``,
    and ``used_snippet_indices``. On parse failure ``world_verdict`` defaults
    to ``"unknown"`` and ``used_snippet_indices`` to an empty list.
    """
    cleaned = _CODE_FENCE_OPEN_RE.sub("", raw_text or "").strip()
    cleaned = _CODE_FENCE_CLOSE_RE.sub("", cleaned).strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        logger.warning(
            "external_factuality: no JSON object found. Raw (first 200): %r", (raw_text or "")[:200]
        )
        return {"world_verdict": "unknown", "reasoning": "", "used_snippet_indices": []}

    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("external_factuality: JSON decode error: %s", exc)
        return {"world_verdict": "unknown", "reasoning": "", "used_snippet_indices": []}

    if not isinstance(obj, dict):
        logger.warning("external_factuality: expected object, got %s", type(obj).__name__)
        return {"world_verdict": "unknown", "reasoning": "", "used_snippet_indices": []}

    verdict = str(obj.get("world_verdict", "")).strip().lower()
    if verdict not in WORLD_VERDICT_VALUES:
        logger.warning("external_factuality: unknown verdict %r, coercing to 'unknown'", verdict)
        verdict = "unknown"

    reasoning = str(obj.get("reasoning", "") or "").strip()

    raw_indices = obj.get("used_snippet_indices", []) or []
    used_indices: List[int] = []
    if isinstance(raw_indices, list):
        for idx in raw_indices:
            try:
                used_indices.append(int(idx))
            except (TypeError, ValueError):
                continue

    return {
        "world_verdict": verdict,
        "reasoning": reasoning,
        "used_snippet_indices": used_indices,
    }
