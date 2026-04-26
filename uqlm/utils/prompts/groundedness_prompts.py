"""
Prompt templates for the single-prompt groundedness scorer.

This module provides a single-call prompt that simultaneously decomposes an
answer into atomic claims and verifies each claim against the provided
context. When ``enable_external_verification`` is set, the prompt also asks
the model to emit, for each baseless / overclaim claim, a ``need_external_check``
flag and 1..3 ``search_queries`` that will be used downstream to gather
evidence from a web search engine.
"""

from __future__ import annotations


UNIFIED_GROUNDEDNESS_SYSTEM_PROMPT = """\
You are an expert fact-checking assistant specialized in detecting hallucinations in AI-generated text.
Your task is to decompose an answer into atomic factual claims and verify each claim against the provided context.
You must respond with a valid JSON array and nothing else — no markdown fences, no explanation.\
"""


def get_unified_groundedness_prompt(
    context: str,
    answer: str,
    include_reasoning: bool = True,
    include_relevant_context: bool = True,
    enable_external_verification: bool = False,
) -> str:
    """Build the user prompt for unified decomposition + groundedness verification.

    Parameters
    ----------
    context : str
        The retrieved context (one or more document chunks) used as the ground
        truth source.
    answer : str
        The generated answer to be analyzed.
    include_reasoning : bool, default=True
        If True, the prompt asks the LLM to provide a short rationale before
        the verdict. Improves accuracy at the cost of longer output.
    include_relevant_context : bool, default=True
        If True, the prompt asks the LLM to quote relevant context excerpts
        per claim. Helps the model ground its verdict in specific evidence.
    enable_external_verification : bool, default=False
        If True, expand the verdict vocabulary with ``"overclaim"`` and ask the
        model, per claim, for two additional fields:

        - ``need_external_check`` — boolean flag that is True only when the
          claim is ``baseless`` / ``overclaim`` AND it looks worth checking
          via web search (contains named entities, numbers, dates, etc.).
        - ``search_queries`` — 1..3 short (<=40 words, <=400 chars) search
          queries useful to verify the claim against world knowledge.

        ``verdict`` always remains the last field so the LLM reasons before
        committing to a label.

    Returns
    -------
    str
        The formatted user prompt.
    """
    # ------------------------------------------------------------------
    # Per-claim field list. Ordering matters: reasoning-like fields first,
    # verdict always last so the model reasons before committing.
    # ------------------------------------------------------------------
    field_lines = [
        '1. "claim" — a single atomic factual statement (subject + verb + object). '
        "Each claim must contain exactly one independent fact.",
        (
            '2. "anchor_text" — the EXACT verbatim contiguous substring from the Answer '
            "that this claim was extracted from. Copy it character-by-character from the Answer. "
            "It must be findable in the Answer via exact string matching."
        ),
    ]
    field_idx = 3

    if include_relevant_context:
        field_lines.append(
            f'{field_idx}. "relevant_context" — list of short verbatim excerpts from the Context '
            "that are relevant to this claim (empty list [] if none found)"
        )
        field_idx += 1

    if include_reasoning:
        field_lines.append(
            f'{field_idx}. "reasoning" — a brief explanation of why you chose the verdict'
        )
        field_idx += 1

    if enable_external_verification:
        field_lines.append(
            f'{field_idx}. "need_external_check" — boolean. Set to true only if the verdict '
            'would be "baseless" or "overclaim" AND the claim is a checkable world-knowledge '
            "fact (contains named entities, numbers, dates, events, scientific or historical "
            "statements) that is worth verifying via a web search. Otherwise set false."
        )
        field_idx += 1
        field_lines.append(
            f'{field_idx}. "search_queries" — list of 1..3 concise web search queries that, if '
            "executed, would help confirm or refute this claim against world knowledge. Each "
            "query must be ≤ 40 words and ≤ 400 characters. Use an empty list [] when "
            '"need_external_check" is false.'
        )
        field_idx += 1

    verdict_values = '"supported", "baseless", "contradicted"'
    if enable_external_verification:
        verdict_values = '"supported", "baseless", "contradicted", "overclaim"'
    field_lines.append(
        f'{field_idx}. "verdict" — exactly one of: {verdict_values} (this MUST be the last field)'
    )

    fields_str = "\n".join(field_lines)

    # ------------------------------------------------------------------
    # Verdict definitions block.
    # ------------------------------------------------------------------
    verdict_defs = [
        '- **"supported"** — the claim is directly entailed or confirmed by the Context.',
        '- **"baseless"** — the claim introduces information not mentioned or implied anywhere '
        "in the Context.",
        '- **"contradicted"** — the claim conflicts with the Context. This includes:\n'
        "  - Direct factual contradictions (Context says X, Answer says Y)\n"
        "  - Distorted or altered details that change the meaning (wrong dates, wrong numbers, "
        "wrong names, wrong relationships)\n"
        "  - Any claim that would give the reader a false understanding of what the Context states.",
    ]
    if enable_external_verification:
        verdict_defs.append(
            '- **"overclaim"** — the Context supports a weaker form of the statement, but the '
            "claim is strictly stronger than the Context justifies (e.g. Context says 'some', "
            "Answer says 'all'; Context says 'contributed to X', Answer says 'caused X')."
        )

    verdict_defs_str = "\n".join(verdict_defs)

    # ------------------------------------------------------------------
    # JSON example schema. verdict is always the last field.
    # ------------------------------------------------------------------
    example_lines = [
        '    "claim": "...",',
        '    "anchor_text": "..."',
    ]
    if include_relevant_context:
        example_lines[-1] += ","
        example_lines.append('    "relevant_context": ["..."]')
    if include_reasoning:
        example_lines[-1] += ","
        example_lines.append('    "reasoning": "..."')
    if enable_external_verification:
        example_lines[-1] += ","
        example_lines.append('    "need_external_check": false,')
        example_lines.append('    "search_queries": []')
    example_lines[-1] += ","
    example_lines.append('    "verdict": "supported"')
    example_fields = "\n".join(example_lines)

    # ------------------------------------------------------------------
    # Optional extra rules for external verification.
    # ------------------------------------------------------------------
    external_rules = ""
    if enable_external_verification:
        external_rules = """
- For every claim with verdict "supported" or "contradicted", set "need_external_check" to false \
and "search_queries" to [].
- Set "need_external_check" to true only when the verdict is "baseless" or "overclaim" AND the \
claim is a checkable world-knowledge fact (named entities, numbers, dates, events, etc.). \
Subjective claims, opinions, and trivially nonsensical claims must have "need_external_check" \
false and "search_queries" [].
- "search_queries" must each be ≤ 40 words and ≤ 400 characters.
"""

    prompt = f"""\
## Context (source of truth)
{context}

## Answer to verify
{answer}

## Task

You must perform TWO steps:

### Step 1: Decompose the Answer into atomic claims

Go through the Answer sentence by sentence. For each sentence, break it into independent atomic \
facts. Each fact should be in the form "subject + verb + object" and contain exactly one piece of \
information. Do NOT use pronouns (he, she, it, they, this, that) — always use the original subject. \
Every factual statement in the Answer must be covered. Do not skip any sentence.

### Step 2: Verify each claim against the Context

For each claim, determine its verdict:

{verdict_defs_str}

For each claim you MUST provide:
{fields_str}

## Important rules
- Process EVERY sentence in the Answer. Do not skip any part.
- Each sentence should produce at least one claim (unless it contains no factual content).
- "anchor_text" must be copied EXACTLY from the Answer — do not paraphrase, trim, or modify it.
  It must be a contiguous substring that can be located via exact string search in the Answer.
- "verdict" must always be the LAST field in each claim object.{external_rules}

## Output format
Respond with ONLY a JSON array. No markdown, no explanation outside the JSON.

[
  {{
{example_fields}
  }}
]
"""
    return prompt
