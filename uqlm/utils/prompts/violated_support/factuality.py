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
Prompts for the ``factuality`` violated_support.

The factuality verifier works in two LLM stages:

- Stage 2a (triage) — given an atomic claim (and optionally the surrounding
  Query/Context), decide whether searching the web could meaningfully
  resolve the claim. If yes, emit 1..N short search queries.
- Stage 2b (verification) — feed retrieved search snippets to the LLM,
  ask it to compare them with the claim and emit a final verdict.

The Stage 2b prompt is the existing one from
``external_factuality_prompts.py`` and is reused unchanged. This module
only owns the triage prompt.
"""

from __future__ import annotations


FACTUALITY_TRIAGE_SYSTEM_PROMPT = """\
You assess whether a single atomic claim can be meaningfully verified \
against real-world knowledge via web search. You must respond with a valid \
JSON object and nothing else — no markdown fences, no explanation outside \
the JSON.\
"""


def get_factuality_triage_prompt(
    *,
    input_text: str,
    claim: str,
    anchor_text: str,
) -> str:
    """Render the Stage 2a triage prompt for a single claim.

    The prompt asks the LLM to make two decisions:

    1. ``need_external_verification`` — should we even bother running a
       web search for this claim? It is ``true`` only when the claim makes
       a checkable real-world statement that a search engine could plausibly
       confirm or refute (named entities, numbers, dates, events,
       historical / scientific statements, etc.). Pure opinions, format
       decisions, math reasoning steps, or restatements of the provided
       Context get ``false``.
    2. ``search_queries`` — when the flag is ``true``, emit 1..3 concise
       (<=40 words, <=400 chars) web search queries that would help
       confirm or refute the claim against world knowledge.

    Parameters
    ----------
    input_text:
        Everything the model saw. Helps detect claims that merely repeat
        grounding material inside the Input text (and therefore do not
        need external verification).
    claim:
        The atomic claim under triage.
    anchor_text:
        The verbatim span of the Answer the claim was extracted from.
    """
    return f"""\
## Input text (everything the model saw)
{input_text}

## Claim under triage
"{claim}"

The claim was extracted from this span of the Answer:
"{anchor_text}"

## Task

Decide whether this Claim should be verified via web search against \
real-world knowledge.

Set ``need_external_verification`` to **true** only when ALL of the \
following hold:

- The Claim makes a checkable real-world statement (named entities, \
numbers, dates, events, historical / scientific / geographical facts, etc.).
- A general-purpose web search engine could plausibly return evidence \
confirming or refuting it.

Set ``need_external_verification`` to **false** when:

- The Claim is purely subjective (opinion, taste, recommendation).
- The Claim is a formatting / scope decision about the answer itself.
- The Claim is a step of mathematical or logical reasoning rather than a \
world-knowledge statement.
- The Claim merely paraphrases or summarises the Context (its truth \
depends on the Context, not on external knowledge).
- The Claim is trivially true regardless of evidence (tautology).

If ``need_external_verification`` is true, also emit 1..3 ``search_queries``: \
short web search queries (<=40 words, <=400 chars each) most likely to \
return snippets that confirm or refute the Claim. Otherwise emit ``[]``.

## Output format

Return ONLY a JSON object with these fields, in this exact order:

{{
  "reasoning": "brief explanation of why the flag is true or false",
  "search_queries": ["query 1", "query 2"],
  "need_external_verification": true
}}
"""
