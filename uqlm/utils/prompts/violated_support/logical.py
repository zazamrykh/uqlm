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
Prompts for the ``logical`` violated_support verifier.

The verifier judges whether a claim is logically consistent with the
preceding claims emitted earlier in the same answer (e.g. reasoning chain,
math derivation steps). It receives the user's query / context only as
intent context; the actual support is the list of prior claims.
"""

from __future__ import annotations

from typing import List


LOGICAL_SYSTEM_PROMPT = """\
You assess whether a single atomic claim from a model's answer is \
logically consistent with the claims that were stated earlier in the same \
answer. You must respond with a valid JSON object and nothing else — no \
markdown fences, no explanation outside the JSON.\
"""


def _format_prior_claims(prior_claims: List[dict]) -> str:
    if not prior_claims:
        return "(none — this is the first claim in the answer)"
    lines: List[str] = []
    for i, c in enumerate(prior_claims):
        lines.append(f'[{i}] "{c.get("claim", "")}"')
    return "\n".join(lines)


def get_logical_prompt(
    *,
    input_text: str,
    answer: str,
    claim: str,
    anchor_text: str,
    prior_claims: List[dict],
) -> str:
    """Render the user prompt for the logical-support verifier.

    Parameters
    ----------
    input_text:
        Provided as intent context only — it is NOT the support being
        checked. The verifier judges the Claim against the prior claims.
    answer:
        The full Answer (for orientation).
    claim:
        The atomic claim to evaluate.
    anchor_text:
        The verbatim span this claim was extracted from.
    prior_claims:
        Claims emitted earlier in the Answer (ordered by ``start_offset``).
        Each item is a dict with at least ``claim``.
    """
    prior_block = _format_prior_claims(prior_claims)
    return f"""\
## Input text (intent only)
{input_text}

## Full Answer (for orientation)
{answer}

## Prior claims in the Answer
{prior_block}

## Claim under evaluation
"{claim}"

The claim was extracted from this span of the Answer:
"{anchor_text}"

## Task

Evaluate whether the Claim is **logically consistent with the prior \
claims listed above**. You are NOT checking it against the Context, the \
Instruction, or real-world facts — only against what the model itself has \
already stated in this Answer.

Choose exactly one verdict, specialised to the logical-violation:

- **supported**     — the Claim is consistent with the prior claims, or \
follows from them via valid reasoning. If the Claim is the very first one \
in the Answer, default to "supported" (there is nothing to contradict). \
**Use this as the default** when the prior claims simply give independent \
information without setting up an explicit chain of reasoning that the \
Claim then breaks.
- **baseless**      — apply ONLY when the Answer is engaged in a chain \
of reasoning (mathematical derivation, step-by-step argument, logical \
deduction, problem solving) AND the Claim is a clear non-sequitur within \
that chain — i.e. the prior reasoning sets up an expectation and the \
Claim makes an unjustified leap. Examples:
  - prior derivation arrives at `1 - 2 + 3 - 4 + 5 = 3`, and the Claim \
abruptly states "the remainder when 12345 is divided by 11 is 3" without \
linking the two;
  - prior: "the sky is blue today" → Claim: "therefore the giraffe likes \
apples".
  Do NOT flag a Claim as ``baseless`` just because the prior claims do not \
explicitly entail it — most factual answers add independent facts and \
that is fine. The unjustified-leap pattern must be present.
- **overclaim**     — the prior claims set up a chain of reasoning that \
permits a weaker conclusion, but the Claim asserts a strictly stronger \
one (priors permit a possibility → Claim states a certainty).
- **contradiction** — the Claim directly contradicts at least one prior \
claim (e.g. the priors give numbers that do not add up to what the Claim \
asserts; the priors establish A but the Claim asserts not-A).

## Output format

Return ONLY a JSON object with these fields, in this exact order:

{{
  "reasoning": "brief explanation referencing prior claim indices when relevant",
  "evidence": ["short verbatim quotes from the prior claims you used"],
  "verdict": "supported" | "baseless" | "overclaim" | "contradiction"
}}
"""
