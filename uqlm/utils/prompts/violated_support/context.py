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
Prompts for the ``context`` violated_support verifier.

The verifier judges a single atomic claim against any **grounding material**
present inside the Input text — retrieved documents, JSON, citations,
structured data, etc. The Input text may also contain the user's
instruction; the verifier must look past it and focus only on the parts
that are factual source material the model was supposed to use.

If the Input text contains no such grounding material (e.g. a chat-style
question with no documents), the verifier must return ``supported`` for
the claim — there is no context to violate.
"""

from __future__ import annotations


CONTEXT_SYSTEM_PROMPT = """\
You are a fact-checking assistant. You judge whether a single atomic claim \
extracted from a model's answer is faithful to the grounding material \
contained in the Input text. You must respond with a valid JSON object and \
nothing else — no markdown fences, no explanation outside the JSON.\
"""


def get_context_prompt(
    *,
    input_text: str,
    answer: str,
    claim: str,
    anchor_text: str,
) -> str:
    """Render the user prompt for the context-support verifier.

    Parameters
    ----------
    input_text:
        Everything the model saw when it produced ``answer``. May contain
        the user's instruction, retrieved documents, JSON, citations,
        structured data, or just a chat-style question. The verifier must
        decide for itself what counts as "grounding material" (i.e. the
        Context) and ignore the rest.
    answer:
        The full Answer (for orientation only — the verdict is about ``claim``).
    claim:
        The atomic claim to evaluate.
    anchor_text:
        The verbatim span in the Answer that this claim was extracted from.
    """
    return f"""\
## Input text (everything the model saw)
{input_text}

## Answer (for reference)
{answer}

## Claim to evaluate
"{claim}"

The claim was extracted from this span of the Answer:
"{anchor_text}"

## Task

Step 1 — identify grounding material inside the Input text.

"Grounding material" means factual source content the model was supposed \
to use as its source of truth: retrieved documents, JSON / structured \
data, citations, passages, transcripts, knowledge-base snippets. It does \
NOT include the user's instruction itself, framing language, or chat-style \
question.

If the Input text contains **no grounding material** (e.g. a plain chat \
question with no documents), set the verdict to **supported** — there is \
nothing for the Claim to contradict.

Step 2 — when grounding material IS present, evaluate the Claim ONLY \
against it. You are not evaluating world knowledge, the user's \
instruction, or internal logical consistency — those are checked by \
separate verifiers.

Choose exactly one verdict from this set:

- **supported**     — the Claim is directly entailed or confirmed by the \
grounding material, OR the Input text contains no grounding material to \
evaluate against.
- **baseless**      — grounding material IS present and the Claim \
introduces information that is not mentioned or implied anywhere in it \
(the grounding is silent on this claim).
- **overclaim**     — the grounding material supports a weaker form of \
the Claim, but the Claim is strictly stronger than the grounding \
justifies (e.g. grounding says "some", Claim says "all"; grounding says \
"contributed to", Claim says "caused").
- **contradiction** — the Claim directly conflicts with the grounding \
material. This includes wrong dates / numbers / names / relationships \
compared to the grounding, and any wording that would give the reader a \
false understanding of what the grounding states.

## Output format

Return ONLY a JSON object with these fields, in this exact order:

{{
  "reasoning": "brief explanation; if you concluded no grounding is present, say so explicitly",
  "evidence": ["short verbatim excerpt(s) from the grounding material you used"],
  "verdict": "supported" | "baseless" | "overclaim" | "contradiction"
}}
"""
