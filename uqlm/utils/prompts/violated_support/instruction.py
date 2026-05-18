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
Prompts for the ``instruction`` violated_support verifier.

The verifier judges a single atomic claim against the **user's instruction**
contained somewhere inside the Input text. The Input text may also contain
grounding material (documents, JSON, etc.); the verifier must look past it
and focus only on what counts as instructional language ("write …",
"summarise …", "answer using only the data below", scope / format /
length constraints, etc.).
"""

from __future__ import annotations


INSTRUCTION_SYSTEM_PROMPT = """\
You assess whether a single atomic claim extracted from a model's answer \
complies with the user's instruction. You must respond with a valid JSON \
object and nothing else — no markdown fences, no explanation outside the \
JSON.\
"""


def get_instruction_prompt(
    *,
    input_text: str,
    answer: str,
    claim: str,
    anchor_text: str,
) -> str:
    """Render the user prompt for the instruction-support verifier.

    Parameters
    ----------
    input_text:
        Everything the model saw when it produced ``answer``. The verifier
        must locate the user's instruction inside this text on its own.
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

Step 1 — identify the user's instruction inside the Input text. The \
Instruction is whatever the user is asking the model to do: imperative \
phrases ("write …", "list …", "summarise …"), questions, format / scope / \
length constraints. Grounding material (documents, JSON, citations) is \
NOT part of the instruction unless the user explicitly says "use ONLY the \
data below" — in which case that statement IS instructional.

Step 2 — evaluate the Claim against the Instruction. You are not \
evaluating factual correctness against the world, the grounding material, \
or internal logical consistency — those are checked by separate verifiers.

Choose exactly one verdict, specialised to instruction-violation:

- **supported**     — the Claim is consistent with what the Instruction \
asked for. The Answer obeys the Instruction at this fragment.
- **baseless**      — the Claim adds an action, scope, or format that the \
Instruction did not ask for (extra unrequested content).
- **overclaim**     — the Claim is broadly within the Instruction's scope \
but interprets it too widely, doing more / stronger / with more \
confidence than the Instruction warrants.
- **contradiction** — the Claim does something the Instruction explicitly \
forbids, or directly violates a hard constraint (format / length / scope) \
stated in the Instruction.

If the Instruction is silent on the topic raised by the Claim, prefer \
**supported** (no instruction was violated) unless the Claim clearly adds \
unrequested content (then **baseless**).

## Output format

Return ONLY a JSON object with these fields, in this exact order:

{{
  "reasoning": "brief explanation of why you chose the verdict",
  "evidence": ["short verbatim excerpt(s) from the Instruction you used"],
  "verdict": "supported" | "baseless" | "overclaim" | "contradiction"
}}
"""
