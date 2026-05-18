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
This module is used to store LLM prompt templates that can be used for various tasks.
"""


# The claim_brekadown_template is a modified version of the prompt from "Atomic Calibration of LLMs in Long-Form Generations"
# @misc{zhang2025atomiccalibrationllmslongform,
#       title={Atomic Calibration of LLMs in Long-Form Generations},
#       author={Caiqi Zhang and Ruihan Yang and Zhisong Zhang and Xinting Huang and Sen Yang and Dong Yu and Nigel Collier},
#       year={2025},
#       eprint={2410.13246},
#       archivePrefix={arXiv},
#       primaryClass={cs.CL},
#       url={https://arxiv.org/abs/2410.13246},
# }
def get_claim_breakdown_prompt(response: str) -> str:
    """
    Parameters
    ----------
    response: str
        The response to be broken down into fact pieces.

    Returns
    -------
    str
        The prompt template for breaking down the response into fact pieces.
    """

    claim_breakdown_prompt = f"""
    Please breakdown the following passage into independent fact pieces.

    Step 1: For each sentence, you should break it into several fact pieces. Each fact piece should only contain one single independent fact. Normally the format of a fact piece is "subject + verb + object". If the sentence does not contain a verb, you can use "be" as the verb.

    Step 2: Do this for all the sentences. Output each piece of fact in one single line starting with ###. Do not include other formatting.

    Step 3: Each atomic fact should be self-contained. Do not use pronouns as the subject of a piece of fact, such as he, she, it, this that, use the original subject whenever possible.

    Step 4: If the sentence does not contain any independent fact, you should output "### NONE".

    Here are some examples:

    Example 1:
    Michael Collins (born October 31, 1930) is a retired American astronaut and test pilot who was the Command Module Pilot for the Apollo 11 mission in 1969.
    ### Michael Collins was born on October 31, 1930.
    ### Michael Collins is retired.
    ### Michael Collins is an American.
    ### Michael Collins was an astronaut.
    ### Michael Collins was a test pilot.
    ### Michael Collins was the Command Module Pilot.
    ### Michael Collins was the Command Module Pilot for the Apollo 11 mission.
    ### Michael Collins was the Command Module Pilot for the Apollo 11 mission in 1969.

    Example 2:
    League of Legends (often abbreviated as LoL) is a multiplayer online battle arena (MOBA) video game developed and published by Riot Games.
    ### League of Legends is a video game.
    ### League of Legends is often abbreviated as LoL.
    ### League of Legends is a multiplayer online battle arena.
    ### League of Legends is a MOBA video game.
    ### League of Legends is developed by Riot Games.
    ### League of Legends is published by Riot Games.

    Example 3:
    Emory University has a strong athletics program, competing in the National Collegiate Athletic Association (NCAA) Division I Atlantic Coast Conference (ACC). The university's mascot is the Eagle.
    ### Emory University has a strong athletics program.
    ### Emory University competes in the National Collegiate Athletic Association Division I.
    ### Emory University competes in the Atlantic Coast Conference.
    ### Emory University is part of the ACC.
    ### Emory University's mascot is the Eagle.

    Example 4:
    Hi
    ### NONE

    Now it's your turn. Here is the passage:

    {response}

    You should only return the final answer. Now your answer is:
    """

    return claim_breakdown_prompt


def get_factoid_breakdown_template(response: str) -> str:
    """
    Parameters
    ----------
    response: str
        The response to be broken down into fact pieces.
    """

    factoid_template = f"""Please list the specific factual propositions included in the paragraph below. Be complete and do not leave any factual claims out. Provide each claim as a separate sentence in a separate bullet point. Do this for all the sentences. Output each piece of fact in one single line starting with ###. Do not include other formatting.

    Here are some examples:

    Example 1:
    Sir Paul McCartney is a renowned English musician, singer, and songwriter who gained worldwide fame as a member of The Beatles, one of the most influential and successful bands in the history of popular music. Born on June 18, 1942, in Liverpool, England, McCartney's career spans over six decades, during which he has become renowned not only for his role as a bass guitarist and vocalist with The Beatles but also for his songwriting partnership with John Lennon, which produced some of the most celebrated songs of the 20th century. After The Beatles disbanded in 1970, McCartney pursued a successful solo career and formed the band Wings with his late wife, Linda McCartney. His contributions to music have been recognized with numerous awards, including multiple Grammys and an induction into the Rock and Roll Hall of Fame both as a member of The Beatles and as a solo artist. Known for his melodic bass lines, versatile vocals, and enduring performances, McCartney continues to captivate audiences around the world with his artistry and dedication to his craft.
    ### Sir Paul McCartney is a renowned English musician and songwriter who gained worldwide fame as a member of The Beatles.
    ### He formed the band Wings after The Beatles disbanded and has had a successful solo career.

    Example 2:
    John Lennon was an iconic English musician, singer, and songwriter, best known as one of the founding members of The Beatles, the most commercially successful and critically acclaimed band in the history of popular music. Born on October 9, 1940, in Liverpool, Lennon grew up to become a pivotal figure in the cultural revolution of the 1960s. His songwriting partnership with Paul McCartney produced enduring classics that helped define the era. Known for his wit and outspoken personality, Lennon was also a passionate peace activist whose pursuit of social justice resonated with millions worldwide. After The Beatles disbanded, Lennon continued his musical career as a solo artist, producing hits like "Imagine," which became an anthem for peace. Tragically, his life was cut short on December 8, 1980, when he was assassinated in New York City, but his legacy continues to influence generations and inspire artists around the globe.
    ### John Lennon was an iconic English musician, singer, and songwriter. He was best known as one of the founding members of The Beatles.
    ### Born on October 9, 1940, in Liverpool, Lennon grew up to become a pivotal figure in the cultural revolution of the 1960s. His songwriting partnership with Paul McCartney produced enduring classics that helped define the era.
    ### Lennon was also a passionate peace activist whose pursuit of social justice resonated with millions worldwide. After The Beatles disbanded, Lennon continued his musical career as a solo artist, producing hits like "Imagine."

    Here is the paragraph:

    {response}

    You should only return the final answer. Now your answer is:
    """

    return factoid_template


# ---------------------------------------------------------------------------
# Multi-class decomposer prompt (violated-support-agnostic)
# ---------------------------------------------------------------------------
#
# Used by ``MultiClassScorer``. Produces a JSON array of atomic claims with a
# verbatim ``anchor_text`` substring from the answer. The prompt describes the
# four ``violated_supports`` (context / factuality / instruction / logical)
# that will be evaluated downstream so the decomposer extracts every kind of
# proposition any verifier will need, including reasoning steps that a plain
# "factual claim" prompt would skip.
#
# Output schema per item:
#
#   {"claim": "<atomic proposition>", "anchor_text": "<verbatim substring>"}
#
# No verdict and no per-support fields are produced here — decomposition is
# strictly violated-support-agnostic.

MULTICLASS_DECOMPOSITION_SYSTEM_PROMPT = """\
You decompose an LLM answer into atomic claims for downstream hallucination \
detection. You must respond with a valid JSON array and nothing else — no \
markdown fences, no explanation outside the JSON.\
"""


def get_multiclass_decomposition_prompt(input_text: str, answer: str) -> str:
    """Build the user prompt for violated-support-agnostic claim decomposition.

    The detector receives a single ``input_text`` — everything that was
    visible to the model when it produced ``answer``. This single string
    may contain both the user's instruction and grounding material
    (documents, JSON, citations) intermixed; downstream verifiers decide
    which parts they care about.

    Downstream each emitted claim is independently evaluated against four
    "violated_supports":

    - ``context``     — claim vs grounding material inside ``input_text``;
    - ``factuality``  — claim vs real-world knowledge;
    - ``instruction`` — claim vs the user's instruction inside ``input_text``;
    - ``logical``     — claim vs prior claims in the same answer, including
      step-by-step reasoning.

    Parameters
    ----------
    input_text:
        The full input that was visible to the model. May contain the
        user's instruction, retrieved documents, structured data, etc.
        All of it intermixed.
    answer:
        The model's answer; the source of every emitted ``anchor_text``.
    """
    return f"""\
## Input text (everything the model saw)
{input_text}

## Answer to decompose
{answer}

## Task

Decompose the Answer into atomic claims. Each claim must be a single, \
self-contained proposition (typically "subject + verb + object"). Do NOT use \
pronouns (he, she, it, this, that); always restate the original subject.

You are NOT judging whether a claim is correct. You only extract claims.

### What counts as a claim

A downstream system will independently evaluate each claim against four \
violated_supports. You must therefore emit a claim for any proposition that \
*any* of these supports could meaningfully evaluate:

1. **context**     — propositions whose truth depends on grounding material \
inside the Input text (retrieved documents, JSON, citations, structured data).
2. **factuality**  — propositions whose truth depends on real-world \
knowledge that is NOT provided in the Input text.
3. **instruction** — propositions about whether the Answer obeys the user's \
instruction inside the Input text (its scope, format, length, constraints). \
Emit such a claim only when the Answer makes a clear decision that can be \
checked against the instruction.
4. **logical**     — propositions inside a chain of reasoning, including \
intermediate steps. Math / algebra / logical derivation steps must each be \
emitted as a separate claim, even if they would not look like "factual" \
claims in isolation.

If a sentence carries no proposition (greeting, filler), skip it.

### Examples of biographical / world-knowledge claims

Example A:
Michael Collins (born October 31, 1930) is a retired American astronaut and test pilot who was the Command Module Pilot for the Apollo 11 mission in 1969.
### Michael Collins was born on October 31, 1930.
### Michael Collins is retired.
### Michael Collins is an American.
### Michael Collins was an astronaut.
### Michael Collins was a test pilot.
### Michael Collins was the Command Module Pilot for the Apollo 11 mission in 1969.

Example B:
League of Legends (often abbreviated as LoL) is a multiplayer online battle arena (MOBA) video game developed and published by Riot Games.
### League of Legends is a video game.
### League of Legends is often abbreviated as LoL.
### League of Legends is a multiplayer online battle arena.
### League of Legends is a MOBA video game.
### League of Legends is developed by Riot Games.
### League of Legends is published by Riot Games.

### Example of a reasoning chain (logical-support claims)

Example C:
We have x^2 - 3x = 4, so x^2 - 3x - 4 = 0, which factors as (x - 4)(x + 1) = 0.
### x^2 - 3x = 4.
### x^2 - 3x - 4 = 0.
### x^2 - 3x - 4 factors as (x - 4)(x + 1).
### (x - 4)(x + 1) = 0.

(The ``###`` lines above are only illustrative of the granularity expected; \
your final output must be JSON, not lines starting with ``###``.)

### anchor_text rules

For every claim you emit, also return ``anchor_text``: an EXACT verbatim \
contiguous substring of the Answer that the claim was extracted from. Copy \
it character-by-character; ``answer.find(anchor_text)`` must succeed.

- Do not paraphrase the anchor.
- Do not trim punctuation in a way that breaks exact matching.
- For multi-clause sentences, use the smallest contiguous span that fully \
contains the claim.
- If the same proposition appears multiple times in the Answer, anchor it \
to its first occurrence.

### Output format

Return ONLY a JSON array. No markdown fences. No prose around the JSON.

[
  {{
    "claim": "atomic proposition rephrased as a complete sentence",
    "anchor_text": "verbatim substring from the Answer"
  }}
]

If the Answer contains no propositions to extract, return ``[]``.
"""
