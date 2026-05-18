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
Per-violated_support verifier functions.

Every verifier follows the same async signature:

    async def verify_<support>(
        *,
        claims: list[dict],   # produced by ResponseDecomposer.decompose_multiclass
        query: str,
        context: str,
        answer: str,
        llm,                  # LangChain BaseChatModel
        **support_specific_kwargs,
    ) -> list[dict]

It returns one dict per input claim in the same order, with at minimum the
fields ``verdict``, ``hallucination_score``, ``reasoning``, ``evidence``,
``raw_response`` and ``prompt``. Support-specific extra fields (e.g.
``evidence_urls`` for factuality) may also be present.
"""

from uqlm.scorers.longform.multiclass.verifiers.context import verify_context
from uqlm.scorers.longform.multiclass.verifiers.factuality import verify_factuality
from uqlm.scorers.longform.multiclass.verifiers.instruction import verify_instruction
from uqlm.scorers.longform.multiclass.verifiers.logical import verify_logical

__all__ = [
    "verify_context",
    "verify_factuality",
    "verify_instruction",
    "verify_logical",
]
