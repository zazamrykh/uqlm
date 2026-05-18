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
Multi-class hallucination detector.

Implements the two-axis taxonomy:

- **violated_support** — *which* source of truth a claim contradicts.
- **violation_type**   — *how* the claim violates that source.

The detector decomposes an LLM answer into atomic claims
(violated-support-agnostic) and then runs a configurable set of
support-specific verifiers on every claim. Each verifier emits a single
``violation_type`` verdict for its own violated_support.

Available violated_supports:

- ``context``     — claim vs retrieved context
- ``factuality``  — claim vs real-world knowledge (external search)
- ``instruction`` — claim vs the user's instruction / constraints
- ``logical``     — claim vs prior claims emitted earlier in the same answer

Each verifier returns a verdict from the uniform vocabulary
``supported | contradiction | baseless | overclaim`` and a numerical
``hallucination_score`` where higher = more hallucination risk.

This package intentionally uses plain ``dict`` / ``list`` payloads (no
dataclasses) to stay consistent with the rest of the ``uqlm`` codebase.
"""

from uqlm.scorers.longform.multiclass._aggregation import (
    SUPPORTED_VERDICTS,
    SUPPORT_NAMES,
    VERDICT_HALLUCINATION_SCORE,
    aggregate_per_support,
    aggregate_overall,
    is_hallucinated_verdict,
    normalize_verdict,
    verdict_to_score,
)
from uqlm.scorers.longform.multiclass.scorer import MultiClassScorer
from uqlm.scorers.longform.multiclass.verifiers import (
    verify_context,
    verify_factuality,
    verify_instruction,
    verify_logical,
)

__all__ = [
    "MultiClassScorer",
    "SUPPORTED_VERDICTS",
    "SUPPORT_NAMES",
    "VERDICT_HALLUCINATION_SCORE",
    "aggregate_per_support",
    "aggregate_overall",
    "is_hallucinated_verdict",
    "normalize_verdict",
    "verdict_to_score",
    "verify_context",
    "verify_factuality",
    "verify_instruction",
    "verify_logical",
]
