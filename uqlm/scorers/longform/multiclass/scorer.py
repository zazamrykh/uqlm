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
``MultiClassScorer`` — top-level orchestrator for the multi-class hallucination
detector.

Responsibilities:

- Decompose each answer into atomic claims (violated-support-agnostic).
- For every violated_support the caller requested at runtime, run the
  corresponding verifier in parallel.
- Aggregate per-support and overall ``hallucination_score`` per answer.
- Return a :class:`UQResult` payload that the service layer consumes.

Each verifier is registered at construction time (one entry per violated
support). The set of registered verifiers determines
:attr:`available_supports`; the caller can pick a subset of those for any
particular ``score()`` call via the ``violated_supports`` argument.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from uqlm.longform.decomposition.response_decomposer import ResponseDecomposer
from uqlm.scorers.longform.multiclass._aggregation import (
    SUPPORT_NAMES,
    aggregate_overall,
    aggregate_per_support,
)
from uqlm.scorers.longform.multiclass.verifiers import (
    verify_context,
    verify_factuality,
    verify_instruction,
    verify_logical,
)
from uqlm.utils.results import UQResult

logger = logging.getLogger(__name__)


# A verifier is any async callable matching the documented protocol.
VerifierFn = Callable[..., Awaitable[List[dict]]]


class MultiClassScorer:
    """Multi-class hallucination detector.

    Parameters
    ----------
    llm:
        Default LangChain chat model used for decomposition and every
        verifier that does not get its own LLM. Required.
    external_verifier:
        Instance of
        :class:`uqlm.scorers.longform.external_verifier.ExternalVerifier`
        (or any duck-typed substitute). Required iff the ``factuality``
        violated_support should be available.
    context_llm, factuality_llm, instruction_llm, logical_llm:
        Optional per-support LLM overrides. Default = ``llm``.
    aggregation:
        ``"mean"`` or ``"min"`` (worst-case) — passed to the per-support
        aggregators.
    decomposition_llm:
        Optional LLM override for the decomposer. Defaults to ``llm``.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        external_verifier: Optional[Any] = None,
        context_llm: Optional[BaseChatModel] = None,
        factuality_llm: Optional[BaseChatModel] = None,
        instruction_llm: Optional[BaseChatModel] = None,
        logical_llm: Optional[BaseChatModel] = None,
        decomposition_llm: Optional[BaseChatModel] = None,
        aggregation: str = "mean",
    ) -> None:
        if llm is None:
            raise ValueError("MultiClassScorer requires a non-None default LLM.")
        if aggregation not in ("mean", "min"):
            raise ValueError(
                f"Invalid aggregation: {aggregation!r}. Expected 'mean' or 'min'."
            )

        self._llm = llm
        self._external_verifier = external_verifier
        self._aggregation = aggregation

        self._decomposer = ResponseDecomposer(
            claim_decomposition_llm=decomposition_llm or llm
        )

        # Register one verifier per available violated_support.
        # Each entry is a tuple ``(verifier_fn, kwargs_for_verifier)``.
        self._verifiers: Dict[str, tuple[VerifierFn, Dict[str, Any]]] = {}

        # Context is always available — no extra dependency.
        self._verifiers["context"] = (
            verify_context,
            {"llm": context_llm or llm},
        )

        # Factuality requires an external verifier.
        if external_verifier is not None:
            self._verifiers["factuality"] = (
                verify_factuality,
                {
                    "llm": factuality_llm or llm,
                    "external_verifier": external_verifier,
                },
            )

        # Instruction and logical have no extra dependencies.
        self._verifiers["instruction"] = (
            verify_instruction,
            {"llm": instruction_llm or llm},
        )
        self._verifiers["logical"] = (
            verify_logical,
            {"llm": logical_llm or llm},
        )

        logger.info(
            "MultiClassScorer initialised. available_supports=%s",
            self.available_supports,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available_supports(self) -> List[str]:
        """List of violated_supports the scorer can evaluate (config-driven)."""
        return [s for s in SUPPORT_NAMES if s in self._verifiers]

    def register_verifier(
        self,
        support: str,
        verifier: VerifierFn,
        **default_kwargs: Any,
    ) -> None:
        """Register or replace a verifier for ``support``.

        Used by Phase 2 (instruction) and Phase 3 (logical) to plug in the
        remaining supports without subclassing.
        """
        if support not in SUPPORT_NAMES:
            raise ValueError(
                f"Unknown violated_support: {support!r}. "
                f"Allowed: {SUPPORT_NAMES}."
            )
        self._verifiers[support] = (verifier, default_kwargs)

    async def score(
        self,
        input_texts: List[str],
        answers: List[str],
        *,
        violated_supports: Optional[Iterable[str]] = None,
        return_prompts: bool = False,
    ) -> UQResult:
        """Score a batch of (input_text, answer) pairs.

        Parameters
        ----------
        input_texts:
            Everything visible to the model when it produced each answer
            (instruction + grounding material intermixed). One per answer.
        answers:
            Parallel list of model answers.
        violated_supports:
            Optional subset of :attr:`available_supports` to evaluate. When
            ``None``, every available support is evaluated.
        return_prompts:
            If True, the rendered decomposer prompts are returned under
            ``result.data["decomposer_prompts"]`` for traceability.
        """
        if len(input_texts) != len(answers):
            raise ValueError(
                "input_texts and answers must have equal length. "
                f"Got {len(input_texts)} and {len(answers)}."
            )

        requested = self._resolve_requested_supports(violated_supports)

        # Stage 1 — decomposition.
        claim_sets = await self._decomposer.decompose_multiclass(
            input_texts=input_texts, answers=answers
        )

        # Initialise the violated_supports dict for every claim.
        for claim_list in claim_sets:
            for claim in claim_list:
                claim["violated_supports"] = {}

        # Stage 2 — per-support verification, one verifier-call per (answer, support).
        per_support_outputs: Dict[str, List[List[dict]]] = {s: [] for s in requested}
        for i, claim_list in enumerate(claim_sets):
            tasks: Dict[str, Awaitable[List[dict]]] = {}
            for support in requested:
                verifier_fn, base_kwargs = self._verifiers[support]
                kwargs = dict(base_kwargs)
                kwargs.setdefault("input_text", input_texts[i])
                kwargs.setdefault("answer", answers[i])
                tasks[support] = verifier_fn(claims=claim_list, **kwargs)

            logger.info(
                "MultiClassScorer: dispatching answer %d/%d -> %d claims x %d supports = %d "
                "verifier batches (each verifier internally fans out across all claims)",
                i + 1, len(claim_sets), len(claim_list), len(tasks),
                len(tasks),
            )
            # Run all supports for this answer concurrently.
            support_results = await asyncio.gather(*tasks.values())
            for support, results in zip(tasks.keys(), support_results):
                per_support_outputs[support].append(results)

        # Merge per-support results into each claim.
        for support, answer_results in per_support_outputs.items():
            for claim_list, results in zip(claim_sets, answer_results):
                for claim, block in zip(claim_list, results):
                    claim["violated_supports"][support] = block

        # Aggregate.
        response_scores: Dict[str, List[float]] = {}
        for support in requested:
            response_scores[support] = [
                aggregate_per_support(claim_list, support, mode=self._aggregation)
                for claim_list in claim_sets
            ]
        overall_response_scores = [
            aggregate_overall(claim_list, requested, mode=self._aggregation)
            for claim_list in claim_sets
        ]

        data: Dict[str, Any] = {
            "input_texts": input_texts,
            "answers": answers,
            "claims_data": claim_sets,
            "response_scores": response_scores,
            "overall_response_scores": overall_response_scores,
        }
        metadata: Dict[str, Any] = {
            "mode": "multiclass",
            "aggregation": self._aggregation,
            "available_supports": self.available_supports,
            "requested_supports": requested,
        }
        if return_prompts:
            data["decomposer_prompts"] = None  # decomposer does not surface them yet
        return UQResult({"data": data, "metadata": metadata})

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_requested_supports(
        self, requested: Optional[Iterable[str]]
    ) -> List[str]:
        if requested is None:
            return list(self.available_supports)
        requested_list = list(requested)
        unknown = [s for s in requested_list if s not in self._verifiers]
        if unknown:
            raise ValueError(
                f"Requested violated_supports {unknown!r} are not available. "
                f"Available: {self.available_supports}."
            )
        # Preserve canonical order for deterministic output.
        return [s for s in SUPPORT_NAMES if s in requested_list]
