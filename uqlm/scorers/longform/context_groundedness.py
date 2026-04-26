"""
Context Groundedness Scorer for RAG hallucination detection.

This module provides a single public scorer, ``ContextGroundednessScorer``,
with two internal scoring modes:

1. ``mode="two_stage"`` — decompose answer into claims, then verify each claim
   against the context with an entailment classifier.
2. ``mode="single_prompt"`` — use one prompt that simultaneously decomposes the
   answer and verifies each claim against the context, returning structured JSON.

Both modes return a unified ``UQResult`` payload so the scorer matches the
broader ``uqlm`` API style.
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from rich.errors import LiveError
from rich.progress import Progress, TextColumn

from uqlm.longform.decomposition import ResponseDecomposer
from uqlm.nli.entailment import EntailmentClassifier
from uqlm.utils.display import (
    ConditionalBarColumn,
    ConditionalSpinnerColumn,
    ConditionalTextColumn,
    ConditionalTimeElapsedColumn,
)
from uqlm.utils.prompts.groundedness_prompts import (
    UNIFIED_GROUNDEDNESS_SYSTEM_PROMPT,
    get_unified_groundedness_prompt,
)
from uqlm.utils.results import UQResult

logger = logging.getLogger(__name__)

VERDICT_SCORE_MAP: Dict[str, float] = {
    "supported": 1.0,
    "baseless": 0.5,
    "overclaim": 0.5,
    "contradicted": 0.0,
    "contradiction": 0.0,
    "unknown": np.nan,
    "not_checked": np.nan,
}

# Verdicts that trigger Stage 3 external factuality verification.
EXTERNAL_CHECK_VERDICTS = frozenset({"baseless", "overclaim"})


class _BaseGroundednessEngine:
    """Shared helpers for groundedness scoring engines."""

    def __init__(self, aggregation: str) -> None:
        if aggregation not in ("mean", "min"):
            raise ValueError(
                f"Invalid aggregation: {aggregation!r}. Must be 'mean' or 'min'."
            )
        self.aggregation = aggregation

    @staticmethod
    def build_progress_bar(show_progress_bars: bool) -> Optional[Progress]:
        """Create and start a rich progress bar, or return None if disabled."""
        if not show_progress_bars:
            return None
        try:
            completion_text = "[progress.percentage]{task.completed}/{task.total}"
            progress_bar = Progress(
                ConditionalSpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                ConditionalBarColumn(),
                ConditionalTextColumn(completion_text),
                ConditionalTimeElapsedColumn(),
            )
            progress_bar.start()
            return progress_bar
        except LiveError:
            logger.debug("Could not create progress bar (LiveError), continuing without it")
            return None

    def aggregate_scores(self, claim_scores: List[List[float]]) -> List[float]:
        """Aggregate claim-level scores to response-level scores."""
        response_scores: List[float] = []
        for scores in claim_scores:
            if not scores:
                response_scores.append(np.nan)
                continue

            valid_scores = [s for s in scores if not np.isnan(s)]
            if not valid_scores:
                response_scores.append(np.nan)
                continue

            if self.aggregation == "mean":
                response_scores.append(float(np.mean(valid_scores)))
            else:
                response_scores.append(float(np.min(valid_scores)))

        return response_scores

    @staticmethod
    def build_claims_data_from_lists(
        claim_sets: List[List[str]],
        claim_labels: List[List[str]],
        claim_scores: List[List[float]],
        raw_judge_responses: Optional[List[List[str]]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Build normalized claim dictionaries from parallel list fields."""
        claims_data: List[List[Dict[str, Any]]] = []
        for i, claim_set in enumerate(claim_sets):
            claims_data_i: List[Dict[str, Any]] = []
            labels_i = claim_labels[i] if i < len(claim_labels) else []
            scores_i = claim_scores[i] if i < len(claim_scores) else []
            raw_i = raw_judge_responses[i] if raw_judge_responses and i < len(raw_judge_responses) else []

            for j, claim in enumerate(claim_set):
                label = labels_i[j] if j < len(labels_i) else "unknown"
                score = scores_i[j] if j < len(scores_i) else np.nan
                raw_response = raw_i[j] if j < len(raw_i) else None
                claims_data_i.append(
                    {
                        "claim": claim,
                        "anchor_text": "",
                        "verdict": label,
                        "score": score,
                        "start_offset": -1,
                        "end_offset": -1,
                        "reasoning": "",
                        "relevant_context": [],
                        "raw_judge_response": raw_response,
                        # External-verification fields: present but empty for modes
                        # that do not run Stage 3 (backward compatible default).
                        "context_verdict": label,
                        "need_external_check": False,
                        "search_queries": [],
                        "world_verdict": "not_checked",
                        "world_score": np.nan,
                        "world_reasoning": "",
                        "evidence_snippets": [],
                        "evidence_urls": [],
                        "raw_world_response": None,
                    }
                )
            claims_data.append(claims_data_i)
        return claims_data

    @staticmethod
    def build_parallel_lists_from_claims_data(
        claims_data: List[List[Dict[str, Any]]]
    ) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
        """Derive convenience list fields from normalized claim dictionaries."""
        claim_sets = [[claim["claim"] for claim in claim_list] for claim_list in claims_data]
        claim_labels = [[claim["verdict"] for claim in claim_list] for claim_list in claims_data]
        claim_scores = [[claim["score"] for claim in claim_list] for claim_list in claims_data]
        return claim_sets, claim_labels, claim_scores

    def build_result(
        self,
        queries: List[str],
        contexts: List[str],
        answers: List[str],
        claims_data: List[List[Dict[str, Any]]],
        metadata: Dict[str, Any],
    ) -> UQResult:
        """Construct a standardized ``UQResult`` payload."""
        claim_sets, claim_labels, claim_scores = self.build_parallel_lists_from_claims_data(
            claims_data
        )
        response_scores = self.aggregate_scores(claim_scores)

        # World-axis parallel views (present regardless of mode so consumers can rely on them).
        claim_world_labels: List[List[str]] = [
            [str(c.get("world_verdict", "not_checked")) for c in claim_list]
            for claim_list in claims_data
        ]
        claim_world_scores: List[List[float]] = [
            [float(c.get("world_score", np.nan)) for c in claim_list]
            for claim_list in claims_data
        ]
        response_world_scores = self.aggregate_scores(claim_world_scores)

        return UQResult(
            {
                "data": {
                    "queries": queries,
                    "contexts": contexts,
                    "answers": answers,
                    "claims_data": claims_data,
                    "claim_sets": claim_sets,
                    "claim_labels": claim_labels,
                    "claim_scores": claim_scores,
                    "response_scores": response_scores,
                    # World-axis views (all "not_checked" / NaN unless Stage 3 ran).
                    "claim_context_labels": claim_labels,
                    "claim_world_labels": claim_world_labels,
                    "claim_world_scores": claim_world_scores,
                    "response_world_scores": response_world_scores,
                },
                "metadata": metadata,
            }
        )


class _TwoStageGroundednessEngine(_BaseGroundednessEngine):
    """Two-stage groundedness engine: decomposition followed by NLI verification."""

    def __init__(
        self,
        nli_llm: BaseChatModel,
        claim_decomposition_llm: Optional[BaseChatModel] = None,
        entailment_style: str = "nli_classification",
        aggregation: str = "mean",
    ) -> None:
        super().__init__(aggregation=aggregation)
        if nli_llm is None:
            raise ValueError(
                "nli_llm is required for ContextGroundednessScorer in two_stage mode. "
                "Provide a LangChain BaseChatModel instance."
            )

        self.nli_llm = nli_llm
        self.entailment_style = entailment_style
        decomposition_llm = claim_decomposition_llm if claim_decomposition_llm else nli_llm
        self.decomposer = ResponseDecomposer(claim_decomposition_llm=decomposition_llm)
        self.entailment_classifier = EntailmentClassifier(
            nli_llm=nli_llm,
            style=entailment_style,
        )

    async def score(
        self,
        queries: List[str],
        contexts: List[str],
        answers: List[str],
        return_raw_judge_responses: bool = False,
        show_progress_bars: bool = True,
        return_prompts: bool = False,
        **_: Any,
    ) -> UQResult:
        progress_bar = self.build_progress_bar(show_progress_bars)
        raw_responses: Optional[List[List[str]]] = None

        try:
            if progress_bar:
                progress_bar.add_task("✂️  Decomposition")

            claim_sets = await self.decomposer.decompose_claims(
                responses=answers,
                progress_bar=progress_bar,
            )

            if progress_bar:
                progress_bar.add_task("")
                progress_bar.add_task("📈 Verification")

            if return_raw_judge_responses:
                claim_scores, claim_labels, raw_responses = await self._verify_claims_against_contexts(
                    claim_sets=claim_sets,
                    contexts=contexts,
                    return_raw_judge_responses=True,
                    progress_bar=progress_bar,
                )
            else:
                claim_scores, claim_labels = await self._verify_claims_against_contexts(
                    claim_sets=claim_sets,
                    contexts=contexts,
                    return_raw_judge_responses=False,
                    progress_bar=progress_bar,
                )
        finally:
            if progress_bar:
                progress_bar.stop()

        claims_data = self.build_claims_data_from_lists(
            claim_sets=claim_sets,
            claim_labels=claim_labels,
            claim_scores=claim_scores,
            raw_judge_responses=raw_responses,
        )
        return self.build_result(
            queries=queries,
            contexts=contexts,
            answers=answers,
            claims_data=claims_data,
            metadata={
                "mode": "two_stage",
                "aggregation": self.aggregation,
                "entailment_style": self.entailment_style,
                "raw_judge_responses": raw_responses,
                "prompts": None,
            },
        )

    async def _verify_claims_against_contexts(
        self,
        claim_sets: List[List[str]],
        contexts: List[str],
        return_raw_judge_responses: bool = False,
        progress_bar: Optional[Progress] = None,
    ):
        """Verify each claim against its corresponding context via entailment."""
        flat_premises: List[str] = []
        flat_hypotheses: List[str] = []
        structure: List[Tuple[int, int]] = []

        for i, (claim_set, context) in enumerate(zip(claim_sets, contexts)):
            for j, claim in enumerate(claim_set):
                flat_premises.append(context)
                flat_hypotheses.append(claim)
                structure.append((i, j))

        if not flat_premises:
            if return_raw_judge_responses:
                return [[] for _ in claim_sets], [[] for _ in claim_sets], [[] for _ in claim_sets]
            return [[] for _ in claim_sets], [[] for _ in claim_sets]

        nli_result = await self.entailment_classifier.judge_entailment(
            premises=flat_premises,
            hypotheses=flat_hypotheses,
            return_labels=True,
            progress_bar=progress_bar,
        )

        flat_scores = nli_result["scores"]
        flat_labels = nli_result["labels"]
        claim_scores: List[List[float]] = [[] for _ in claim_sets]
        claim_labels: List[List[str]] = [[] for _ in claim_sets]

        for idx, (answer_i, _) in enumerate(structure):
            claim_scores[answer_i].append(flat_scores[idx])
            claim_labels[answer_i].append(flat_labels[idx])

        if return_raw_judge_responses:
            flat_raw_responses = nli_result["judge_responses"]
            raw_responses: List[List[str]] = [[] for _ in claim_sets]
            for idx, (answer_i, _) in enumerate(structure):
                raw_responses[answer_i].append(flat_raw_responses[idx])
            return claim_scores, claim_labels, raw_responses

        return claim_scores, claim_labels


class _SinglePromptGroundednessEngine(_BaseGroundednessEngine):
    """Single-prompt groundedness engine with structured JSON parsing."""

    def __init__(
        self,
        llm: BaseChatModel,
        aggregation: str = "mean",
        include_reasoning: bool = True,
        include_relevant_context: bool = True,
        enable_external_verification: bool = False,
    ) -> None:
        super().__init__(aggregation=aggregation)
        if llm is None:
            raise ValueError(
                "llm is required for ContextGroundednessScorer in single_prompt mode. "
                "Provide a LangChain BaseChatModel instance."
            )
        self.llm = llm
        self.include_reasoning = include_reasoning
        self.include_relevant_context = include_relevant_context
        self.enable_external_verification = enable_external_verification

    async def score(
        self,
        queries: List[str],
        contexts: List[str],
        answers: List[str],
        show_progress_bars: bool = True,
        return_prompts: bool = False,
        **_: Any,
    ) -> UQResult:
        score_outputs, prompts = await self._run_stage1(
            contexts=contexts,
            answers=answers,
            show_progress_bars=show_progress_bars,
            return_prompts=return_prompts,
        )
        claims_data = [output["claims_data"] for output in score_outputs]

        result = self.build_result(
            queries=queries,
            contexts=contexts,
            answers=answers,
            claims_data=claims_data,
            metadata={
                "mode": "single_prompt",
                "aggregation": self.aggregation,
                "include_reasoning": self.include_reasoning,
                "include_relevant_context": self.include_relevant_context,
                "enable_external_verification": self.enable_external_verification,
                "prompts": prompts,
            },
        )
        if return_prompts:
            result.data["prompts"] = prompts
        return result

    async def _run_stage1(
        self,
        contexts: List[str],
        answers: List[str],
        show_progress_bars: bool,
        return_prompts: bool,
    ) -> Tuple[List[Dict[str, Any]], Optional[List[str]]]:
        """Run the Stage 1 single-prompt call for every (context, answer) pair.

        Returned as a helper so that composite engines (e.g. the search-augmented
        one) can reuse the exact same decomposition pipeline.
        """
        progress_bar = self.build_progress_bar(show_progress_bars)
        try:
            if progress_bar:
                progress_bar.add_task("🔍 Unified scoring", total=len(answers))

            tasks = [
                self._score_single(
                    context=context,
                    answer=answer,
                    progress_bar=progress_bar,
                    return_prompt=return_prompts,
                )
                for context, answer in zip(contexts, answers)
            ]
            score_outputs = await asyncio.gather(*tasks)
        finally:
            if progress_bar:
                progress_bar.stop()

        prompts = [out["prompt"] for out in score_outputs] if return_prompts else None
        return score_outputs, prompts

    async def _score_single(
        self,
        context: str,
        answer: str,
        progress_bar: Optional[Progress] = None,
        return_prompt: bool = False,
    ) -> Dict[str, Any]:
        prompt = get_unified_groundedness_prompt(
            context=context,
            answer=answer,
            include_reasoning=self.include_reasoning,
            include_relevant_context=self.include_relevant_context,
            enable_external_verification=self.enable_external_verification,
        )
        messages = [
            SystemMessage(UNIFIED_GROUNDEDNESS_SYSTEM_PROMPT),
            HumanMessage(prompt),
        ]

        try:
            generation = await self.llm.ainvoke(messages)
            raw_text = generation.content
        except Exception as exc:
            logger.warning("LLM call failed for single_prompt groundedness scoring: %s", exc)
            raw_text = ""

        if progress_bar:
            try:
                task_ids = [task.id for task in progress_bar.tasks if not task.finished]
                if task_ids:
                    progress_bar.update(task_ids[0], advance=1)
            except Exception:
                logger.debug("Could not update progress bar for single_prompt scoring")

        return {
            "claims_data": self._parse_response(raw_text=raw_text, answer=answer),
            "prompt": prompt if return_prompt else None,
        }

    def _parse_response(self, raw_text: str, answer: str) -> List[Dict[str, Any]]:
        """Parse the LLM JSON response into normalized claim dictionaries."""
        cleaned = re.sub(r"```(?:json)?\s*", "", raw_text).strip()
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()

        json_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not json_match:
            logger.warning(
                "Could not find JSON array in LLM response. Raw text (first 200 chars): %s",
                raw_text[:200],
            )
            return []

        try:
            items = json.loads(json_match.group())
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse error in single_prompt groundedness response: %s", exc)
            return []

        if not isinstance(items, list):
            logger.warning("Expected JSON array, got %s", type(items).__name__)
            return []

        claims_data: List[Dict[str, Any]] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                logger.debug("Skipping non-dict item at index %d: %r", i, item)
                continue

            claim = str(item.get("claim", "") or "").strip()
            anchor_text = str(item.get("anchor_text", "") or "").strip()
            raw_verdict = str(item.get("verdict", "") or "").strip().lower()

            if not claim or not anchor_text:
                logger.debug("Skipping item %d: missing 'claim' or 'anchor_text'", i)
                continue

            verdict = raw_verdict if raw_verdict in VERDICT_SCORE_MAP else "baseless"
            score = VERDICT_SCORE_MAP[verdict]
            start_offset, end_offset = self._compute_offsets(anchor_text, answer)

            relevant_context = item.get("relevant_context", []) or []
            if not isinstance(relevant_context, list):
                relevant_context = [str(relevant_context)]

            # Stage-1 external-verification hints (only present when the flag is on).
            need_external_check = bool(item.get("need_external_check", False))
            raw_queries = item.get("search_queries", []) or []
            if not isinstance(raw_queries, list):
                raw_queries = [raw_queries]
            search_queries = [str(q).strip() for q in raw_queries if str(q).strip()]

            # Only trust the flag for verdicts where external check makes sense.
            if verdict not in EXTERNAL_CHECK_VERDICTS:
                need_external_check = False
                search_queries = []

            claims_data.append(
                {
                    "claim": claim,
                    "anchor_text": anchor_text,
                    "verdict": verdict,
                    "score": score,
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                    "reasoning": str(item.get("reasoning", "") or "").strip(),
                    "relevant_context": [str(x) for x in relevant_context],
                    "raw_judge_response": raw_text,
                    "context_verdict": verdict,
                    "need_external_check": need_external_check,
                    "search_queries": search_queries,
                    "world_verdict": "not_checked",
                    "world_score": np.nan,
                    "world_reasoning": "",
                    "evidence_snippets": [],
                    "evidence_urls": [],
                    "raw_world_response": None,
                }
            )

        return claims_data

    @staticmethod
    def _compute_offsets(anchor_text: str, answer: str) -> Tuple[int, int]:
        """Find character offsets of ``anchor_text`` within ``answer``."""
        start = answer.find(anchor_text)
        if start == -1:
            logger.debug("anchor_text not found in answer (first 80 chars): %r", anchor_text[:80])
            return -1, -1
        return start, start + len(anchor_text)


class _SinglePromptWithSearchEngine(_SinglePromptGroundednessEngine):
    """Single-prompt engine followed by an external factuality verification stage.

    Claims whose Stage-1 ``context_verdict`` is in
    :data:`EXTERNAL_CHECK_VERDICTS` and whose ``need_external_check`` flag is
    ``True`` are routed through the provided :class:`ExternalVerifier` to
    receive a ``world_verdict``. Everything else stays ``"not_checked"``.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        external_verifier: Any,
        aggregation: str = "mean",
        include_reasoning: bool = True,
        include_relevant_context: bool = True,
    ) -> None:
        if external_verifier is None:
            raise ValueError(
                "_SinglePromptWithSearchEngine requires a non-None ExternalVerifier instance."
            )
        super().__init__(
            llm=llm,
            aggregation=aggregation,
            include_reasoning=include_reasoning,
            include_relevant_context=include_relevant_context,
            enable_external_verification=True,
        )
        self.external_verifier = external_verifier

    async def score(
        self,
        queries: List[str],
        contexts: List[str],
        answers: List[str],
        show_progress_bars: bool = True,
        return_prompts: bool = False,
        return_world_prompts: bool = True,
        **_: Any,
    ) -> UQResult:
        # Stage 1: reuse parent pipeline to decompose + context-verify.
        score_outputs, prompts = await self._run_stage1(
            contexts=contexts,
            answers=answers,
            show_progress_bars=show_progress_bars,
            return_prompts=return_prompts,
        )
        claims_data = [out["claims_data"] for out in score_outputs]

        # Stage 2+3: collect baseless / overclaim claims that should be checked.
        from uqlm.scorers.longform.external_verifier import ClaimForExternal

        pending: List[ClaimForExternal] = []
        for i, claim_list in enumerate(claims_data):
            for j, claim in enumerate(claim_list):
                if (
                    claim.get("context_verdict") in EXTERNAL_CHECK_VERDICTS
                    and claim.get("need_external_check")
                    and claim.get("search_queries")
                ):
                    pending.append(
                        ClaimForExternal(
                            claim=claim["claim"],
                            search_queries=list(claim.get("search_queries", [])),
                            context_reasoning=claim.get("reasoning") or None,
                            key=(i, j),
                        )
                    )

        if pending:
            logger.debug(
                "_SinglePromptWithSearchEngine: dispatching %d claims to ExternalVerifier",
                len(pending),
            )
            verdicts = await self.external_verifier.verify(pending)
            verdict_by_key = {v.key: v for v in verdicts}
            for (i, j), v in verdict_by_key.items():
                claim = claims_data[i][j]
                claim["world_verdict"] = v.world_verdict
                claim["world_score"] = VERDICT_SCORE_MAP.get(v.world_verdict, np.nan)
                claim["world_reasoning"] = v.reasoning
                claim["evidence_snippets"] = list(v.evidence_snippets)
                claim["evidence_urls"] = list(v.evidence_urls)
                claim["raw_world_response"] = v.raw_response or None
                # Stage 3 prompt — stored per-claim for full pipeline traceability.
                if return_world_prompts:
                    claim["world_prompt"] = v.world_prompt or ""

        result = self.build_result(
            queries=queries,
            contexts=contexts,
            answers=answers,
            claims_data=claims_data,
            metadata={
                "mode": "single_prompt_with_search",
                "aggregation": self.aggregation,
                "include_reasoning": self.include_reasoning,
                "include_relevant_context": self.include_relevant_context,
                "enable_external_verification": True,
                "num_external_checks": len(pending),
                "prompts": prompts,
            },
        )
        if return_prompts:
            result.data["prompts"] = prompts
        if return_world_prompts:
            # Collect world_prompts as a parallel list: per-answer, per-claim.
            # Claims that were not externally checked get an empty string.
            result.data["world_prompts"] = [
                [c.get("world_prompt", "") for c in claim_list]
                for claim_list in claims_data
            ]
        return result


_VALID_MODES = ("two_stage", "single_prompt", "single_prompt_with_search")


class ContextGroundednessScorer:
    """
    Public groundedness scorer with mode-based strategy selection.

    Parameters
    ----------
    mode : str, default="single_prompt"
        Scoring strategy. One of ``"two_stage"``, ``"single_prompt"``, or
        ``"single_prompt_with_search"``.
    nli_llm : BaseChatModel, optional
        LLM for entailment verification in ``two_stage`` mode.
    claim_decomposition_llm : BaseChatModel, optional
        LLM for claim decomposition in ``two_stage`` mode.
    entailment_style : str, default="nli_classification"
        Entailment prompt style for ``two_stage`` mode.
    llm : BaseChatModel, optional
        LLM for ``single_prompt`` / ``single_prompt_with_search`` Stage 1.
    aggregation : str, default="mean"
        Response-level aggregation method.
    include_reasoning : bool, default=True
        Whether single-prompt modes should request reasoning.
    include_relevant_context : bool, default=True
        Whether single-prompt modes should request relevant context excerpts.
    external_verifier : ExternalVerifier, optional
        Required for ``single_prompt_with_search`` mode. Performs Stage 3
        world-knowledge verification for baseless / overclaim claims.
    """

    def __init__(
        self,
        mode: str = "single_prompt",
        nli_llm: Optional[BaseChatModel] = None,
        claim_decomposition_llm: Optional[BaseChatModel] = None,
        entailment_style: str = "nli_classification",
        llm: Optional[BaseChatModel] = None,
        aggregation: str = "mean",
        include_reasoning: bool = True,
        include_relevant_context: bool = True,
        external_verifier: Optional[Any] = None,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid mode: {mode!r}. Must be one of {_VALID_MODES}."
            )

        self.mode = mode
        self.aggregation = aggregation

        if mode == "two_stage":
            self._engine = _TwoStageGroundednessEngine(
                nli_llm=nli_llm,
                claim_decomposition_llm=claim_decomposition_llm,
                entailment_style=entailment_style,
                aggregation=aggregation,
            )
        elif mode == "single_prompt":
            self._engine = _SinglePromptGroundednessEngine(
                llm=llm,
                aggregation=aggregation,
                include_reasoning=include_reasoning,
                include_relevant_context=include_relevant_context,
            )
        else:  # single_prompt_with_search
            if external_verifier is None:
                raise ValueError(
                    "mode='single_prompt_with_search' requires an external_verifier argument."
                )
            self._engine = _SinglePromptWithSearchEngine(
                llm=llm,
                external_verifier=external_verifier,
                aggregation=aggregation,
                include_reasoning=include_reasoning,
                include_relevant_context=include_relevant_context,
            )

        logger.debug(
            "ContextGroundednessScorer initialized with mode=%s, aggregation=%s",
            mode,
            aggregation,
        )

    @staticmethod
    def _build_progress_bar(show_progress_bars: bool) -> Optional[Progress]:
        """Backward-compatible wrapper for internal progress bar creation."""
        return _BaseGroundednessEngine.build_progress_bar(show_progress_bars)

    async def score(
        self,
        queries: List[str],
        contexts: List[str],
        answers: List[str],
        return_raw_judge_responses: bool = False,
        show_progress_bars: bool = True,
        return_prompts: bool = False,
        return_world_prompts: bool = True,
    ) -> UQResult:
        """Score pre-generated answers against contexts and return ``UQResult``.

        Parameters
        ----------
        return_world_prompts : bool, default=True
            When ``True`` and ``mode="single_prompt_with_search"``, the full
            Stage 3 (world-verifier) prompt is stored per-claim in
            ``claims_data[i][j]["world_prompt"]`` and collected into
            ``result.data["world_prompts"]`` (list-of-lists, parallel to
            ``claims_data``). Claims that were not externally checked get an
            empty string. Enabled by default so the full pipeline is always
            traceable without extra flags.
        """
        if not (len(queries) == len(contexts) == len(answers)):
            raise ValueError(
                f"Input lists must have equal length. "
                f"Got queries={len(queries)}, contexts={len(contexts)}, answers={len(answers)}."
            )

        return await self._engine.score(
            queries=queries,
            contexts=contexts,
            answers=answers,
            return_raw_judge_responses=return_raw_judge_responses,
            show_progress_bars=show_progress_bars,
            return_prompts=return_prompts,
            return_world_prompts=return_world_prompts,
        )
