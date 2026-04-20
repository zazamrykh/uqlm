"""
Unit tests for uqlm.scorers.longform.context_groundedness.

Tests cover:
- VERDICT_SCORE_MAP constants
- _BaseGroundednessEngine: aggregate_scores, build_claims_data_from_lists,
  build_parallel_lists_from_claims_data, build_result
- _SinglePromptGroundednessEngine: _parse_response, _compute_offsets,
  and end-to-end score() with a mocked LLM
- _TwoStageGroundednessEngine: construction validation
- ContextGroundednessScorer: mode validation, input length validation,
  delegation to the selected engine
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from uqlm.scorers.longform.context_groundedness import (
    VERDICT_SCORE_MAP,
    ContextGroundednessScorer,
    _BaseGroundednessEngine,
    _SinglePromptGroundednessEngine,
    _TwoStageGroundednessEngine,
)
from uqlm.utils.results import UQResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Concrete base engine for testing abstract helpers
# ---------------------------------------------------------------------------


class _ConcreteBase(_BaseGroundednessEngine):
    """Concrete subclass to instantiate _BaseGroundednessEngine in tests."""

    pass


# ---------------------------------------------------------------------------
# VERDICT_SCORE_MAP
# ---------------------------------------------------------------------------


class TestVerdictScoreMap:
    def test_supported_maps_to_one(self):
        logger.debug("Testing 'supported' verdict maps to 1.0")
        assert VERDICT_SCORE_MAP["supported"] == 1.0

    def test_baseless_maps_to_half(self):
        logger.debug("Testing 'baseless' verdict maps to 0.5")
        assert VERDICT_SCORE_MAP["baseless"] == 0.5

    def test_contradicted_maps_to_zero(self):
        logger.debug("Testing 'contradicted' verdict maps to 0.0")
        assert VERDICT_SCORE_MAP["contradicted"] == 0.0

    def test_contradiction_alias_maps_to_zero(self):
        """Alias 'contradiction' should behave like 'contradicted'."""
        logger.debug("Testing 'contradiction' alias maps to 0.0")
        assert VERDICT_SCORE_MAP["contradiction"] == 0.0

    def test_unknown_maps_to_nan(self):
        logger.debug("Testing 'unknown' verdict maps to NaN")
        assert np.isnan(VERDICT_SCORE_MAP["unknown"])


# ---------------------------------------------------------------------------
# _BaseGroundednessEngine
# ---------------------------------------------------------------------------


class TestBaseEngineAggregate:
    def test_invalid_aggregation_raises(self):
        logger.debug("Testing invalid aggregation raises ValueError")
        with pytest.raises(ValueError, match="Invalid aggregation"):
            _ConcreteBase(aggregation="max")

    def test_aggregate_mean(self):
        logger.debug("Testing aggregate_scores with mean aggregation")
        eng = _ConcreteBase(aggregation="mean")
        assert eng.aggregate_scores([[0.8, 0.6], [0.9]]) == [
            pytest.approx(0.7),
            pytest.approx(0.9),
        ]

    def test_aggregate_min(self):
        logger.debug("Testing aggregate_scores with min aggregation")
        eng = _ConcreteBase(aggregation="min")
        assert eng.aggregate_scores([[0.8, 0.6], [0.9]]) == [
            pytest.approx(0.6),
            pytest.approx(0.9),
        ]

    def test_aggregate_empty_scores_returns_nan(self):
        logger.debug("Testing aggregate_scores returns NaN for empty claim list")
        eng = _ConcreteBase(aggregation="mean")
        out = eng.aggregate_scores([[]])
        assert len(out) == 1
        assert np.isnan(out[0])

    def test_aggregate_all_nan_scores_returns_nan(self):
        logger.debug("Testing aggregate_scores returns NaN when all scores are NaN")
        eng = _ConcreteBase(aggregation="mean")
        out = eng.aggregate_scores([[float("nan"), float("nan")]])
        assert np.isnan(out[0])

    def test_aggregate_filters_nan_values(self):
        logger.debug("Testing aggregate_scores skips NaN values")
        eng = _ConcreteBase(aggregation="mean")
        out = eng.aggregate_scores([[0.5, float("nan"), 1.0]])
        assert out[0] == pytest.approx(0.75)


class TestBaseEngineBuildClaimsData:
    def test_build_claims_data_from_lists(self):
        logger.debug("Testing build_claims_data_from_lists creates normalized records")
        eng = _ConcreteBase(aggregation="mean")
        claims_data = eng.build_claims_data_from_lists(
            claim_sets=[["c1", "c2"]],
            claim_labels=[["supported", "baseless"]],
            claim_scores=[[1.0, 0.5]],
            raw_judge_responses=[["raw1", "raw2"]],
        )
        assert len(claims_data) == 1
        assert len(claims_data[0]) == 2
        assert claims_data[0][0]["claim"] == "c1"
        assert claims_data[0][0]["verdict"] == "supported"
        assert claims_data[0][0]["score"] == 1.0
        assert claims_data[0][0]["raw_judge_response"] == "raw1"
        assert claims_data[0][1]["verdict"] == "baseless"

    def test_build_claims_data_handles_missing_fields(self):
        logger.debug("Testing build_claims_data_from_lists fills missing fields with defaults")
        eng = _ConcreteBase(aggregation="mean")
        claims_data = eng.build_claims_data_from_lists(
            claim_sets=[["c1", "c2"]],
            claim_labels=[["supported"]],  # only 1 label for 2 claims
            claim_scores=[[1.0]],
        )
        assert claims_data[0][1]["verdict"] == "unknown"
        assert np.isnan(claims_data[0][1]["score"])

    def test_build_parallel_lists_roundtrip(self):
        logger.debug("Testing round-trip between claims_data and parallel lists")
        eng = _ConcreteBase(aggregation="mean")
        original = eng.build_claims_data_from_lists(
            claim_sets=[["c1", "c2"], ["c3"]],
            claim_labels=[["supported", "baseless"], ["contradicted"]],
            claim_scores=[[1.0, 0.5], [0.0]],
        )
        cs, cl, sc = eng.build_parallel_lists_from_claims_data(original)
        assert cs == [["c1", "c2"], ["c3"]]
        assert cl == [["supported", "baseless"], ["contradicted"]]
        assert sc == [[1.0, 0.5], [0.0]]


class TestBaseEngineBuildResult:
    def test_build_result_returns_uqresult(self):
        logger.debug("Testing build_result returns a UQResult instance")
        eng = _ConcreteBase(aggregation="mean")
        claims_data = [
            [
                {
                    "claim": "c1",
                    "anchor_text": "c1",
                    "verdict": "supported",
                    "score": 1.0,
                    "start_offset": 0,
                    "end_offset": 2,
                    "reasoning": "",
                    "relevant_context": [],
                    "raw_judge_response": None,
                }
            ]
        ]
        result = eng.build_result(
            queries=["q"],
            contexts=["ctx"],
            answers=["a"],
            claims_data=claims_data,
            metadata={"mode": "test"},
        )
        assert isinstance(result, UQResult)
        assert result.data["queries"] == ["q"]
        assert result.data["answers"] == ["a"]
        assert result.data["claim_labels"] == [["supported"]]
        assert result.data["response_scores"] == [pytest.approx(1.0)]
        assert result.metadata["mode"] == "test"


class TestBuildProgressBar:
    def test_disabled_returns_none(self):
        logger.debug("Testing build_progress_bar returns None when disabled")
        assert _BaseGroundednessEngine.build_progress_bar(False) is None

    def test_static_wrapper_delegates(self):
        logger.debug("Testing ContextGroundednessScorer._build_progress_bar wrapper")
        assert ContextGroundednessScorer._build_progress_bar(False) is None


# ---------------------------------------------------------------------------
# _SinglePromptGroundednessEngine
# ---------------------------------------------------------------------------


class TestSinglePromptParseResponse:
    @pytest.fixture
    def engine(self):
        mock_llm = MagicMock(spec=BaseChatModel)
        return _SinglePromptGroundednessEngine(llm=mock_llm)

    def test_requires_llm(self):
        logger.debug("Testing _SinglePromptGroundednessEngine requires an LLM")
        with pytest.raises(ValueError, match="llm is required"):
            _SinglePromptGroundednessEngine(llm=None)

    def test_parses_valid_json_response(self, engine):
        logger.debug("Testing _parse_response parses a valid JSON array")
        raw = """[
            {"claim": "Paris is capital", "anchor_text": "Paris", "verdict": "supported"},
            {"claim": "Founded in 52 BC", "anchor_text": "52 BC", "verdict": "baseless"}
        ]"""
        answer = "Paris is capital. Founded in 52 BC."
        result = engine._parse_response(raw, answer)
        assert len(result) == 2
        assert result[0]["verdict"] == "supported"
        assert result[0]["score"] == 1.0
        assert result[1]["verdict"] == "baseless"
        assert result[1]["score"] == 0.5

    def test_parses_json_with_markdown_fences(self, engine):
        logger.debug("Testing _parse_response strips markdown fences")
        raw = """```json
[{"claim": "Foo", "anchor_text": "Foo", "verdict": "supported"}]
```"""
        result = engine._parse_response(raw, "Foo")
        assert len(result) == 1
        assert result[0]["claim"] == "Foo"

    def test_offsets_computed(self, engine):
        logger.debug("Testing _parse_response computes char offsets from anchor_text")
        raw = '[{"claim": "X", "anchor_text": "Paris", "verdict": "supported"}]'
        answer = "The capital of France is Paris."
        result = engine._parse_response(raw, answer)
        expected_start = answer.find("Paris")
        assert result[0]["start_offset"] == expected_start
        assert result[0]["end_offset"] == expected_start + len("Paris")

    def test_anchor_not_found_gives_negative_offsets(self, engine):
        logger.debug("Testing _parse_response returns -1 offsets when anchor not found")
        raw = '[{"claim": "X", "anchor_text": "NOT_IN_ANSWER", "verdict": "supported"}]'
        result = engine._parse_response(raw, "some other answer")
        assert result[0]["start_offset"] == -1
        assert result[0]["end_offset"] == -1

    def test_no_json_array_returns_empty(self, engine):
        logger.debug("Testing _parse_response returns empty list when no JSON array present")
        assert engine._parse_response("I cannot comply.", "answer") == []

    def test_malformed_json_returns_empty(self, engine):
        logger.debug("Testing _parse_response returns empty list on malformed JSON")
        assert engine._parse_response('[{"claim": "X"', "answer") == []

    def test_non_array_json_returns_empty(self, engine):
        logger.debug("Testing _parse_response rejects non-array JSON")
        # Regex still finds [...] inside — ensure a literal non-array doesn't parse as array
        assert engine._parse_response("just text", "answer") == []

    def test_skips_item_without_claim_or_anchor(self, engine):
        logger.debug("Testing _parse_response skips items missing claim or anchor_text")
        raw = """[
            {"claim": "", "anchor_text": "x", "verdict": "supported"},
            {"claim": "valid", "anchor_text": "valid", "verdict": "supported"}
        ]"""
        result = engine._parse_response(raw, "valid")
        assert len(result) == 1
        assert result[0]["claim"] == "valid"

    def test_unknown_verdict_defaults_to_baseless(self, engine):
        logger.debug("Testing _parse_response falls back to 'baseless' on unknown verdict")
        raw = '[{"claim": "X", "anchor_text": "X", "verdict": "weird-label"}]'
        result = engine._parse_response(raw, "X")
        assert result[0]["verdict"] == "baseless"
        assert result[0]["score"] == 0.5

    def test_relevant_context_normalized_to_list(self, engine):
        logger.debug("Testing _parse_response normalizes relevant_context to list")
        raw = (
            '[{"claim": "X", "anchor_text": "X", "verdict": "supported", '
            '"relevant_context": "not a list"}]'
        )
        result = engine._parse_response(raw, "X")
        assert isinstance(result[0]["relevant_context"], list)
        assert result[0]["relevant_context"] == ["not a list"]


class TestSinglePromptComputeOffsets:
    def test_anchor_found(self):
        logger.debug("Testing _compute_offsets when anchor is found")
        s, e = _SinglePromptGroundednessEngine._compute_offsets("Paris", "The capital is Paris.")
        assert s == 15
        assert e == 20

    def test_anchor_not_found(self):
        logger.debug("Testing _compute_offsets when anchor is not found")
        assert _SinglePromptGroundednessEngine._compute_offsets("X", "Y") == (-1, -1)


class TestSinglePromptScore:
    @pytest.mark.asyncio
    async def test_score_end_to_end_mocked_llm(self):
        """End-to-end score() call with a mocked LLM returning valid JSON."""
        logger.debug("Testing single-prompt score() with mocked LLM")
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_gen = MagicMock()
        mock_gen.content = (
            '[{"claim": "Paris is capital", "anchor_text": "Paris", '
            '"verdict": "supported"}]'
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_gen)

        engine = _SinglePromptGroundednessEngine(llm=mock_llm)
        result = await engine.score(
            queries=["Q"],
            contexts=["Paris is the capital of France."],
            answers=["Paris is capital"],
            show_progress_bars=False,
        )
        assert isinstance(result, UQResult)
        assert result.data["claim_labels"] == [["supported"]]
        assert result.data["response_scores"] == [pytest.approx(1.0)]
        mock_llm.ainvoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_score_llm_failure_yields_empty_claims(self):
        logger.debug("Testing single-prompt score() handles LLM failure gracefully")
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
        engine = _SinglePromptGroundednessEngine(llm=mock_llm)
        result = await engine.score(
            queries=["Q"],
            contexts=["C"],
            answers=["A"],
            show_progress_bars=False,
        )
        assert result.data["claim_sets"] == [[]]
        # aggregate of empty list → NaN
        assert np.isnan(result.data["response_scores"][0])


# ---------------------------------------------------------------------------
# _TwoStageGroundednessEngine (just construction + validation)
# ---------------------------------------------------------------------------


class TestTwoStageEngine:
    def test_requires_nli_llm(self):
        logger.debug("Testing _TwoStageGroundednessEngine requires nli_llm")
        with pytest.raises(ValueError, match="nli_llm is required"):
            _TwoStageGroundednessEngine(nli_llm=None)

    def test_construct_with_nli_llm(self):
        logger.debug("Testing _TwoStageGroundednessEngine constructs with nli_llm")
        mock_llm = MagicMock(spec=BaseChatModel)
        with (
            patch("uqlm.scorers.longform.context_groundedness.ResponseDecomposer"),
            patch("uqlm.scorers.longform.context_groundedness.EntailmentClassifier"),
        ):
            eng = _TwoStageGroundednessEngine(nli_llm=mock_llm)
            assert eng.nli_llm is mock_llm
            assert eng.aggregation == "mean"


# ---------------------------------------------------------------------------
# ContextGroundednessScorer (public API)
# ---------------------------------------------------------------------------


class TestContextGroundednessScorer:
    def test_invalid_mode_raises(self):
        logger.debug("Testing ContextGroundednessScorer rejects invalid mode")
        with pytest.raises(ValueError, match="Invalid mode"):
            ContextGroundednessScorer(mode="invalid", llm=MagicMock(spec=BaseChatModel))

    def test_single_prompt_mode_instantiates(self):
        logger.debug("Testing ContextGroundednessScorer single_prompt mode")
        scorer = ContextGroundednessScorer(
            mode="single_prompt",
            llm=MagicMock(spec=BaseChatModel),
        )
        assert scorer.mode == "single_prompt"
        assert isinstance(scorer._engine, _SinglePromptGroundednessEngine)

    def test_two_stage_mode_instantiates(self):
        logger.debug("Testing ContextGroundednessScorer two_stage mode")
        with (
            patch("uqlm.scorers.longform.context_groundedness.ResponseDecomposer"),
            patch("uqlm.scorers.longform.context_groundedness.EntailmentClassifier"),
        ):
            scorer = ContextGroundednessScorer(
                mode="two_stage",
                nli_llm=MagicMock(spec=BaseChatModel),
            )
            assert scorer.mode == "two_stage"
            assert isinstance(scorer._engine, _TwoStageGroundednessEngine)

    @pytest.mark.asyncio
    async def test_score_length_mismatch_raises(self):
        logger.debug("Testing score() raises on input length mismatch")
        scorer = ContextGroundednessScorer(
            mode="single_prompt",
            llm=MagicMock(spec=BaseChatModel),
        )
        with pytest.raises(ValueError, match="Input lists must have equal length"):
            await scorer.score(
                queries=["q1"],
                contexts=["c1", "c2"],
                answers=["a1"],
            )

    @pytest.mark.asyncio
    async def test_score_delegates_to_engine(self):
        logger.debug("Testing score() delegates to underlying engine")
        scorer = ContextGroundednessScorer(
            mode="single_prompt",
            llm=MagicMock(spec=BaseChatModel),
        )
        fake_result = MagicMock(spec=UQResult)
        scorer._engine.score = AsyncMock(return_value=fake_result)
        out = await scorer.score(queries=["q"], contexts=["c"], answers=["a"])
        assert out is fake_result
        scorer._engine.score.assert_awaited_once()
