"""
Unit tests for uqlm.longform.benchmark.ragtruth_grader.

Tests cover:
- _spans_overlap helper
- RAGTruthGrader._get_claim_field (dict and object access)
- RAGTruthGrader._compute_response_level
- RAGTruthGrader._evaluate_single
- RAGTruthGrader.evaluate (full pipeline, edge cases)
"""

import logging
import math
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from uqlm.longform.benchmark.ragtruth_grader import (
    RAGTRUTH_LABEL_TO_CATEGORY,
    RAGTruthGrader,
    _spans_overlap,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_gt_span(start: int, end: int, label_type: str, text: str = "") -> Dict[str, Any]:
    return {"start": start, "end": end, "label_type": label_type, "text": text or f"[{start}:{end}]"}


def _make_verdict_dict(
    verdict: str,
    start_offset: int,
    end_offset: int,
    anchor_text: str = "",
    claim: str = "",
) -> Dict[str, Any]:
    return {
        "verdict": verdict,
        "start_offset": start_offset,
        "end_offset": end_offset,
        "anchor_text": anchor_text,
        "claim": claim,
    }


def _make_verdict_obj(
    verdict: str,
    start_offset: int,
    end_offset: int,
    anchor_text: str = "",
    claim: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        verdict=verdict,
        start_offset=start_offset,
        end_offset=end_offset,
        anchor_text=anchor_text,
        claim=claim,
    )


# ---------------------------------------------------------------------------
# _spans_overlap
# ---------------------------------------------------------------------------


class TestSpansOverlap:
    def test_no_overlap_left(self):
        logger.debug("Testing non-overlapping spans where first is to the left")
        assert _spans_overlap(0, 5, 5, 10) is False

    def test_no_overlap_right(self):
        logger.debug("Testing non-overlapping spans where first is to the right")
        assert _spans_overlap(10, 20, 0, 10) is False

    def test_partial_overlap_left(self):
        logger.debug("Testing partial overlap on the left side")
        assert _spans_overlap(0, 6, 5, 10) is True

    def test_partial_overlap_right(self):
        logger.debug("Testing partial overlap on the right side")
        assert _spans_overlap(5, 15, 0, 10) is True

    def test_full_containment(self):
        logger.debug("Testing full containment (inner span inside outer)")
        assert _spans_overlap(2, 8, 0, 10) is True

    def test_identical_spans(self):
        logger.debug("Testing identical spans")
        assert _spans_overlap(3, 7, 3, 7) is True

    def test_single_char_overlap(self):
        logger.debug("Testing single-character overlap")
        assert _spans_overlap(0, 4, 3, 7) is True

    def test_adjacent_no_overlap(self):
        """Half-open intervals [0,5) and [5,10) must NOT overlap."""
        logger.debug("Testing adjacent half-open intervals — should not overlap")
        assert _spans_overlap(0, 5, 5, 10) is False


# ---------------------------------------------------------------------------
# RAGTRUTH_LABEL_TO_CATEGORY constant
# ---------------------------------------------------------------------------


class TestLabelToCategory:
    def test_all_known_labels_present(self):
        logger.debug("Verifying all expected RAGTruth label types are mapped")
        expected = {
            "Evident Conflict",
            "Subtle Conflict",
            "Evident Baseless Info",
            "Subtle Baseless Info",
        }
        assert set(RAGTRUTH_LABEL_TO_CATEGORY.keys()) == expected

    def test_conflict_labels_map_to_conflict(self):
        assert RAGTRUTH_LABEL_TO_CATEGORY["Evident Conflict"] == "conflict"
        assert RAGTRUTH_LABEL_TO_CATEGORY["Subtle Conflict"] == "conflict"

    def test_baseless_labels_map_to_baseless(self):
        assert RAGTRUTH_LABEL_TO_CATEGORY["Evident Baseless Info"] == "baseless"
        assert RAGTRUTH_LABEL_TO_CATEGORY["Subtle Baseless Info"] == "baseless"


# ---------------------------------------------------------------------------
# RAGTruthGrader._get_claim_field
# ---------------------------------------------------------------------------


class TestGetClaimField:
    def test_dict_access_existing_key(self):
        logger.debug("Testing dict access for existing key")
        d = {"verdict": "supported", "start_offset": 0}
        assert RAGTruthGrader._get_claim_field(d, "verdict") == "supported"

    def test_dict_access_missing_key_returns_default(self):
        logger.debug("Testing dict access for missing key returns default")
        d = {"verdict": "supported"}
        assert RAGTruthGrader._get_claim_field(d, "start_offset", -1) == -1

    def test_object_access_existing_attr(self):
        logger.debug("Testing object attribute access for existing attribute")
        obj = SimpleNamespace(verdict="baseless", start_offset=10)
        assert RAGTruthGrader._get_claim_field(obj, "verdict") == "baseless"

    def test_object_access_missing_attr_returns_default(self):
        logger.debug("Testing object attribute access for missing attribute returns default")
        obj = SimpleNamespace(verdict="baseless")
        assert RAGTruthGrader._get_claim_field(obj, "start_offset", -99) == -99

    def test_none_default(self):
        logger.debug("Testing that None is returned as default when key is missing")
        d = {}
        assert RAGTruthGrader._get_claim_field(d, "nonexistent") is None


# ---------------------------------------------------------------------------
# RAGTruthGrader._compute_response_level
# ---------------------------------------------------------------------------


class TestComputeResponseLevel:
    def test_no_response_scores_returns_nan_auc(self):
        logger.debug("Testing response level with no response scores → NaN AUC")
        gt = [[_make_gt_span(0, 5, "Evident Conflict")], []]
        result = RAGTruthGrader._compute_response_level(gt, None)
        assert math.isnan(result["auc_roc"])
        assert result["n_responses"] == 2
        assert result["n_hallucinated"] == 1
        assert result["n_clean"] == 1

    def test_all_clean_returns_nan_auc(self):
        logger.debug("Testing response level with all clean responses → NaN AUC")
        gt = [[], []]
        result = RAGTruthGrader._compute_response_level(gt, [0.9, 0.8])
        assert math.isnan(result["auc_roc"])
        assert result["n_hallucinated"] == 0

    def test_all_hallucinated_returns_nan_auc(self):
        logger.debug("Testing response level with all hallucinated responses → NaN AUC")
        gt = [
            [_make_gt_span(0, 5, "Evident Conflict")],
            [_make_gt_span(0, 3, "Subtle Baseless Info")],
        ]
        result = RAGTruthGrader._compute_response_level(gt, [0.2, 0.3])
        assert math.isnan(result["auc_roc"])
        assert result["n_clean"] == 0

    def test_mixed_responses_computes_auc(self):
        logger.debug("Testing response level with mixed responses computes AUC-ROC")
        gt = [
            [_make_gt_span(0, 5, "Evident Conflict")],  # hallucinated
            [],  # clean
        ]
        # Lower score = more hallucinated → pred_proba = [1-0.2, 1-0.9] = [0.8, 0.1]
        # GT binary = [1, 0] → perfect separation → AUC = 1.0
        result = RAGTruthGrader._compute_response_level(gt, [0.2, 0.9])
        assert not math.isnan(result["auc_roc"])
        assert result["auc_roc"] == pytest.approx(1.0)

    def test_unknown_label_type_not_counted_as_hallucinated(self):
        logger.debug("Testing that unknown label types are not counted as hallucinated")
        gt = [[{"start": 0, "end": 5, "label_type": "Unknown Type", "text": "foo"}]]
        result = RAGTruthGrader._compute_response_level(gt, [0.5])
        assert result["n_hallucinated"] == 0


# ---------------------------------------------------------------------------
# RAGTruthGrader._evaluate_single
# ---------------------------------------------------------------------------


class TestEvaluateSingle:
    @pytest.fixture
    def grader(self):
        return RAGTruthGrader()

    def test_empty_gt_and_verdicts(self, grader):
        logger.debug("Testing _evaluate_single with empty GT and verdicts")
        detail = grader._evaluate_single(gt_spans=[], verdicts=[], answer_idx=0)
        assert detail["n_gt_spans"] == 0
        assert detail["n_pred_hallucinations"] == 0
        assert detail["n_gt_matched"] == 0
        assert detail["n_pred_matched"] == 0
        assert detail["confusion_entries"] == []

    def test_no_predicted_hallucinations(self, grader):
        logger.debug("Testing _evaluate_single with only supported verdicts")
        gt = [_make_gt_span(0, 10, "Evident Conflict")]
        verdicts = [_make_verdict_dict("supported", 0, 10)]
        detail = grader._evaluate_single(gt_spans=gt, verdicts=verdicts, answer_idx=0)
        assert detail["n_pred_hallucinations"] == 0
        assert detail["n_gt_matched"] == 0

    def test_perfect_match_dict_verdicts(self, grader):
        logger.debug("Testing _evaluate_single with perfect match using dict verdicts")
        gt = [_make_gt_span(0, 10, "Evident Conflict", "some text")]
        verdicts = [_make_verdict_dict("contradicted", 0, 10, "some text")]
        detail = grader._evaluate_single(gt_spans=gt, verdicts=verdicts, answer_idx=0)
        assert detail["n_gt_matched"] == 1
        assert detail["n_pred_matched"] == 1
        assert len(detail["confusion_entries"]) == 1
        entry = detail["confusion_entries"][0]
        assert entry["gt_category"] == "conflict"
        assert entry["predicted_verdict"] == "contradicted"

    def test_perfect_match_object_verdicts(self, grader):
        logger.debug("Testing _evaluate_single with perfect match using object verdicts")
        gt = [_make_gt_span(5, 15, "Subtle Baseless Info", "baseless text")]
        verdicts = [_make_verdict_obj("baseless", 5, 15, "baseless text")]
        detail = grader._evaluate_single(gt_spans=gt, verdicts=verdicts, answer_idx=0)
        assert detail["n_gt_matched"] == 1
        assert detail["n_pred_matched"] == 1
        entry = detail["confusion_entries"][0]
        assert entry["gt_category"] == "baseless"
        assert entry["predicted_verdict"] == "baseless"

    def test_no_overlap_no_match(self, grader):
        logger.debug("Testing _evaluate_single with non-overlapping spans → no match")
        gt = [_make_gt_span(0, 5, "Evident Conflict")]
        verdicts = [_make_verdict_dict("contradicted", 10, 20)]
        detail = grader._evaluate_single(gt_spans=gt, verdicts=verdicts, answer_idx=0)
        assert detail["n_gt_matched"] == 0
        assert detail["n_pred_matched"] == 0

    def test_invalid_gt_span_filtered_out(self, grader):
        logger.debug("Testing _evaluate_single filters out GT spans with unknown label_type")
        gt = [{"start": 0, "end": 10, "label_type": "Unknown", "text": "x"}]
        verdicts = [_make_verdict_dict("contradicted", 0, 10)]
        detail = grader._evaluate_single(gt_spans=gt, verdicts=verdicts, answer_idx=0)
        assert detail["n_gt_spans"] == 0

    def test_verdict_with_negative_offset_excluded(self, grader):
        logger.debug("Testing _evaluate_single excludes verdicts with negative start_offset")
        gt = [_make_gt_span(0, 10, "Evident Conflict")]
        verdicts = [_make_verdict_dict("contradicted", -1, 10)]
        detail = grader._evaluate_single(gt_spans=gt, verdicts=verdicts, answer_idx=0)
        assert detail["n_pred_hallucinations"] == 0

    def test_multiple_gt_spans_partial_match(self, grader):
        logger.debug("Testing _evaluate_single with multiple GT spans, only one matched")
        gt = [
            _make_gt_span(0, 10, "Evident Conflict"),
            _make_gt_span(50, 60, "Subtle Baseless Info"),
        ]
        verdicts = [_make_verdict_dict("contradicted", 0, 10)]
        detail = grader._evaluate_single(gt_spans=gt, verdicts=verdicts, answer_idx=0)
        assert detail["n_gt_matched"] == 1
        assert detail["n_pred_matched"] == 1

    def test_one_pred_matches_multiple_gt_spans(self, grader):
        logger.debug("Testing _evaluate_single where one prediction overlaps multiple GT spans")
        gt = [
            _make_gt_span(0, 10, "Evident Conflict"),
            _make_gt_span(5, 15, "Subtle Conflict"),
        ]
        verdicts = [_make_verdict_dict("contradicted", 0, 15)]
        detail = grader._evaluate_single(gt_spans=gt, verdicts=verdicts, answer_idx=0)
        assert detail["n_gt_matched"] == 2
        assert detail["n_pred_matched"] == 1


# ---------------------------------------------------------------------------
# RAGTruthGrader.evaluate (full pipeline)
# ---------------------------------------------------------------------------


class TestRAGTruthGraderEvaluate:
    @pytest.fixture
    def grader(self):
        return RAGTruthGrader()

    def test_length_mismatch_raises(self, grader):
        logger.debug("Testing evaluate raises ValueError on length mismatch")
        with pytest.raises(ValueError, match="Length mismatch"):
            grader.evaluate(
                gt_hallucination_labels=[[_make_gt_span(0, 5, "Evident Conflict")]],
                claim_verdicts=[[], []],
            )

    def test_all_clean_no_predictions(self, grader):
        logger.debug("Testing evaluate with all clean responses and no predictions")
        result = grader.evaluate(
            gt_hallucination_labels=[[], []],
            claim_verdicts=[[], []],
        )
        cl = result["claim_level"]
        # No GT spans → recall defaults to 1.0; no predictions → precision defaults to 1.0
        assert cl["recall"] == pytest.approx(1.0)
        assert cl["precision"] == pytest.approx(1.0)
        assert cl["n_gt_spans"] == 0

    def test_perfect_detection(self, grader):
        logger.debug("Testing evaluate with perfect hallucination detection")
        gt = [[_make_gt_span(0, 10, "Evident Conflict")]]
        verdicts = [[_make_verdict_dict("contradicted", 0, 10)]]
        result = grader.evaluate(gt, verdicts)
        cl = result["claim_level"]
        assert cl["recall"] == pytest.approx(1.0)
        assert cl["precision"] == pytest.approx(1.0)
        assert cl["f1"] == pytest.approx(1.0)

    def test_zero_recall_when_no_predictions(self, grader):
        logger.debug("Testing evaluate gives zero recall when no predictions made")
        gt = [[_make_gt_span(0, 10, "Evident Conflict")]]
        verdicts = [[]]
        result = grader.evaluate(gt, verdicts)
        cl = result["claim_level"]
        assert cl["recall"] == pytest.approx(0.0)
        assert cl["precision"] == pytest.approx(1.0)  # no predictions → precision defaults to 1.0
        assert cl["f1"] == pytest.approx(0.0)

    def test_zero_precision_when_all_false_positives(self, grader):
        logger.debug("Testing evaluate gives zero precision when all predictions are false positives")
        gt = [[]]  # no GT hallucinations
        verdicts = [[_make_verdict_dict("contradicted", 0, 10)]]
        result = grader.evaluate(gt, verdicts)
        cl = result["claim_level"]
        assert cl["precision"] == pytest.approx(0.0)
        assert cl["recall"] == pytest.approx(1.0)  # no GT spans → recall defaults to 1.0

    def test_confusion_matrix_populated(self, grader):
        logger.debug("Testing evaluate populates confusion matrix correctly")
        gt = [[_make_gt_span(0, 10, "Evident Conflict")]]
        verdicts = [[_make_verdict_dict("baseless", 0, 10)]]
        result = grader.evaluate(gt, verdicts)
        cm = result["confusion_matrix"]
        assert "conflict" in cm
        assert cm["conflict"].get("baseless", 0) == 1

    def test_response_level_included_when_scores_provided(self, grader):
        logger.debug("Testing evaluate includes response-level AUC when scores provided")
        gt = [
            [_make_gt_span(0, 10, "Evident Conflict")],
            [],
        ]
        verdicts = [
            [_make_verdict_dict("contradicted", 0, 10)],
            [],
        ]
        result = grader.evaluate(gt, verdicts, response_scores=[0.2, 0.9])
        rl = result["response_level"]
        assert not math.isnan(rl["auc_roc"])

    def test_per_answer_details_length(self, grader):
        logger.debug("Testing evaluate returns per_answer_details with correct length")
        gt = [[], [_make_gt_span(0, 5, "Subtle Conflict")]]
        verdicts = [[], [_make_verdict_dict("contradicted", 0, 5)]]
        result = grader.evaluate(gt, verdicts)
        assert len(result["per_answer_details"]) == 2

    def test_multiple_answers_aggregated_correctly(self, grader):
        logger.debug("Testing evaluate aggregates metrics across multiple answers")
        gt = [
            [_make_gt_span(0, 10, "Evident Conflict")],
            [_make_gt_span(20, 30, "Subtle Baseless Info")],
        ]
        verdicts = [
            [_make_verdict_dict("contradicted", 0, 10)],
            [],  # missed second GT span
        ]
        result = grader.evaluate(gt, verdicts)
        cl = result["claim_level"]
        # 2 GT spans, 1 matched → recall = 0.5
        assert cl["recall"] == pytest.approx(0.5)
        assert cl["n_gt_spans"] == 2
        assert cl["n_gt_matched"] == 1

    def test_f1_formula(self, grader):
        logger.debug("Testing F1 formula: 2*P*R/(P+R)")
        # 2 GT spans, 1 matched (recall=0.5); 1 prediction, 1 matched (precision=1.0)
        gt = [
            [_make_gt_span(0, 10, "Evident Conflict"), _make_gt_span(20, 30, "Subtle Conflict")],
        ]
        verdicts = [
            [_make_verdict_dict("contradicted", 0, 10)],
        ]
        result = grader.evaluate(gt, verdicts)
        cl = result["claim_level"]
        expected_f1 = 2 * 1.0 * 0.5 / (1.0 + 0.5)
        assert cl["f1"] == pytest.approx(expected_f1)
