"""
Unit tests for uqlm.scorers.longform.linear_probe.

The module has heavy optional dependencies (``torch``, ``transformers``, ``peft``,
``huggingface_hub``). These tests focus on:

- The pure-python dataclasses (``HallucinationSpan``, ``LinearProbeResult``)
- The availability-check helpers (``_check_torch`` etc.)
- The ``_add_hooks`` context manager
- The ``_get_model_layers`` helper with fake model objects
- The ``_scores_to_spans`` post-processing logic (tested through a probe instance
  whose heavy initialization is mocked away)

Scorer initialization is tested via full mocking of the model/tokenizer/probe
loading pipeline so tests run even without torch installed.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from uqlm.scorers.longform import linear_probe as lp_module
from uqlm.scorers.longform.linear_probe import (
    HallucinationSpan,
    LinearProbeResult,
    LinearProbeScorer,
    _add_hooks,
    _check_hf_hub,
    _check_peft,
    _check_torch,
    _check_transformers,
    _get_model_layers,
    _require_torch,
    _require_transformers,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Availability checks (cached globals)
# ---------------------------------------------------------------------------


class TestAvailabilityChecks:
    def test_check_torch_returns_bool(self):
        logger.debug("Testing _check_torch returns a bool")
        assert isinstance(_check_torch(), bool)

    def test_check_transformers_returns_bool(self):
        assert isinstance(_check_transformers(), bool)

    def test_check_peft_returns_bool(self):
        assert isinstance(_check_peft(), bool)

    def test_check_hf_hub_returns_bool(self):
        assert isinstance(_check_hf_hub(), bool)

    def test_check_torch_cached(self):
        """Second call must return the same value (caching via module-level global)."""
        logger.debug("Testing _check_torch result is cached")
        first = _check_torch()
        second = _check_torch()
        assert first == second

    def test_require_torch_raises_when_missing(self, monkeypatch):
        logger.debug("Testing _require_torch raises ImportError when torch missing")
        monkeypatch.setattr(lp_module, "_TORCH_AVAILABLE", False)
        with pytest.raises(ImportError, match="PyTorch"):
            _require_torch()

    def test_require_transformers_raises_when_missing(self, monkeypatch):
        logger.debug("Testing _require_transformers raises ImportError when missing")
        monkeypatch.setattr(lp_module, "_TRANSFORMERS_AVAILABLE", False)
        with pytest.raises(ImportError, match="transformers"):
            _require_transformers()


# ---------------------------------------------------------------------------
# _add_hooks
# ---------------------------------------------------------------------------


class TestAddHooks:
    def test_registers_and_removes_hooks(self):
        logger.debug("Testing _add_hooks installs and cleans up handles")
        fake_handle = MagicMock()
        fake_module = MagicMock()
        fake_module.register_forward_hook.return_value = fake_handle

        def hook_fn(m, i, o):
            return None

        with _add_hooks([(fake_module, hook_fn)]):
            fake_module.register_forward_hook.assert_called_once_with(hook_fn)

        fake_handle.remove.assert_called_once()

    def test_cleanup_on_exception(self):
        logger.debug("Testing _add_hooks removes handles even when exception raised")
        fake_handle = MagicMock()
        fake_module = MagicMock()
        fake_module.register_forward_hook.return_value = fake_handle

        with pytest.raises(RuntimeError, match="boom"):
            with _add_hooks([(fake_module, lambda m, i, o: None)]):
                raise RuntimeError("boom")

        fake_handle.remove.assert_called_once()


# ---------------------------------------------------------------------------
# _get_model_layers
# ---------------------------------------------------------------------------


class TestGetModelLayers:
    def test_llama_qwen_style(self, monkeypatch):
        """Models with ``.model.layers`` (LLaMA/Qwen/Mistral)."""
        logger.debug("Testing _get_model_layers for LLaMA/Qwen-style model")
        # Force peft check to return False so no PeftModel unwrap
        monkeypatch.setattr(lp_module, "_PEFT_AVAILABLE", False)

        layers = [MagicMock(), MagicMock(), MagicMock()]
        inner = SimpleNamespace(layers=layers)
        model = SimpleNamespace(model=inner)
        assert _get_model_layers(model) == layers

    def test_gpt2_style(self, monkeypatch):
        """Models with ``.transformer.h`` (GPT-2)."""
        logger.debug("Testing _get_model_layers for GPT-2-style model")
        monkeypatch.setattr(lp_module, "_PEFT_AVAILABLE", False)

        layers = [MagicMock()]
        model = SimpleNamespace(transformer=SimpleNamespace(h=layers))
        assert _get_model_layers(model) == layers

    def test_gpt_neox_style(self, monkeypatch):
        """Models with ``.gpt_neox.layers`` (GPT-NeoX)."""
        logger.debug("Testing _get_model_layers for GPT-NeoX-style model")
        monkeypatch.setattr(lp_module, "_PEFT_AVAILABLE", False)

        layers = [MagicMock(), MagicMock()]
        model = SimpleNamespace(gpt_neox=SimpleNamespace(layers=layers))
        assert _get_model_layers(model) == layers

    def test_unknown_architecture_raises(self, monkeypatch):
        logger.debug("Testing _get_model_layers raises on unknown architecture")
        monkeypatch.setattr(lp_module, "_PEFT_AVAILABLE", False)

        model = SimpleNamespace(foo="bar")
        with pytest.raises(ValueError, match="Unknown model architecture"):
            _get_model_layers(model)


# ---------------------------------------------------------------------------
# HallucinationSpan / LinearProbeResult dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_hallucination_span_defaults(self):
        logger.debug("Testing HallucinationSpan instantiation with default token_scores")
        span = HallucinationSpan(
            text="foo", start_offset=0, end_offset=3, score=0.9
        )
        assert span.text == "foo"
        assert span.start_offset == 0
        assert span.end_offset == 3
        assert span.score == 0.9
        assert span.token_scores == []

    def test_hallucination_span_with_scores(self):
        logger.debug("Testing HallucinationSpan with custom token_scores")
        span = HallucinationSpan(
            text="bar", start_offset=5, end_offset=8, score=0.7,
            token_scores=[0.6, 0.7],
        )
        assert span.token_scores == [0.6, 0.7]

    def test_linear_probe_result_fields(self):
        logger.debug("Testing LinearProbeResult basic field assignment")
        result = LinearProbeResult(
            queries=["q"],
            contexts=["c"],
            answers=["a"],
            token_scores=[[0.1, 0.9]],
            hallucination_spans=[[]],
            response_scores=[0.1],
        )
        assert result.queries == ["q"]
        assert result.metadata == {}


# ---------------------------------------------------------------------------
# LinearProbeScorer with fully mocked init — tests post-processing logic
# ---------------------------------------------------------------------------


def _make_scorer_without_init() -> LinearProbeScorer:
    """Build a LinearProbeScorer instance bypassing __init__ for unit tests."""
    scorer = LinearProbeScorer.__new__(LinearProbeScorer)
    scorer.threshold = 0.5
    scorer.merge_gap = 2
    return scorer


class TestScoresToSpans:
    def test_empty_token_probs_returns_empty(self):
        logger.debug("Testing _scores_to_spans with empty token_probs")
        s = _make_scorer_without_init()
        assert s._scores_to_spans("answer", [], []) == []

    def test_no_tokens_above_threshold(self):
        logger.debug("Testing _scores_to_spans with no tokens above threshold")
        s = _make_scorer_without_init()
        probs = [0.1, 0.2, 0.3]
        offsets = [(0, 1), (1, 2), (2, 3)]
        assert s._scores_to_spans("abc", probs, offsets) == []

    def test_single_hallucinated_token(self):
        logger.debug("Testing _scores_to_spans with a single hallucinated token")
        s = _make_scorer_without_init()
        probs = [0.1, 0.9, 0.1]
        offsets = [(0, 1), (2, 5), (6, 7)]
        answer = "a bcd e"
        spans = s._scores_to_spans(answer, probs, offsets)
        assert len(spans) == 1
        span = spans[0]
        assert span.start_offset == 2
        assert span.end_offset == 5
        assert span.text == "bcd"
        assert span.score == pytest.approx(0.9)
        assert span.token_scores == [pytest.approx(0.9)]

    def test_adjacent_tokens_merged(self):
        """Adjacent hallucinated tokens should merge into one span."""
        logger.debug("Testing _scores_to_spans merges adjacent hallucinated tokens")
        s = _make_scorer_without_init()
        probs = [0.8, 0.9, 0.1]
        offsets = [(0, 3), (4, 7), (8, 11)]
        answer = "foo bar baz"
        spans = s._scores_to_spans(answer, probs, offsets)
        assert len(spans) == 1
        assert spans[0].start_offset == 0
        assert spans[0].end_offset == 7
        assert spans[0].score == pytest.approx(0.9)

    def test_tokens_separated_by_gap_greater_than_merge_gap_split(self):
        """Tokens with gap > merge_gap yield separate spans."""
        logger.debug("Testing _scores_to_spans splits tokens separated by large gap")
        s = _make_scorer_without_init()
        s.merge_gap = 0  # no merging across any gap
        probs = [0.9, 0.1, 0.9]
        offsets = [(0, 3), (4, 7), (8, 11)]
        answer = "foo bar baz"
        spans = s._scores_to_spans(answer, probs, offsets)
        assert len(spans) == 2
        assert spans[0].text == "foo"
        assert spans[1].text == "baz"

    def test_tokens_with_small_gap_merged(self):
        """Tokens with gap <= merge_gap should merge."""
        logger.debug("Testing _scores_to_spans merges tokens with small gap")
        s = _make_scorer_without_init()
        s.merge_gap = 2
        probs = [0.9, 0.1, 0.9]  # gap of 1 non-hallucinated token
        offsets = [(0, 3), (4, 7), (8, 11)]
        answer = "foo bar baz"
        spans = s._scores_to_spans(answer, probs, offsets)
        assert len(spans) == 1
        assert spans[0].start_offset == 0
        assert spans[0].end_offset == 11

    def test_tokens_with_invalid_offsets_skipped(self):
        logger.debug("Testing _scores_to_spans skips tokens with (-1,-1) offsets")
        s = _make_scorer_without_init()
        probs = [0.9, 0.9]
        offsets = [(-1, -1), (0, 3)]
        answer = "foo"
        spans = s._scores_to_spans(answer, probs, offsets)
        assert len(spans) == 1
        assert spans[0].start_offset == 0
        assert spans[0].end_offset == 3

    def test_spans_sorted_by_start_offset(self):
        logger.debug("Testing _scores_to_spans returns spans sorted by start_offset")
        s = _make_scorer_without_init()
        s.merge_gap = 0
        probs = [0.9, 0.1, 0.9]
        offsets = [(5, 8), (9, 10), (0, 3)]  # out-of-order offsets (unusual)
        answer = "abc x fgh"  # length 9 — offsets constructed just for test logic
        spans = s._scores_to_spans(answer, probs, offsets)
        # If the order of hallucinated tokens is preserved → first span starts at 5,
        # second at 0; sorting should reorder them
        starts = [sp.start_offset for sp in spans]
        assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# LinearProbeScorer input validation
# ---------------------------------------------------------------------------


class TestScoreInputValidation:
    def test_length_mismatch_raises(self):
        logger.debug("Testing score() raises on input length mismatch")
        # Skip heavy init by going directly through __new__ + attr setup
        scorer = _make_scorer_without_init()
        with pytest.raises(ValueError, match="Input lists must have equal length"):
            scorer.score(queries=["q1"], contexts=["c1", "c2"], answers=["a1"])


# ---------------------------------------------------------------------------
# Scorer initialization with fully mocked dependencies
# ---------------------------------------------------------------------------


class TestScorerInitialization:
    """Initialization is tested with every heavy dependency mocked away."""

    def test_init_with_mocked_model(self):
        logger.debug("Testing LinearProbeScorer.__init__ with fully mocked dependencies")
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_torch.backends.mps.is_available.return_value = False
        fake_torch.device = lambda s: f"device({s})"
        fake_torch.float32 = "float32"

        with (
            patch.object(lp_module, "_require_torch", return_value=fake_torch),
            patch.object(lp_module, "_require_transformers"),
            patch.object(
                LinearProbeScorer, "_load_model_and_tokenizer",
                return_value=(MagicMock(), MagicMock()),
            ),
            patch.object(
                LinearProbeScorer, "_load_probe",
                return_value=(MagicMock(), 5, MagicMock()),
            ),
        ):
            scorer = LinearProbeScorer(
                model_name="fake/model",
                probe_repo_id="fake/repo",
                probe_id="fake_probe",
                device="cpu",
                torch_dtype="float32",
                threshold=0.4,
                merge_gap=3,
            )
            assert scorer.model_name == "fake/model"
            assert scorer.probe_id == "fake_probe"
            assert scorer.threshold == 0.4
            assert scorer.merge_gap == 3
            assert scorer._probe_layer_idx == 5
