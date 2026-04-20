"""
Unit tests for uqlm.utils.prompts.groundedness_prompts.

Tests cover:
- Presence and formatting of UNIFIED_GROUNDEDNESS_SYSTEM_PROMPT
- get_unified_groundedness_prompt with different flag combinations
- Required structural elements in the generated prompt
"""

import logging

import pytest

from uqlm.utils.prompts.groundedness_prompts import (
    UNIFIED_GROUNDEDNESS_SYSTEM_PROMPT,
    get_unified_groundedness_prompt,
)

logger = logging.getLogger(__name__)


class TestSystemPrompt:
    def test_system_prompt_is_nonempty_string(self):
        logger.debug("Checking system prompt is a non-empty string")
        assert isinstance(UNIFIED_GROUNDEDNESS_SYSTEM_PROMPT, str)
        assert len(UNIFIED_GROUNDEDNESS_SYSTEM_PROMPT.strip()) > 0

    def test_system_prompt_mentions_json(self):
        logger.debug("System prompt should instruct for JSON output")
        assert "JSON" in UNIFIED_GROUNDEDNESS_SYSTEM_PROMPT

    def test_system_prompt_mentions_hallucination(self):
        assert "hallucination" in UNIFIED_GROUNDEDNESS_SYSTEM_PROMPT.lower()


class TestGetUnifiedGroundednessPrompt:
    @pytest.fixture
    def sample_context(self):
        return "Paris is the capital of France."

    @pytest.fixture
    def sample_answer(self):
        return "The capital of France is Paris, founded in 52 BC."

    def test_returns_string(self, sample_context, sample_answer):
        logger.debug("Testing prompt is returned as a string")
        prompt = get_unified_groundedness_prompt(sample_context, sample_answer)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_context_included(self, sample_context, sample_answer):
        logger.debug("Testing context is included in the prompt")
        prompt = get_unified_groundedness_prompt(sample_context, sample_answer)
        assert sample_context in prompt

    def test_answer_included(self, sample_context, sample_answer):
        logger.debug("Testing answer is included in the prompt")
        prompt = get_unified_groundedness_prompt(sample_context, sample_answer)
        assert sample_answer in prompt

    def test_required_fields_present(self, sample_context, sample_answer):
        """Every prompt variant must request `claim`, `anchor_text`, and `verdict`."""
        logger.debug("Testing required fields claim/anchor_text/verdict are in the prompt")
        prompt = get_unified_groundedness_prompt(sample_context, sample_answer)
        assert '"claim"' in prompt
        assert '"anchor_text"' in prompt
        assert '"verdict"' in prompt

    def test_verdict_enum_present(self, sample_context, sample_answer):
        """The three verdict categories must be documented."""
        logger.debug("Testing verdict enum values are present")
        prompt = get_unified_groundedness_prompt(sample_context, sample_answer)
        assert "supported" in prompt
        assert "baseless" in prompt
        assert "contradicted" in prompt

    def test_reasoning_included_when_flag_true(self, sample_context, sample_answer):
        logger.debug("Testing reasoning field is included when include_reasoning=True")
        prompt = get_unified_groundedness_prompt(
            sample_context, sample_answer, include_reasoning=True
        )
        assert '"reasoning"' in prompt

    def test_reasoning_excluded_when_flag_false(self, sample_context, sample_answer):
        logger.debug("Testing reasoning field is excluded when include_reasoning=False")
        prompt = get_unified_groundedness_prompt(
            sample_context, sample_answer, include_reasoning=False
        )
        assert '"reasoning"' not in prompt

    def test_relevant_context_included_when_flag_true(self, sample_context, sample_answer):
        logger.debug("Testing relevant_context field is included when flag=True")
        prompt = get_unified_groundedness_prompt(
            sample_context, sample_answer, include_relevant_context=True
        )
        assert '"relevant_context"' in prompt

    def test_relevant_context_excluded_when_flag_false(self, sample_context, sample_answer):
        logger.debug("Testing relevant_context field is excluded when flag=False")
        prompt = get_unified_groundedness_prompt(
            sample_context, sample_answer, include_relevant_context=False
        )
        assert '"relevant_context"' not in prompt

    def test_both_flags_false(self, sample_context, sample_answer):
        """With both optional flags disabled, only core fields remain."""
        logger.debug("Testing both optional flags disabled leaves only core fields")
        prompt = get_unified_groundedness_prompt(
            sample_context,
            sample_answer,
            include_reasoning=False,
            include_relevant_context=False,
        )
        assert '"claim"' in prompt
        assert '"anchor_text"' in prompt
        assert '"verdict"' in prompt
        assert '"reasoning"' not in prompt
        assert '"relevant_context"' not in prompt

    def test_field_numbering_is_sequential(self, sample_context, sample_answer):
        """
        Fields should be numbered 1..N with no gaps regardless of flag combo.
        Core claim/anchor are 1 and 2; verdict is always last.
        """
        logger.debug("Testing field numbering is sequential")
        prompt = get_unified_groundedness_prompt(
            sample_context,
            sample_answer,
            include_reasoning=False,
            include_relevant_context=False,
        )
        # Only 3 fields total: claim(1), anchor_text(2), verdict(3)
        assert '1. "claim"' in prompt
        assert '2. "anchor_text"' in prompt
        assert '3. "verdict"' in prompt

    def test_field_numbering_all_flags(self, sample_context, sample_answer):
        logger.debug("Testing field numbering with all flags enabled")
        prompt = get_unified_groundedness_prompt(
            sample_context,
            sample_answer,
            include_reasoning=True,
            include_relevant_context=True,
        )
        assert '1. "claim"' in prompt
        assert '2. "anchor_text"' in prompt
        assert '3. "relevant_context"' in prompt
        assert '4. "reasoning"' in prompt
        assert '5. "verdict"' in prompt

    def test_empty_context_and_answer_still_works(self):
        logger.debug("Testing prompt generation with empty context and answer")
        prompt = get_unified_groundedness_prompt("", "")
        assert isinstance(prompt, str)
        assert '"verdict"' in prompt

    def test_output_format_mentions_json_array(self, sample_context, sample_answer):
        logger.debug("Testing prompt instructs to output a JSON array")
        prompt = get_unified_groundedness_prompt(sample_context, sample_answer)
        assert "JSON array" in prompt or "JSON" in prompt
        assert "[" in prompt and "]" in prompt
