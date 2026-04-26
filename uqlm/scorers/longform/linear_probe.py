"""
Token-level Linear Probe Scorer for hallucination detection.

This module implements a hallucination detector based on the paper
*"Real-Time Detection of Hallucinated Entities in Long-Form Generation"*
(Obeso et al., 2026). It uses pre-trained linear probes that read hidden
states from an intermediate layer of a language model and predict per-token
hallucination probabilities.

The scorer loads a full ``transformers`` model, attaches a lightweight linear
head (and optionally LoRA adapters via ``peft``), and runs a single forward
pass to produce token-level scores. These scores are then post-processed
into character-level hallucination spans compatible with
:class:`~uqlm.longform.benchmark.ragtruth_grader.RAGTruthGrader`.

Why hooks instead of ``output_hidden_states=True``?
    Using ``output_hidden_states=True`` forces the model to return hidden
    states from **all** layers, which is very memory-expensive for 7B+
    parameter models. A forward hook captures only the single layer we
    need, keeping memory usage minimal. This is the same approach used
    in the original hallucination probes repository.

Example
-------
>>> scorer = LinearProbeScorer(
...     model_name="Qwen/Qwen2.5-7B-Instruct",
...     probe_repo_id="obalcells/hallucination-probes",
...     probe_id="qwen2_5_7b_linear",
... )
>>> result = scorer.score(
...     queries=["What is the capital of France?"],
...     contexts=["France is a country in Western Europe. Its capital is Paris."],
...     answers=["The capital of France is Paris, founded in 52 BC by the Romans."],
... )
>>> result.response_scores
[0.73]
>>> result.hallucination_spans[0]
[HallucinationSpan(text='founded in 52 BC by the Romans', ...)]
"""

import contextlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports for heavy dependencies (torch, transformers, peft, huggingface_hub)
# ---------------------------------------------------------------------------

_TORCH_AVAILABLE = None
_TRANSFORMERS_AVAILABLE = None
_PEFT_AVAILABLE = None
_HF_HUB_AVAILABLE = None


def _check_torch():
    global _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is None:
        try:
            import torch  # noqa: F401
            _TORCH_AVAILABLE = True
        except ImportError:
            _TORCH_AVAILABLE = False
    return _TORCH_AVAILABLE


def _check_transformers():
    global _TRANSFORMERS_AVAILABLE
    if _TRANSFORMERS_AVAILABLE is None:
        try:
            import transformers  # noqa: F401
            _TRANSFORMERS_AVAILABLE = True
        except ImportError:
            _TRANSFORMERS_AVAILABLE = False
    return _TRANSFORMERS_AVAILABLE


def _check_peft():
    global _PEFT_AVAILABLE
    if _PEFT_AVAILABLE is None:
        try:
            import peft  # noqa: F401
            _PEFT_AVAILABLE = True
        except ImportError:
            _PEFT_AVAILABLE = False
    return _PEFT_AVAILABLE


def _check_hf_hub():
    global _HF_HUB_AVAILABLE
    if _HF_HUB_AVAILABLE is None:
        try:
            import huggingface_hub  # noqa: F401
            _HF_HUB_AVAILABLE = True
        except ImportError:
            _HF_HUB_AVAILABLE = False
    return _HF_HUB_AVAILABLE


def _require_torch():
    if not _check_torch():
        raise ImportError(
            "LinearProbeScorer requires PyTorch. Install it with: pip install torch"
        )
    import torch
    return torch


def _require_transformers():
    if not _check_transformers():
        raise ImportError(
            "LinearProbeScorer requires transformers. "
            "Install it with: pip install transformers"
        )
    import transformers
    return transformers


# ---------------------------------------------------------------------------
# Hook context manager
# Adapted from hallucination_probes/utils/hooks.py — captures hidden states
# from a single layer without storing all layers' outputs.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _add_hooks(module_forward_hooks: List[Tuple[Any, Any]]):
    """
    Context manager for temporarily adding forward hooks to model modules.

    Parameters
    ----------
    module_forward_hooks : list of (module, hook_fn)
        Pairs of modules and their hook functions.
    """
    handles = []
    try:
        for module, hook in module_forward_hooks:
            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


# ---------------------------------------------------------------------------
# Model utilities
# Adapted from hallucination_probes/utils/model_utils.py
# ---------------------------------------------------------------------------


def _get_model_layers(model) -> list:
    """
    Get the list of transformer decoder layers from a model.

    Supports LLaMA, Qwen, Mistral, Gemma, GPT-2, GPT-NeoX architectures.
    Handles ``PeftModel`` wrappers transparently.
    """
    if _check_peft():
        from peft import PeftModel
        if isinstance(model, PeftModel):
            base_model = model.get_base_model()
        else:
            base_model = model
    else:
        base_model = model

    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        return list(base_model.model.layers)
    elif hasattr(base_model, "transformer") and hasattr(base_model.transformer, "h"):
        return list(base_model.transformer.h)
    elif hasattr(base_model, "gpt_neox") and hasattr(base_model.gpt_neox, "layers"):
        return list(base_model.gpt_neox.layers)
    else:
        raise ValueError(
            f"Unknown model architecture: {type(base_model)}. "
            "Cannot extract transformer layers."
        )


# ---------------------------------------------------------------------------
# Probe downloading
# Adapted from hallucination_probes/utils/probe_loader.py
# ---------------------------------------------------------------------------


def _download_probe_from_hf(
    repo_id: str,
    probe_id: str,
    local_folder: Path,
    token: Optional[str] = None,
) -> None:
    """
    Download probe files from a HuggingFace repository.

    Downloads ``probe_head.bin``, ``probe_config.json``, and optionally
    LoRA adapter files (``adapter_config.json``, ``adapter_model.*``).

    Parameters
    ----------
    repo_id : str
        HuggingFace repository ID (e.g. ``"obalcells/hallucination-probes"``).
    probe_id : str
        Subfolder name within the repo (e.g. ``"qwen2_5_7b_linear"``).
    local_folder : Path
        Local directory to save the downloaded files.
    token : str, optional
        HuggingFace API token for private repos.
    """
    if not _check_hf_hub():
        raise ImportError(
            "Downloading probes requires huggingface_hub. "
            "Install it with: pip install huggingface_hub"
        )
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    local_folder.mkdir(parents=True, exist_ok=True)

    repo_files = api.list_repo_files(
        repo_id=repo_id, repo_type="model", revision="main"
    )

    prefix = f"{probe_id}/"
    subfolder_files = [f for f in repo_files if f.startswith(prefix)]

    if not subfolder_files:
        raise FileNotFoundError(
            f"No files found for probe_id={probe_id!r} in repo {repo_id}. "
            f"Available top-level entries: "
            f"{sorted(set(f.split('/')[0] for f in repo_files))}"
        )

    for file_path in subfolder_files:
        relative_path = file_path[len(prefix):]
        local_file_path = local_folder / relative_path
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            token=token,
        )
        shutil.copy(downloaded_file, local_file_path)

    logger.info("Downloaded probe %s from %s to %s", probe_id, repo_id, local_folder)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HallucinationSpan:
    """
    A contiguous span of text in the answer detected as hallucinated.

    Attributes
    ----------
    text : str
        Verbatim text from the answer.
    start_offset : int
        Character offset in the original answer (inclusive).
    end_offset : int
        End character offset in the original answer (exclusive).
    score : float
        Aggregated hallucination score for this span (max of token probabilities).
    token_scores : List[float]
        Per-token hallucination probabilities within this span.
    """

    text: str
    start_offset: int
    end_offset: int
    score: float
    token_scores: List[float] = field(default_factory=list)


@dataclass
class LinearProbeResult:
    """
    Result of linear probe hallucination scoring.

    Attributes
    ----------
    queries : List[str]
        The original queries.
    contexts : List[str]
        The contexts (retrieved documents).
    answers : List[str]
        The generated answers that were scored.
    token_scores : List[List[float]]
        Per-token hallucination probabilities for the answer portion of each example.
    hallucination_spans : List[List[HallucinationSpan]]
        Detected hallucination spans per answer.
    response_scores : List[float]
        Response-level groundedness score per answer.
        Computed as ``1 - max(token_probs)`` so that lower = more hallucinated,
        matching the convention of ``UnifiedGroundednessScorer``.
    metadata : Dict[str, Any]
        Scoring metadata (model name, probe id, threshold, etc.).
    """

    queries: List[str]
    contexts: List[str]
    answers: List[str]
    token_scores: List[List[float]]
    hallucination_spans: List[List[HallucinationSpan]]
    response_scores: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def claim_verdicts(self) -> List[List[Any]]:
        """
        Convert hallucination spans to ``ClaimVerdict`` objects for
        compatibility with
        :class:`~uqlm.longform.benchmark.ragtruth_grader.RAGTruthGrader`.

        Each ``HallucinationSpan`` becomes a ``ClaimVerdict`` with
        ``verdict="contradicted"`` and ``score=0.0``.

        Non-hallucinated portions of the answer are NOT included —
        only detected hallucination spans are returned. This matches
        the RAGTruthGrader's expectation.
        """
        from uqlm.scorers.longform.context_groundedness import ClaimVerdict

        all_verdicts: List[List[ClaimVerdict]] = []
        for spans in self.hallucination_spans:
            verdicts = []
            for span in spans:
                verdicts.append(
                    ClaimVerdict(
                        claim=span.text,
                        anchor_text=span.text,
                        verdict="contradicted",
                        score=0.0,
                        start_offset=span.start_offset,
                        end_offset=span.end_offset,
                        relevant_context=[],
                        reasoning=f"Probe score: {span.score:.3f}",
                    )
                )
            all_verdicts.append(verdicts)
        return all_verdicts


# ---------------------------------------------------------------------------
# Main scorer class
# ---------------------------------------------------------------------------


class LinearProbeScorer:
    """
    Scores RAG answers using a pre-trained linear probe on LLM hidden states.

    This scorer implements the approach from *"Real-Time Detection of
    Hallucinated Entities in Long-Form Generation"* (Obeso et al., 2026).
    It loads a ``transformers`` causal LM, attaches a linear probe head
    (and optionally LoRA adapters) to an intermediate layer, and predicts
    per-token hallucination probabilities via a single forward pass.

    The probe hooks into a single layer using a forward hook (not
    ``output_hidden_states=True``) to avoid storing hidden states from
    all layers, which would be prohibitively expensive for 7B+ models.

    Parameters
    ----------
    model_name : str
        HuggingFace model name (e.g. ``"Qwen/Qwen2.5-7B-Instruct"``).
    probe_repo_id : str
        HuggingFace repository containing pre-trained probes.
    probe_id : str
        Subfolder name within the probe repo (e.g. ``"qwen2_5_7b_linear"``).
    probe_local_path : Path or str, optional
        Path to a **locally-trained** probe directory (e.g. the output of
        ``external/hallucination_probes`` training pipeline, typically at
        ``external/hallucination_probes/value_head_probes/<probe_id>/``).
        When provided, ``probe_repo_id`` and ``probe_cache_dir`` are ignored
        and the probe is loaded directly from this path.  The directory must
        contain ``probe_config.json`` and ``probe_head.bin``.
    device : str, optional
        Device to load the model on. If ``None``, auto-detects (CUDA > MPS > CPU).
    torch_dtype : optional
        Data type for model weights. If ``None``, uses bfloat16 if supported.
    threshold : float
        Hallucination probability threshold for span detection.
    merge_gap : int
        Maximum gap (in tokens) between hallucinated tokens to merge into
        one contiguous span.
    probe_cache_dir : Path or str, optional
        Local directory for caching downloaded probe weights.
        Defaults to ``~/.cache/hallucination_probes``.
    hf_token : str, optional
        HuggingFace API token for private repos.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        probe_repo_id: str = "obalcells/hallucination-probes",
        probe_id: str = "qwen2_5_7b_linear",
        probe_local_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[Any] = None,
        threshold: float = 0.3,
        merge_gap: int = 2,
        probe_cache_dir: Optional[Union[str, Path]] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        torch = _require_torch()
        _require_transformers()

        self.model_name = model_name
        self.probe_repo_id = probe_repo_id
        self.probe_id = probe_id
        self.probe_local_path = Path(probe_local_path) if probe_local_path is not None else None
        self.threshold = threshold
        self.merge_gap = merge_gap

        # Resolve device
        if device is None:
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)

        # Resolve dtype
        if torch_dtype is None:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self._dtype = torch.bfloat16
            else:
                self._dtype = torch.float32
        else:
            self._dtype = torch_dtype

        # Resolve probe cache directory
        if probe_cache_dir is None:
            self._probe_cache_dir = Path.home() / ".cache" / "hallucination_probes"
        else:
            self._probe_cache_dir = Path(probe_cache_dir)

        self._hf_token = hf_token

        # Load everything
        self._model, self._tokenizer = self._load_model_and_tokenizer()
        self._probe_head, self._probe_layer_idx, self._target_module = (
            self._load_probe()
        )

        logger.info(
            "LinearProbeScorer initialized: model=%s, probe=%s, "
            "layer=%d, device=%s, dtype=%s, threshold=%.2f",
            model_name,
            probe_id,
            self._probe_layer_idx,
            self._device,
            self._dtype,
            threshold,
        )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _load_model_and_tokenizer(self):
        """Load the base model and tokenizer from HuggingFace."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(
            "Loading model %s on %s (%s)...",
            self.model_name, self._device, self._dtype,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=(
                str(self._device) if self._device.type != "cpu" else "cpu"
            ),
            torch_dtype=self._dtype,
            trust_remote_code=True,
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def _load_probe(self):
        """
        Load probe weights (and optionally LoRA adapters) from cache or HF.

        If ``self.probe_local_path`` is set, the probe is loaded directly from
        that directory (bypassing HF download).  Otherwise the probe is
        downloaded from ``self.probe_repo_id`` and cached locally.

        Returns
        -------
        probe_head : torch.nn.Linear
            The linear probe head.
        probe_layer_idx : int
            The layer index where the probe hooks.
        target_module : torch.nn.Module
            The specific layer module to hook into.
        """
        torch = _require_torch()
        import torch.nn as nn

        # Determine where to load the probe from
        if self.probe_local_path is not None:
            # Load directly from a locally-trained probe directory
            probe_local_dir = self.probe_local_path
            if not probe_local_dir.is_dir():
                raise FileNotFoundError(
                    f"probe_local_path={probe_local_dir!r} is not a directory. "
                    "Make sure the training pipeline has completed and the path "
                    "points to the probe output directory (containing "
                    "probe_config.json and probe_head.bin)."
                )
            if not (probe_local_dir / "probe_config.json").exists():
                raise FileNotFoundError(
                    f"probe_config.json not found in {probe_local_dir}. "
                    "The probe directory appears incomplete."
                )
            logger.info("Loading probe from local path: %s", probe_local_dir)
        else:
            probe_local_dir = self._probe_cache_dir / self.probe_id

            # Download if not cached
            if not (probe_local_dir / "probe_config.json").exists():
                logger.info(
                    "Probe not found locally at %s, downloading from %s...",
                    probe_local_dir,
                    self.probe_repo_id,
                )
                _download_probe_from_hf(
                    repo_id=self.probe_repo_id,
                    probe_id=self.probe_id,
                    local_folder=probe_local_dir,
                    token=self._hf_token,
                )

        # Load probe config
        with open(probe_local_dir / "probe_config.json") as f:
            probe_config = json.load(f)

        hidden_size = probe_config["hidden_size"]
        probe_layer_idx = probe_config["layer_idx"]

        logger.info(
            "Probe config: hidden_size=%d, layer_idx=%d",
            hidden_size,
            probe_layer_idx,
        )

        # Load LoRA adapters if present
        adapter_config_path = probe_local_dir / "adapter_config.json"
        if adapter_config_path.exists():
            if not _check_peft():
                raise ImportError(
                    f"Probe {self.probe_id} includes LoRA adapters but peft "
                    "is not installed. Install it with: pip install peft"
                )
            from peft import PeftModel

            logger.info("Loading LoRA adapters from %s", probe_local_dir)
            self._model = PeftModel.from_pretrained(
                self._model, str(probe_local_dir)
            )
            self._model.eval()

        # Load probe head weights
        probe_head = nn.Linear(
            hidden_size, 1, device=self._device, dtype=self._dtype
        )
        state_dict = torch.load(
            probe_local_dir / "probe_head.bin",
            map_location=self._device,
            weights_only=True,
        )
        probe_head.load_state_dict(state_dict)
        probe_head.eval()

        # Get the target layer module
        model_layers = _get_model_layers(self._model)
        if probe_layer_idx >= len(model_layers):
            raise ValueError(
                f"Probe layer_idx={probe_layer_idx} exceeds model layer count "
                f"({len(model_layers)}). Is the probe compatible with "
                f"{self.model_name}?"
            )
        target_module = model_layers[probe_layer_idx]

        return probe_head, probe_layer_idx, target_module

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        queries: List[str],
        contexts: List[str],
        answers: List[str],
        batch_size: int = 1,
    ) -> LinearProbeResult:
        """
        Score answers using the linear probe on model hidden states.

        Parameters
        ----------
        queries : List[str]
            The original queries.
        contexts : List[str]
            The contexts (retrieved documents).
        answers : List[str]
            The generated answers to score.
        batch_size : int
            Batch size for inference. Default is 1 to minimize memory usage.

        Returns
        -------
        LinearProbeResult
            Contains per-token scores, hallucination spans, and response scores.

        Raises
        ------
        ValueError
            If input lists have mismatched lengths.
        """
        if not (len(queries) == len(contexts) == len(answers)):
            raise ValueError(
                f"Input lists must have equal length. "
                f"Got queries={len(queries)}, contexts={len(contexts)}, "
                f"answers={len(answers)}."
            )

        all_token_scores: List[List[float]] = []
        all_spans: List[List[HallucinationSpan]] = []
        all_response_scores: List[float] = []

        for j in range(len(queries)):
            token_scores, spans, response_score = self._score_single(
                query=queries[j],
                context=contexts[j],
                answer=answers[j],
            )
            all_token_scores.append(token_scores)
            all_spans.append(spans)
            all_response_scores.append(response_score)

        return LinearProbeResult(
            queries=queries,
            contexts=contexts,
            answers=answers,
            token_scores=all_token_scores,
            hallucination_spans=all_spans,
            response_scores=all_response_scores,
            metadata={
                "model_name": self.model_name,
                "probe_id": self.probe_id,
                "probe_repo_id": self.probe_repo_id,
                "probe_local_path": str(self.probe_local_path) if self.probe_local_path is not None else None,
                "probe_layer_idx": self._probe_layer_idx,
                "threshold": self.threshold,
                "merge_gap": self.merge_gap,
                "device": str(self._device),
                "dtype": str(self._dtype),
            },
        )

    # ------------------------------------------------------------------
    # Internal scoring pipeline
    # ------------------------------------------------------------------

    def _score_single(
        self,
        query: str,
        context: str,
        answer: str,
    ) -> Tuple[List[float], List[HallucinationSpan], float]:
        """
        Score a single (query, context, answer) tuple.

        Returns
        -------
        token_scores : List[float]
            Per-token hallucination probabilities for the answer tokens.
        spans : List[HallucinationSpan]
            Detected hallucination spans.
        response_score : float
            Response-level groundedness score (1 - max_token_prob).
        """
        torch = _require_torch()

        # Build chat input and identify answer token boundaries
        input_ids, attention_mask, answer_start_idx = self._build_chat_input(
            query=query, context=context, answer=answer,
        )

        # Forward pass with probe hook
        token_probs = self._forward_with_probe(input_ids, attention_mask)

        # Extract answer-only scores
        answer_probs = token_probs[answer_start_idx:].tolist()

        # Compute response-level score: 1 - max(hallucination_prob)
        if answer_probs:
            response_score = 1.0 - float(max(answer_probs))
        else:
            response_score = 1.0

        # Get token-to-character mapping for the answer
        char_offsets = self._get_answer_token_char_offsets(
            input_ids=input_ids,
            answer_start_idx=answer_start_idx,
            answer_text=answer,
        )

        # Convert token scores to hallucination spans
        spans = self._scores_to_spans(
            answer_text=answer,
            token_probs=answer_probs,
            char_offsets=char_offsets,
        )

        return answer_probs, spans, response_score

    def _build_chat_input(
        self,
        query: str,
        context: str,
        answer: str,
    ) -> Tuple[Any, Any, int]:
        """
        Build tokenized chat input with query, context, and answer.

        The context is prepended to the query in the user message.
        The answer is placed in the assistant turn.

        Returns
        -------
        input_ids : torch.Tensor
            Token IDs of shape ``[1, seq_len]``.
        attention_mask : torch.Tensor
            Attention mask of shape ``[1, seq_len]``.
        answer_start_idx : int
            Token index where the answer portion begins.
        """
        # Build the user message with context included in the query
        if context:
            user_content = (
                f"Context:\n{context}\n\n"
                f"Question:\n{query}"
            )
        else:
            user_content = query

        # Tokenize without the answer to find the prefix length.
        # apply_chat_template with add_generation_prompt=True gives us
        # everything up to where the assistant response would start.
        messages_without_answer = [
            {"role": "user", "content": user_content},
        ]
        prefix_text = self._tokenizer.apply_chat_template(
            messages_without_answer,
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix_ids = self._tokenizer.encode(
            prefix_text, add_special_tokens=False,
        )
        prefix_len = len(prefix_ids)

        # Build full conversation with the answer
        messages_with_answer = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ]
        full_text = self._tokenizer.apply_chat_template(
            messages_with_answer,
            tokenize=False,
            add_generation_prompt=False,
        )

        encoded = self._tokenizer(
            full_text,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)

        logger.debug(
            "Chat input: total_tokens=%d, prefix_len=%d, answer_tokens=%d",
            input_ids.shape[1],
            prefix_len,
            input_ids.shape[1] - prefix_len,
        )

        return input_ids, attention_mask, prefix_len

    def _forward_with_probe(
        self,
        input_ids: Any,
        attention_mask: Any,
    ) -> Any:
        """
        Run forward pass through the model and apply the probe head.

        Uses a forward hook on the target layer to capture hidden states
        from only that layer (memory-efficient compared to
        ``output_hidden_states=True`` which returns all layers).

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape ``[1, seq_len]``.
        attention_mask : torch.Tensor
            Attention mask of shape ``[1, seq_len]``.

        Returns
        -------
        token_probs : torch.Tensor
            Per-token hallucination probabilities of shape ``[seq_len]``.
        """
        torch = _require_torch()

        # Mutable container for the hook to write into
        hooked_hidden_states = [None]

        def hook_fn(module, module_input, module_output):
            """Capture hidden states from the target layer."""
            if isinstance(module_output, tuple) and module_output[0].ndim == 3:
                hooked_hidden_states[0] = module_output[0]
            else:
                hooked_hidden_states[0] = module_output

        with torch.no_grad():
            with _add_hooks([(self._target_module, hook_fn)]):
                self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                )

        if hooked_hidden_states[0] is None:
            raise RuntimeError(
                f"Failed to capture hidden states from layer "
                f"{self._probe_layer_idx}. The hook did not fire."
            )

        # Apply probe head: [1, seq_len, hidden_size] -> [1, seq_len, 1]
        hidden = hooked_hidden_states[0].to(self._probe_head.weight.device)
        probe_logits = self._probe_head(hidden)
        token_probs = torch.sigmoid(probe_logits).squeeze(-1).squeeze(0)

        return token_probs.cpu()

    # ------------------------------------------------------------------
    # Token-to-character mapping
    # ------------------------------------------------------------------

    def _get_answer_token_char_offsets(
        self,
        input_ids: Any,
        answer_start_idx: int,
        answer_text: str,
    ) -> List[Tuple[int, int]]:
        """
        Compute character offsets for each answer token relative to the
        original answer text.

        Decodes each answer token and tracks cumulative character positions
        to build a mapping from token index to ``(start_char, end_char)``
        in the original answer.

        Parameters
        ----------
        input_ids : torch.Tensor
            Full input token IDs of shape ``[1, seq_len]``.
        answer_start_idx : int
            Token index where the answer portion begins.
        answer_text : str
            The original answer text.

        Returns
        -------
        char_offsets : List[Tuple[int, int]]
            ``(start_char, end_char)`` for each answer token, relative to
            ``answer_text``. Returns ``(-1, -1)`` for tokens that cannot
            be mapped.
        """
        answer_token_ids = input_ids[0, answer_start_idx:].tolist()

        # Decode the full answer portion from tokens to verify alignment
        decoded_answer = self._tokenizer.decode(
            answer_token_ids,
            skip_special_tokens=True,
        )

        # Build char offsets by decoding tokens one-by-one and searching
        # for each decoded token string in the answer text.
        # We walk through the answer text left-to-right to handle repeated tokens.
        char_offsets: List[Tuple[int, int]] = []
        search_start = 0  # cursor in answer_text

        for token_id in answer_token_ids:
            token_str = self._tokenizer.decode(
                [token_id], skip_special_tokens=True,
            )

            if not token_str:
                # Special token or empty decode — no character span
                char_offsets.append((-1, -1))
                continue

            # Find this token string in the answer starting from search_start.
            # We try exact match first; if not found, skip this token.
            pos = answer_text.find(token_str, search_start)
            if pos == -1:
                # Token string not found — may be a chat-template artifact
                # (e.g. end-of-turn tokens). Mark as unmappable.
                char_offsets.append((-1, -1))
            else:
                end_pos = pos + len(token_str)
                char_offsets.append((pos, end_pos))
                search_start = end_pos

        return char_offsets

    # ------------------------------------------------------------------
    # Post-processing: token scores → hallucination spans
    # ------------------------------------------------------------------

    def _scores_to_spans(
        self,
        answer_text: str,
        token_probs: List[float],
        char_offsets: List[Tuple[int, int]],
    ) -> List[HallucinationSpan]:
        """
        Convert per-token hallucination probabilities into contiguous spans.

        Algorithm:
        1. Threshold token probabilities to get binary hallucination flags.
        2. Merge adjacent hallucinated tokens (with gap <= ``merge_gap``).
        3. Map each merged group to character offsets in the answer text.
        4. Return a ``HallucinationSpan`` for each group.

        Parameters
        ----------
        answer_text : str
            The original answer text.
        token_probs : List[float]
            Per-token hallucination probabilities.
        char_offsets : List[Tuple[int, int]]
            Character offsets for each token (from ``_get_answer_token_char_offsets``).

        Returns
        -------
        List[HallucinationSpan]
            Detected hallucination spans, sorted by start offset.
        """
        if not token_probs:
            return []

        n = len(token_probs)
        # Align lengths in case of mismatch (e.g. trailing special tokens)
        n = min(n, len(char_offsets))

        # Step 1: identify hallucinated token indices
        hallucinated = [
            i for i in range(n)
            if token_probs[i] >= self.threshold and char_offsets[i] != (-1, -1)
        ]

        if not hallucinated:
            return []

        # Step 2: merge adjacent hallucinated tokens with gap <= merge_gap
        groups: List[List[int]] = []
        current_group = [hallucinated[0]]

        for idx in hallucinated[1:]:
            # Gap = number of non-hallucinated tokens between current group end and idx
            gap = idx - current_group[-1] - 1
            if gap <= self.merge_gap:
                current_group.append(idx)
            else:
                groups.append(current_group)
                current_group = [idx]
        groups.append(current_group)

        # Step 3: convert each group to a HallucinationSpan
        spans: List[HallucinationSpan] = []
        for group in groups:
            # Collect all token indices in the group (including gap tokens)
            all_indices = list(range(group[0], group[-1] + 1))

            # Find valid char offsets within the group
            valid_offsets = [
                char_offsets[i] for i in all_indices
                if char_offsets[i] != (-1, -1)
            ]
            if not valid_offsets:
                continue

            span_start = min(s for s, e in valid_offsets)
            span_end = max(e for s, e in valid_offsets)

            # Extract verbatim text from the answer
            span_text = answer_text[span_start:span_end]

            # Collect token scores for the group (hallucinated tokens only)
            group_token_scores = [token_probs[i] for i in group]
            span_score = max(group_token_scores)

            spans.append(
                HallucinationSpan(
                    text=span_text,
                    start_offset=span_start,
                    end_offset=span_end,
                    score=span_score,
                    token_scores=group_token_scores,
                )
            )

        return sorted(spans, key=lambda s: s.start_offset)
