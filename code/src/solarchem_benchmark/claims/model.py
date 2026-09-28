"""Decision model behind the claim pipeline: Laya (convaiinnovations/laya).

Laya is a non-generative encoder (ModernBERT / mmBERT) that answers typed
questions -- ``choice``, ``score``, ``noul`` -- over a state in one forward pass
and returns a probability per option. Options are defined per request, which
is what lets one question list a table's own row and column labels.

Two behaviours from the model card shape how it is called here:

* ``noul`` can follow its ``false``/``true`` labels instead of the state on
  the English checkpoint, so every yes/no is asked as a two-option ``choice``
  with neutral keys ``A``/``B``.
* The state is right-truncated at ``max_len - head_max_len`` tokens, so the
  caller puts the sentence before the table: truncation drops table rows,
  never the claim.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

logger = logging.getLogger(__name__)

LAYA_REPO = "convaiinnovations/laya"

#: checkpoint -> (subfolder, default max_len, default head_max_len)
CHECKPOINTS: dict[str, tuple[str | None, int, int]] = {
    "english": (None, 512, 192),
    "multilingual": ("multilingual", 2048, 320),
    "typed-decisions": ("typed-decisions", 1024, 256),
}

Answers = dict[str, dict[str, Any]]


def _disable_cudnn_attention(device: str | None) -> None:
    """Skip the cuDNN attention kernel on CUDA.

    On an H100 MIG slice that kernel builds no execution plan and the forward
    pass dies. Flash and memory-efficient attention stay enabled.
    """
    if device is not None and device != "cuda":
        return
    import torch

    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)


class DecisionModel(Protocol):
    """Anything that answers the same questions over a batch of states."""

    name: str

    def describe(self) -> dict[str, Any]: ...

    def predict_batch(self, states: list[str], questions: dict[str, dict]) -> list[Answers]:
        """Return, per state, ``{question_id: {"choice": key, "probabilities": {...}}}``."""
        ...


class UniformModel:
    """Stand-in that gives every option the same probability (``--dry-run``)."""

    name = "uniform"

    def describe(self) -> dict[str, Any]:
        return {"model": self.name}

    def predict_batch(self, states: list[str], questions: dict[str, dict]) -> list[Answers]:
        answers: Answers = {}
        for qid, question in questions.items():
            keys = list(question["criteria"])
            answers[qid] = {
                "type": "choice",
                "choice": keys[0],
                "probabilities": {key: round(1 / len(keys), 4) for key in keys},
            }
        return [answers for _ in states]


class LayaModel:
    """Local Laya checkpoint, loaded once and reused for every document."""

    def __init__(
        self,
        checkpoint: str = "multilingual",
        *,
        model_path: str | None = None,
        device: str | None = None,
        max_len: int | None = None,
        head_max_len: int | None = None,
        batch_size: int = 32,
        fast: bool = False,
    ) -> None:
        if model_path is None and checkpoint not in CHECKPOINTS:
            raise ValueError(f"Unknown Laya checkpoint {checkpoint!r}; choose from {sorted(CHECKPOINTS)}")
        subfolder, default_max, default_head = CHECKPOINTS.get(checkpoint, (None, 1024, 256))
        self.checkpoint = checkpoint
        self.model_path = model_path
        self.subfolder = None if model_path else subfolder
        self.device = device
        self.max_len = max_len or default_max
        self.head_max_len = head_max_len or default_head
        self.batch_size = batch_size
        self.fast = fast
        self.name = f"laya-{checkpoint}" if model_path is None else f"laya-local:{model_path}"
        self._agent = None

    def load(self) -> None:
        if self._agent is not None:
            return
        # transformers probes for TensorFlow at import; with TF installed its
        # abseil runtime can deadlock model construction (Laya model card).
        os.environ.setdefault("USE_TF", "0")
        try:
            import laya
        except ImportError as exc:  # pragma: no cover - depends on the server env
            raise ImportError("Laya is not installed: pip install -U laya") from exc

        _disable_cudnn_attention(self.device)
        source = self.model_path or LAYA_REPO
        logger.info("Loading Laya %s (subfolder=%s, device=%s)", source, self.subfolder, self.device)
        self._agent = laya.load(source, device=self.device, subfolder=self.subfolder, fast=self.fast)

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "repo": self.model_path or LAYA_REPO,
            "subfolder": self.subfolder,
            "max_len": self.max_len,
            "head_max_len": self.head_max_len,
            "batch_size": self.batch_size,
            "device": self.device,
        }

    def predict_batch(self, states: list[str], questions: dict[str, dict]) -> list[Answers]:
        if not states:
            return []
        self.load()
        results = self._agent.predict_batch(
            states,
            questions,
            batch_size=self.batch_size,
            max_len=self.max_len,
            head_max_len=self.head_max_len,
        )
        return [result["answers"] for result in results]
