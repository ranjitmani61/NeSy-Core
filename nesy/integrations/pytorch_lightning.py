"""
nesy/integrations/pytorch_lightning.py
=========================================
PyTorch Lightning training loop for NeSy-Core neural backbone.

Handles EWC penalty integration automatically during training.
The module wraps any NeSyBackbone into a ``LightningModule`` that:
    1. Runs forward pass through the backbone
    2. Computes task-specific loss
    3. Adds EWC regularisation penalty from ``ContinualLearner``
    4. Logs combined loss for monitoring

Mathematical basis (EWC — Kirkpatrick et al., 2017):
    L_total = L_new(theta) + (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2

Requires: ``pip install pytorch-lightning torch``

Usage:
    from nesy.integrations.pytorch_lightning import NeSyLightningModule

    module = NeSyLightningModule.build(backbone, learner, loss_fn)
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(module, datamodule)
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, Callable, Tuple


logger = logging.getLogger(__name__)

# Check availability at import time for informational purposes
PL_AVAILABLE = (
    importlib.util.find_spec("pytorch_lightning") is not None
    and importlib.util.find_spec("torch") is not None
)
if not PL_AVAILABLE:  # pragma: no cover
    logger.info(
        "pytorch-lightning/torch not installed. NeSyLightningModule.build() will raise ImportError."
    )


class NeSyLightningModule:
    """Factory for building a PyTorch Lightning module with EWC integration.

    This class provides a ``build()`` static method that returns a fully
    configured ``LightningModule``.  The separation via a factory avoids
    a hard import-time dependency on ``pytorch_lightning``.

    The returned module:
        - Runs forward pass through any NeSyBackbone's ``encode()`` method.
        - Computes task loss via the user-provided ``loss_fn``.
        - Adds EWC penalty from ``ContinualLearner.ewc_penalty()``
          on single-element parameters (scalars serialised to dict).
        - Logs ``train_loss`` every training step.

    Mathematical basis (Kirkpatrick et al., 2017):
        L_total(θ) = L_new(θ) + (λ/2) × Σᵢ Fᵢ × (θᵢ − θ*ᵢ)²

    Usage:
        module = NeSyLightningModule.build(backbone, learner, loss_fn)
        trainer = pl.Trainer(max_epochs=10)
        trainer.fit(module, datamodule)

    Raises:
        ImportError: if pytorch-lightning or torch is not installed.
    """

    @staticmethod
    def is_available() -> bool:
        """Check if PyTorch Lightning and torch are installed.

        Returns:
            True if both packages are importable.
        """
        return PL_AVAILABLE

    @staticmethod
    def build(
        backbone: Any,
        learner: Any,
        loss_fn: Callable,
        learning_rate: float = 1e-4,
    ) -> Any:
        """Build a LightningModule wrapping the NeSy backbone.

        Args:
            backbone:      NeSyBackbone (or any nn.Module with encode()).
            learner:       ContinualLearner for EWC penalty computation.
            loss_fn:       Callable(output, target) → scalar loss.
            learning_rate: Adam optimiser learning rate (default 1e-4).

        Returns:
            A configured ``pytorch_lightning.LightningModule`` instance.

        Raises:
            ImportError: if pytorch-lightning or torch are missing.
        """
        try:
            import pytorch_lightning as pl
            import torch
        except ImportError:
            raise ImportError(
                "NeSyLightningModule requires pytorch-lightning and torch. "
                "Install with: pip install pytorch-lightning torch"
            )

        class _NeSyModule(pl.LightningModule):
            """Internal LightningModule wrapping NeSy backbone + EWC."""

            def __init__(
                self,
                backbone_: Any,
                learner_: Any,
                loss_fn_: Callable,
                lr: float,
            ) -> None:
                super().__init__()
                self._backbone = backbone_
                self._learner = learner_
                self._loss_fn = loss_fn_
                self._lr = lr

            def forward(self, x: Any) -> Any:
                """Forward pass through backbone's encode()."""
                return self._backbone.encode(x)

            def training_step(self, batch: Tuple[Any, Any], batch_idx: int) -> Any:
                """Training step: task loss + EWC penalty.

                Mathematical basis (Kirkpatrick et al., 2017):
                    L_total = L_task + L_ewc
                    L_ewc = (λ/2) × Σᵢ Fᵢ × (θᵢ − θ*ᵢ)²
                """
                x, y = batch
                output = self._backbone.encode(x)
                task_loss = self._loss_fn(output, y)

                # EWC penalty — framework-agnostic dict of scalars
                ewc_penalty = 0.0
                if hasattr(self._backbone, "state_dict"):
                    params = {}
                    for k, v in self._backbone.state_dict().items():
                        if v.numel() == 1:
                            params[k] = v.item()
                    if params:
                        ewc_penalty = self._learner.ewc_penalty(params)

                total = task_loss + ewc_penalty
                self.log("train_loss", total)
                self.log("task_loss", task_loss)
                self.log("ewc_penalty", float(ewc_penalty))
                return total

            def configure_optimizers(self) -> Any:
                """Adam optimiser with configurable learning rate."""
                return torch.optim.Adam(self.parameters(), lr=self._lr)

        module = _NeSyModule(backbone, learner, loss_fn, learning_rate)
        logger.info(
            "Built NeSyLightningModule (lr=%s, backbone=%s)",
            learning_rate,
            type(backbone).__name__,
        )
        return module
