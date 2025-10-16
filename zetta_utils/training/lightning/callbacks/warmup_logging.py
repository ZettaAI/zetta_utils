"""
Warmup logging callback that logs every step during the first N steps,
then falls back to the normal log_every_n_steps behavior.
"""

from typing import Any, Optional

import lightning.pytorch as pl


class WarmupLoggingCallback(pl.callbacks.Callback):
    """
    Callback that enables logging every step during warmup period,
    then falls back to normal log_every_n_steps behavior.
    
    Args:
        warmup_steps: Number of steps to log every step (default: 100)
    """
    
    def __init__(self, warmup_steps: int = 100):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.original_log_every_n_steps: Optional[int] = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Store the original settings and configure for warmup."""
        self.original_log_every_n_steps = trainer.log_every_n_steps
    
    def on_train_batch_start(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        batch: Any, 
        batch_idx: int
    ) -> None:
        """Set log_every_n_steps during warmup."""
        if trainer.global_step < self.warmup_steps:
            trainer.log_every_n_steps = 1
        elif trainer.global_step == self.warmup_steps:
            # Restore original settings
            if self.original_log_every_n_steps is not None:
                trainer.log_every_n_steps = self.original_log_every_n_steps
                print(f"[WarmupCallback] Step {trainer.global_step}: restored log_every_n_steps to {self.original_log_every_n_steps}")
