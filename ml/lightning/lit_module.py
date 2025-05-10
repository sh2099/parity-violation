# neural/lit/lit_module.py

from typing import List

import pytorch_lightning as pl
import torch
from torch.optim import SGD

from ml.models import ParityNet


class ParityLitModule(pl.LightningModule):
    def __init__(self, lr: float, momentum: float) -> None:
        super().__init__()
        self.save_hyperparameters()  # saves lr & momentum
        self.model = ParityNet()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, _ = batch
        y = self(x)
        loss = -(y.mean() / y.std())
        # Log loss to the progress bar, aggregated per epoch
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_epoch_end(self, outputs: List[dict]) -> None:
        # outputs is a list of dicts returned from training_step
        avg_loss = torch.stack([out["loss"] for out in outputs]).mean()
        # This will also show up in the progress bar
        self.log("avg_train_loss", avg_loss, prog_bar=True)

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> dict:
        x, _ = batch
        y = self(x)
        frac_pos = (y > 0).float().mean()
        # Log positive fraction per epoch
        self.log("val_pos_frac", frac_pos, prog_bar=True, on_step=False, on_epoch=True)
        return {"val_frac": frac_pos}

    def validation_epoch_end(self, outputs: List[dict]) -> None:
        # Compute epoch‐level average positive fraction
        avg_frac = torch.stack([out["val_frac"] for out in outputs]).mean()
        self.log("avg_val_pos_frac", avg_frac, prog_bar=True)

    def configure_optimizers(self) -> SGD:
        return SGD(
            self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum
        )
