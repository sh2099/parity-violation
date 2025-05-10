# scripts/train_lit.py

import logging

import rich
from hydra import main
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from ml.lightning.lit_datamodule import ParityLitDataModule
from ml.lightning.lit_module import ParityLitModule
from ml.utils import dict_to_tree, pick_device


@main(config_path="../configs/ml", config_name="config_lit")
def run_lit(cfg: DictConfig) -> None:
    # 1) set up Python logging before Lightning does its own
    logging.basicConfig(
        level=cfg.logging.level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Lightning training with config:")
    rich.print(dict_to_tree(cfg, guide_style="dim"))

    # 2) Device selection
    device = pick_device(
        use_cuda=cfg.hardware.use_cuda,
        use_dml=cfg.hardware.use_dml,
        use_mps=cfg.hardware.use_mps,
    )
    logger.info("Using device: %s", device)

    # 3) Prepare data and model
    dm = ParityLitDataModule(cfg)
    model = ParityLitModule(cfg.train.lr, cfg.train.momentum)

    # 4) Callbacks for checkpointing & early stopping
    ckpt = ModelCheckpoint(
        dirpath=cfg.train.checkpoint_dir,
        monitor="avg_val_pos_frac",
        mode="max",
        save_top_k=3,
    )
    stop = EarlyStopping(
        monitor="avg_val_pos_frac",
        mode="max",
        patience=cfg.train.early_stop_patience,
    )

    # 5) Instantiate Trainer
    trainer = Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        precision=cfg.hardware.precision,
        callbacks=[ckpt, stop],
        log_every_n_steps=cfg.train.log_every_n_steps,
        enable_progress_bar=True,
    )

    # 6) Launch training
    trainer.fit(model, dm)

    logger.info("Training complete. Best model saved at: %s", ckpt.best_model_path)


if __name__ == "__main__":
    run_lit()
