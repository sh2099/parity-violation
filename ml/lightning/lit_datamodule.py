from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ml.data import flip_dataset, get_transforms, load_datasets


class ParityLitDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg  # Hydra config object

    def setup(self) -> None:
        # instantiate train_ds & test_ds
        mean = self.cfg.data.mean
        std = self.cfg.data.std
        size = self.cfg.data.img_size
        tfms = get_transforms(mean, std, size)
        base = Path(self.cfg.data.base_dir)
        self.train_ds, self.test_ds = load_datasets(
            base, self.cfg.data.test_subdir, self.cfg.data.train_subdir, tfms
        )
        if self.cfg.train.flip_augment:
            # convert to list form
            self.train_ds = flip_dataset(self.train_ds)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
        )
