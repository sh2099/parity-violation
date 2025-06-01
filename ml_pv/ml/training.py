import logging
from pathlib import Path

import torch
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ml_pv.ml.data import get_transforms, load_datasets, make_loaders
from ml_pv.ml.models import ParityNet

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: ParityNet,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch; return average loss."""
    model.train()
    total_loss, count = 0.0, 0
    for xb, _ in loader:
        xb = xb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = -(out.mean() / out.std())
        loss.backward()
        optimizer.step()
        total_loss += (-loss).item()
        count += 1
    logger.info("  Avg train loss: %.4f", total_loss / count)
    return total_loss / count


def evaluate(
    model: ParityNet,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute positive‐fraction metric on loader."""
    model.eval()
    with torch.no_grad():
        out_all = []
        for xb, _ in loader:
            xb = xb.to(device)
            out_all.append(model(xb))
        out = torch.cat(out_all)
    frac_pos = (out > 0).float().mean().item()
    logger.info("  Positive fraction: %.5f", frac_pos)
    return frac_pos


def run_training(
    cfg: DictConfig,  # DictConfig with fields below
    device: torch.device,
) -> None:
    logger.info("Using device %s", device)

    # data
    train_ds, test_ds = load_datasets(
        Path(cfg.data.base_dir),
        cfg.data.test_subdir,
        cfg.data.train_subdir,
        get_transforms(cfg.data.mean, cfg.data.std, cfg.data.img_size),
    )
    train_loader, test_loader = make_loaders(
        train_ds,
        test_ds,
        cfg.train.batch_size,
        cfg.train.num_workers,
        flip_aug=cfg.train.flip_augment,
    )

    # model & optimizer
    model = ParityNet().to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum
    )

    # training loop
    for epoch in range(cfg.train.epochs):
        logger.info("Epoch %d/%d", epoch + 1, cfg.train.epochs)
        train_one_epoch(model, train_loader, optimizer, device)
        evaluate(model, test_loader, device)

        # checkpoint
        ckpt_dir = Path(cfg.train.checkpoint_dir) / f"epoch_{epoch+1}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / "model.pth")
