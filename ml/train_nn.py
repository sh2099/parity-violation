import logging

import rich
from hydra import main
from omegaconf import DictConfig

from ml.training import run_training
from ml.utils import dict_to_tree, pick_device

logger = logging.getLogger(__name__)


@main(config_path="../configs/ml", config_name="config")
def cli(cfg: DictConfig) -> None:
    logging.basicConfig(level=cfg.logging.level)
    logger.info("Starting neural‐net training with config:")
    rich.print(dict_to_tree(cfg, guide_style="dim"))

    device = pick_device(
        use_cuda=cfg.hardware.use_cuda,
        use_dml=cfg.hardware.use_dml,
        use_mps=cfg.hardware.use_mps,
    )
    logger.info("Using device %s", device)

    run_training(cfg, device)


if __name__ == "__main__":
    cli()
