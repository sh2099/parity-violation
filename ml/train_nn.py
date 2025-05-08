import logging

from hydra import main
from omegaconf import DictConfig

from ml.training import run_training
from ml.utils import pick_device

logging = logging.getLogger(__name__)


@main(config_path="../configs/neural", config_name="config_nn")
def cli(cfg: DictConfig) -> None:
    logging.basicConfig(level=cfg.logging.level)
    logging.info("Starting neural‐net training with config:\n%s", cfg.pretty())

    device = pick_device(
        use_cuda=cfg.hardware.use_cuda,
        use_dml=cfg.hardware.use_dml,
        use_mps=cfg.hardware.use_mps,
    )
    logging.info("Using device %s", device)

    run_training(cfg, device)


if __name__ == "__main__":
    cli()
