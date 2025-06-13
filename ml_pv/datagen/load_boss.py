import logging

import hydra
from omegaconf import DictConfig

from ml_pv.datagen.boss_loader.data_transforms import filter_by_redshift
from ml_pv.datagen.boss_loader.fits_io import import_data


def prepare_boss_data(
    fits_file: str,
    sample_size: int,
    random_seed: int,
    z_min: float,
    z_max: float,
) -> tuple:
    """
    Load, sample, and redshift‐filter BOSS data.
    Returns: coords, redshift, weights
    """
    coords, z, w = import_data(
        fits_file=fits_file,
        sample_size=sample_size,
        random_seed=random_seed,
    )
    coords, z, w = filter_by_redshift(
        coords,
        z,
        w,
        z_min=z_min,
        z_max=z_max,
    )
    return coords, z, w


logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3", config_path="../../configs/datagen", config_name="config"
)
def main(cfg: DictConfig) -> None:
    """
    Test the BOSS data loading and redshift filtering.
    """
    logger.info("Starting BOSS data pipeline…")
    coords, z, w = prepare_boss_data(
        fits_file=cfg.data.fits_file,
        sample_size=cfg.data.sample_size,
        random_seed=cfg.data.random_seed,
        z_min=cfg.data.z_range.min,
        z_max=cfg.data.z_range.max,
    )

    # z_norm = normalize_redshift(z)
    logger.info(
        "Finished. %d galaxies in range %s–%s",
        len(z),
        cfg.data.z_range.min,
        cfg.data.z_range.max,
    )


if __name__ == "__main__":
    main()
