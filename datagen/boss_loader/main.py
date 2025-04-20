import hydra
from omegaconf import DictConfig
import logging

from fits_io import import_data
from data_transforms import filter_by_redshift, normalize_redshift

logger = logging.getLogger(__name__)

@hydra.main(config_path="../../configs/datagen", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Starting BOSS data pipeline…")
    coords, z, w = import_data(
        fits_file=cfg.data.fits_file,
        sample_size=cfg.data.sample_size,
        random_seed=cfg.data.random_seed
    )

    coords, z, w = filter_by_redshift(
        coords, z, w,
        z_min=cfg.data.z_range.min,
        z_max=cfg.data.z_range.max
    )

    #z_norm = normalize_redshift(z)
    logger.info("Finished. %d galaxies in range %s–%s",
                len(z), cfg.data.z_range.min, cfg.data.z_range.max)

    # (… further analysis, saving outputs, etc.)

if __name__ == "__main__":
    main()

