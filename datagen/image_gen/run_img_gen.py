import logging
from hydra import main
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from datagen.boss_loader.main import prepare_boss_data
from datagen.image_gen.sampler import random_sampling_images

logger = logging.getLogger(__name__)


@main(config_path="../../configs/datagen", config_name="image_gen")
def run(cfg: DictConfig) -> None:
    """
    Entry point for BOSS image generation.

    Reads FITS via boss_processing, filters by redshift, then
    invokes the sampler to produce a set of non‑overlapping images.
    """
    # set up logging
    logging.basicConfig(level=cfg.logging.level)
    logger.info("Starting image generation pipeline")

    # resolve & load data
    fits_path = to_absolute_path(cfg.data.fits_file)
    coords, z, w = prepare_boss_data(
        fits_file=fits_path,
        sample_size=cfg.data.sample_size,
        random_seed=cfg.data.random_seed,
        z_min=cfg.data.z_range.min,
        z_max=cfg.data.z_range.max,
    )
    logger.info("Loaded %d galaxies after redshift filter", len(z))

    # resolve output directory
    out_dir = to_absolute_path(cfg.images.output_dir)

    # run sampler
    existing, scales = random_sampling_images(
        ra=coords.ra.deg,
        dec=coords.dec.deg,
        redshift=z,
        weights=w,
        num_samples=cfg.images.num_samples,
        square_size=cfg.images.square_size,
        img_size=cfg.images.img_size,
        bw_mode=cfg.images.bw_mode,
        output_dir=out_dir,
        prefix=cfg.images.prefix,
    )

    logger.info(
        "Finished: generated %d images in %s",
        cfg.images.num_samples,
        out_dir,
    )


if __name__ == "__main__":
    run()
