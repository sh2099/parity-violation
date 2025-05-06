import logging

from hydra import main
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from datagen.boss_loader.main import prepare_boss_data
from datagen.image_gen.rendering import display_sample_dist, get_unique_dir
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
    requested = to_absolute_path(cfg.images.output_dir)
    out_dir = get_unique_dir(requested)
    logger.info("Output directory: %s", out_dir)

    # run sampler
    testing, test_scales, avg_n_test = random_sampling_images(
        ra=coords.ra.deg,
        dec=coords.dec.deg,
        redshift=z,
        weights=w,
        num_samples=cfg.images.num_test_samples,
        square_size=cfg.images.square_size,
        img_size=cfg.images.img_size,
        bw_mode=cfg.images.bw_mode,
        output_dir=out_dir,
        prefix=f"{cfg.images.prefix}_test_",
    )

    logger.info(
        "Finished: generated %d testing images in %s," "\n avg %d points/image",
        cfg.images.num_test_samples,
        out_dir,
        avg_n_test,
    )

    training, train_scales, avg_n_train = random_sampling_images(
        ra=coords.ra.deg,
        dec=coords.dec.deg,
        redshift=z,
        weights=w,
        num_samples=cfg.images.num_train_samples,
        square_size=cfg.images.square_size,
        img_size=cfg.images.img_size,
        bw_mode=cfg.images.bw_mode,
        output_dir=out_dir,
        prefix=f"{cfg.images.prefix}_train_",
        preexisting_squares=testing,
    )
    logger.info(
        "Finished: generated %d training images in %s," "\n avg %d points/image",
        cfg.images.num_train_samples,
        out_dir,
        avg_n_train,
    )

    if cfg.images.viz.enable:
        viz_out = to_absolute_path(cfg.images.viz.output_file)
        display_sample_dist(
            ra=coords.ra.deg,
            dec=coords.dec.deg,
            train_squares=training,
            test_squares=testing,
            output_path=viz_out,
        )
        logger.info("Saved sample‐distribution plot to %s", viz_out)


if __name__ == "__main__":
    run()
