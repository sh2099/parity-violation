import logging
from typing import Tuple

import astropy.units as u
import numpy as np
from astropy.coordinates import ICRS, SkyCoord
from astropy.io import fits

# TODO: Fix decision logic about random and sample size

logger = logging.getLogger(__name__)


def import_data(
    fits_file: str, sample_size: int, random_seed: int | None = None
) -> Tuple[SkyCoord, np.ndarray, np.ndarray]:
    """
    Load galaxy coordinates, redshifts, and weights from a BOSS FITS file.

    Parameters
    ----------
    fits_file : str
        Path to the input FITS file.
    sample_size : int
        Number of objects to randomly sample from the catalog.
    random_seed : int, optional
        Random seed for reproducible sampling.

    Returns
    -------
    coords : SkyCoord
        Sampled galaxy sky coordinates.
    redshift : np.ndarray
        Sampled redshift values.
    weights : np.ndarray
        Computed weights for each sampled galaxy.
    """
    rng = np.random.default_rng(random_seed)
    # Open fits and extract the data table
    with fits.open(fits_file) as hdul:
        table = hdul[1].data
        total = len(table)
        idx = rng.choice(total, sample_size, replace=False)
        subset = table[idx]

    coords = SkyCoord(
        ra=subset["RA"] * u.degree, dec=subset["DEC"] * u.degree, frame=ICRS
    )
    z_vals = np.asarray(subset["Z"])
    w = get_weights(subset)

    logger.info("Imported %d of %d entries from %s", sample_size, total, fits_file)
    return coords, z_vals, w


def get_weights(data: np.ndarray, random: bool = False) -> np.ndarray:
    """
    Compute observational weights from BOSS catalog columns.

    Parameters
    ----------
    data : np.ndarray
        FITS table rows with WEIGHT_* columns.
    random : bool, optional
        If True, return only FKP weights (for random catalogs).

    Returns
    -------
    weights : np.ndarray
        Final weight for each object.
    """
    wfkp = np.asarray(data["WEIGHT_FKP"])
    if random:
        logger.debug("Random catalog: using only FKP weights.")
        return wfkp

    w_seeing = np.asarray(data["WEIGHT_SEEING"])
    w_star = np.asarray(data["WEIGHT_STAR"])
    w_noz = np.asarray(data["WEIGHT_NOZ"])
    w_cp = np.asarray(data["WEIGHT_CP"])

    sys_weight = w_seeing * w_star
    total_weight = wfkp * sys_weight * (w_noz + w_cp - 1)

    logger.debug(
        "Weights shapes: wfkp=%s, sys=%s, total=%s",
        wfkp.shape,
        sys_weight.shape,
        total_weight.shape,
    )
    return total_weight
