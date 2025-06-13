import logging
from typing import Tuple

import astropy.units as u
import numpy as np
from astropy.coordinates import ICRS, SkyCoord

logger = logging.getLogger(__name__)


def filter_by_redshift(
    coords: SkyCoord,
    redshift: np.ndarray,
    weights: np.ndarray,
    z_min: float,
    z_max: float,
) -> Tuple[SkyCoord, np.ndarray, np.ndarray]:
    """
    Filter galaxy sample by redshift range.

    Parameters
    ----------
    coords : SkyCoord
        Galaxy coordinates.
    redshift : np.ndarray
        Galaxy redshifts.
    weights : np.ndarray
        Galaxy weights.
    z_min : float
        Minimum redshift (inclusive).
    z_max : float
        Maximum redshift (inclusive).

    Returns
    -------
    coords_filt : SkyCoord
        Filtered galaxy coordinates.
    z_filt : np.ndarray
        Filtered redshifts.
    w_filt : np.ndarray
        Filtered weights.
    """
    mask = (redshift >= z_min) & (redshift <= z_max)
    total = len(redshift)
    selected = mask.sum()
    logger.info(
        "Filtering redshift: %d/%d galaxies in [%f, %f]", selected, total, z_min, z_max
    )

    # Apply mask
    coords_filt = SkyCoord(
        ra=coords.ra[mask].to(u.degree), dec=coords.dec[mask].to(u.degree), frame=ICRS
    )
    return coords_filt, redshift[mask], weights[mask]


def normalize_redshift(redshift: np.ndarray) -> np.ndarray:
    """
    Normalise redshift values to the [0, 1] interval.

    Parameters
    ----------
    redshift : np.ndarray
        Galaxy redshifts.

    Returns
    -------
    normalised : np.ndarray
        Redshifts scaled to [0, 1].
    """
    z_min = redshift.min()
    z_max = redshift.max()
    if z_max == z_min:
        logger.warning(
            "Redshift range is zero (min == max == %f). Returning zeros.", z_min
        )
        return np.zeros_like(redshift)

    return (redshift - z_min) / (z_max - z_min)
