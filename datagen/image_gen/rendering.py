import logging
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

logger = logging.getLogger(__name__)


def compute_color(z: float) -> np.ndarray:
    """
    Map a normalized redshift value to an RGB color.

    Parameters
    ----------
    z : float
        Normalized redshift in [0, 1].

    Returns
    -------
    np.ndarray, shape (3,)
        RGB color with values in [0, 1].
    """
    # Linear gradient from blue (z=0) to red (z=1)
    return np.array([z, 0.0, 1.0 - z])


def create_image(
    ra: np.ndarray,
    dec: np.ndarray,
    redshifts: np.ndarray,
    weights: np.ndarray,
    filename: str,
    sq_size: float,
    img_size: int = 64,
    bw: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Render a weighted point cloud into a pseudo-RGB image and save to disk.

    Parameters
    ----------
    ra : np.ndarray, shape (N,)
        X-coordinates of points within a square.
    dec : np.ndarray, shape (N,)
        Y-coordinates of points within a square.
    redshifts : np.ndarray, shape (N,)
        Redshift values (will be normalized for coloring).
    weights : np.ndarray, shape (N,)
        Weights for each point (brightness scaling).
    filename : str
        Path (including basename) where the PNG will be saved.
    sq_size : float
        Side length of the square in same units as ra/dec.
    img_size : int, optional
        Pixel dimensions of the output image (square). Default is 64.
    bw : bool, optional
        If True, ignore redshift coloring and render in grayscale.

    Returns
    -------
    image : np.ndarray, shape (img_size, img_size, 3)
        The rendered image array (uint8).
    scale_factor : float
        Factor used to scale float RGB values to 0-255.
    """
    # Normalize positions to [0, 1] in square
    # TODO: Fix this - technically diff scaling for squares
    norm_ra = (ra - ra.min()) / sq_size
    norm_dec = (dec - dec.min()) / sq_size
    norm_ra = np.clip(norm_ra, 0.0, 1.0)
    norm_dec = np.clip(norm_dec, 0.0, 1.0)

    # Normalize redshift to [0, 1]
    if not bw:
        z_min, z_max = redshifts.min(), redshifts.max()
        if z_max > z_min:
            norm_z = (redshifts - z_min) / (z_max - z_min)
        else:
            norm_z = np.zeros_like(redshifts)
    else:
        norm_z = np.ones_like(redshifts)

    # Map to pixel indices
    x_pix = (norm_ra * (img_size - 1)).astype(int)
    y_pix = (norm_dec * (img_size - 1)).astype(int)

    # Initialize image buffer
    buf = np.zeros((img_size, img_size, 3), dtype=float)

    # Accumulate weighted colors
    for x, y, z_val, w in zip(x_pix, y_pix, norm_z, weights):
        color = compute_color(z_val)
        buf[y, x] += w * color

    # Scale to uint8
    max_val = buf.max()
    if max_val > 0:
        scale_factor = 255.0 / max_val
        img_uint8 = np.clip(buf * scale_factor, 0, 255).astype(np.uint8)
    else:
        scale_factor = 1.0
        img_uint8 = buf.astype(np.uint8)
        logger.warning("Empty image buffer, saved blank image: %s", filename)

    # Render and save
    fig, ax = plt.subplots(figsize=(img_size, img_size), dpi=1)
    ax.imshow(img_uint8, interpolation="nearest")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return img_uint8, scale_factor


def display_sample_dist(
    ra: Union[np.ndarray, List[float]],
    dec: Union[np.ndarray, List[float]],
    train_squares: List[np.ndarray],
    test_squares: List[np.ndarray],
    output_path: str,
    figsize: tuple = (9, 6),
    dpi: int = 100,
) -> None:
    """
    Scatter all (ra, dec) points and overlay two sets of squares.

    Parameters
    ----------
    ra : array‑like of shape (N,)
        Right ascensions.
    dec : array‑like of shape (N,)
        Declinations.
    train_squares : list of (4,2) ndarrays
        List of vertex arrays for “train” squares (plotted in red).
    test_squares : list of (4,2) ndarrays
        List of vertex arrays for “test” squares (plotted in black).
    output_path : str
        Where to save the PNG (will auto‑create dirs if needed).
    figsize : tuple
        Figure size in inches.
    dpi : int
        Resolution of output PNG.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot all data points
    ax.scatter(ra, dec, color="green", s=0.01, marker="o", label="Galaxies")

    # Overlay train squares in red
    for sq in train_squares:
        poly = MplPolygon(sq, edgecolor="r", fill=False, linewidth=1)
        ax.add_patch(poly)
    # Overlay test squares in black
    for sq in test_squares:
        poly = MplPolygon(sq, edgecolor="k", fill=False, linewidth=1)
        ax.add_patch(poly)

    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("DEC (deg)")
    ax.legend(loc="upper right", fontsize="small")

    # Ensure output directory exists
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
