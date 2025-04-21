import os
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from datagen.boss_loader.data_transforms import normalize_redshift
from datagen.image_gen.geometry import (
    generate_non_overlapping_square,
    get_points_in_square,
    rotate_coordinates,
)
from datagen.image_gen.rendering import create_image


def random_sampling_images(
    ra: np.ndarray,
    dec: np.ndarray,
    redshift: np.ndarray,
    weights: np.ndarray,
    num_samples: int,
    square_size: float,
    img_size: int,
    bw_mode: bool,
    output_dir: str,
    prefix: str,
    preexisting_squares: List[np.ndarray] = [],
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Sample non‑overlapping rotated squares and render each as an image.

    Parameters
    ----------
    ra : np.ndarray, shape (N,)
        Right ascension of all galaxies.
    dec : np.ndarray, shape (N,)
        Declination of all galaxies.
    redshift : np.ndarray, shape (N,)
        Galaxy redshifts.
    weights : np.ndarray, shape (N,)
        Galaxy weights.
    num_samples : int
        How many squares/images to generate.
    square_size : float
        Side length of each square (in same units as ra/dec).
    img_size : int
        Width & height (in pixels) of output image.
    bw_mode : bool
        If True, render images in black & white.
    output_dir : str
        Directory to save PNGs into.
    prefix : str
        Filename prefix for each image (e.g. “boss” → boss_0.png, …).

    Returns
    -------
    List[np.ndarray]
        The list of square‑vertex arrays used (one per image).
    List[float]
        The list of scale factors returned by `create_image`.
    """
    # normalize redshift once
    norm_z = normalize_redshift(redshift)

    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    existing_squares: List[np.ndarray] = []
    scale_factors: List[float] = []

    for i in tqdm(range(num_samples), desc="Generating images"):
        # 1) pick a random centre & orientation
        idx = np.random.randint(len(ra))
        centre = np.array([ra[idx], dec[idx]])
        phi = np.random.uniform(0, 2 * np.pi)

        # 2) find a non‑overlapping square
        square = generate_non_overlapping_square(
            ra, dec, centre, square_size, phi, preexisting_squares
        )
        existing_squares.append(square)

        # 3) select & rotate points
        sel_idx = get_points_in_square(ra, dec, square)
        pts = np.column_stack((ra[sel_idx], dec[sel_idx]))
        rotated = rotate_coordinates(pts, centre, -phi)

        # 4) render & save
        fname = os.path.join(output_dir, f"{prefix}_{i}.png")
        _, scale = create_image(
            rotated[:, 0],
            rotated[:, 1],
            norm_z[sel_idx],
            weights[sel_idx],
            fname,
            sq_size=square_size,
            img_size=img_size,
            bw=bw_mode,
        )
        scale_factors.append(scale)

    return existing_squares, scale_factors
