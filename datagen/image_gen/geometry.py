import logging
from typing import List, Tuple

import numpy as np
from numpy.typing import ArrayLike
from matplotlib.path import Path

logger = logging.getLogger(__name__)


def rotate_coordinates(
    coords: np.ndarray,
    centre: np.ndarray,
    phi: float
) -> np.ndarray:
    """
    Rotate points by angle phi around centre.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 2)
        Points to rotate.
    centre : np.ndarray, shape (2,)
        Rotation center.
    phi : float
        Rotation angle in radians (CCW positive).

    Returns
    -------
    np.ndarray, shape (N, 2)
        Rotated points.
    """
    # Shift to origin
    shifted = coords - centre
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s], [s, c]])
    rotated = shifted @ R.T
    return rotated + centre


def compute_overlap(
    poly1: np.ndarray,
    poly2: np.ndarray
) -> bool:
    """
    Quick AABB (axis‑aligned) overlap check for two convex quads.

    Parameters
    ----------
    poly1, poly2 : np.ndarray, shape (4,2)
        Vertices of each square.

    Returns
    -------
    bool
        True if their bounding boxes overlap.
    """
    min1, max1 = poly1.min(axis=0), poly1.max(axis=0)
    min2, max2 = poly2.min(axis=0), poly2.max(axis=0)
    overlap_x = (max1[0] >= min2[0]) and (max2[0] >= min1[0])
    overlap_y = (max1[1] >= min2[1]) and (max2[1] >= min1[1])
    return overlap_x and overlap_y


def get_points_in_square(
    ra: ArrayLike,
    dec: ArrayLike,
    square: np.ndarray
) -> np.ndarray:
    """
    Return indices of points inside a polygon.

    Parameters
    ----------
    ra, dec : array‑like of shape (N,)
        Point coordinates.
    square : np.ndarray, shape (4,2)
        Polygon vertices.

    Returns
    -------
    np.ndarray
        Indices of points inside the square.
    """
    pts = np.column_stack((ra, dec))
    path = Path(square)
    mask = path.contains_points(pts)
    return np.nonzero(mask)[0]


def generate_non_overlapping_square(
    ra: np.ndarray,
    dec: np.ndarray,
    centre: np.ndarray,
    square_size: float,
    phi: float,
    existing: List[np.ndarray]
) -> np.ndarray:
    """
    Randomly place a rotated square that doesn’t overlap any in `existing`.

    Parameters
    ----------
    ra, dec : np.ndarray, shape (M,)
        All point coords, for selecting new centres.
    centre : np.ndarray, shape (1,2)
        Initial guess for square centre.
    square_size : float
    phi : float
    existing : list of np.ndarray
        Previously placed square vertices.

    Returns
    -------
    np.ndarray, shape (4,2)
        Vertices of a non‑overlapping square.
    """
    half = square_size / 2
    verts_base = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half

    attempt = 0
    while True:
        attempt += 1
        # build square
        square = rotate_coordinates(verts_base + centre, centre, phi)
        # fast AABB check
        if not any(compute_overlap(square, e) for e in existing):
            return square

        if attempt % 50 == 0:
            logger.debug("Still searching for non‑overlap after %d tries", attempt)
        # pick a new random centre & phi
        idx = np.random.randint(len(ra))
        centre[:] = [ra[idx], dec[idx]]
        phi = np.random.uniform(0, 2*np.pi)
