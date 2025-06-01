import logging
from typing import List

import numpy as np
from shapely import affinity, vectorized
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


def rotate_coordinates(
    coords: np.ndarray, centre: np.ndarray, phi: float
) -> np.ndarray:
    """
    Rotate an array of 2D points by angle phi around a centre.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 2)
        Points to rotate.
    centre : np.ndarray, shape (2,)
        Rotation origin as [x, y].
    phi : float
        Angle in radians (CCW positive).

    Returns
    -------
    np.ndarray, shape (N, 2)
        Rotated points.
    """
    # shift to origin
    shifted = coords - centre
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s], [s, c]])
    rotated = shifted @ R.T
    return rotated + centre


def get_points_in_square(
    ra: np.ndarray, dec: np.ndarray, square: Polygon
) -> np.ndarray:
    """
    Vectorized selection of points inside a shapely Polygon.

    Parameters
    ----------
    ra, dec : np.ndarray, shape (N,)
        Arrays of point coordinates.
    square : Polygon
        A shapely Polygon object representing the square.

    Returns
    -------
    np.ndarray
        Indices of points inside the square.
    """
    mask = vectorized.contains(square, ra, dec)
    return np.nonzero(mask)[0]


def generate_non_overlapping_square(
    ra: np.ndarray,
    dec: np.ndarray,
    centre: np.ndarray,
    square_size: float,
    phi: float,
    existing: List[Polygon],
) -> Polygon:
    """
    Create a rotated square polygon that does not intersect any in `existing`.

    Parameters
    ----------
    ra, dec : np.ndarray, shape (M,)
        Coordinates for sampling new centres.
    centre : np.ndarray, shape (2,)
        Initial [x, y] centre for the square.
    square_size : float
        Side length of the square.
    phi : float
        Rotation angle in radians.
    existing : list of Polygon
        Previously placed squares to avoid.

    Returns
    -------
    Polygon
        A shapely Polygon for the non-overlapping square.
    """
    half = square_size / 2

    # define axis-aligned square
    def make_square(c: np.ndarray, angle: float) -> Polygon:
        base = Polygon(
            [
                (c[0] - half, c[1] - half),
                (c[0] + half, c[1] - half),
                (c[0] + half, c[1] + half),
                (c[0] - half, c[1] + half),
            ]
        )
        return affinity.rotate(base, angle, origin=(c[0], c[1]))

    attempt = 0
    square = make_square(centre, phi)
    while any(square.intersects(ex) for ex in existing):
        attempt += 1
        if attempt % 50 == 0:
            logger.debug(
                "Still searching non-overlapping square after %d attempts", attempt
            )
        # choose new random centre and orientation
        idx = np.random.randint(len(ra))
        centre = np.array([ra[idx], dec[idx]])
        phi = np.random.uniform(0, 2 * np.pi)
        square = make_square(centre, phi)
    return square
