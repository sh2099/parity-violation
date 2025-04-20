import time
import numpy as np

# import your two geometry modules
from geometry import (
    rotate_coordinates as rotate_np,
    get_points_in_square as pip_np,
    generate_non_overlapping_square as gnop_np,
)
from geometry_2 import (
    rotate_coordinates as rotate_sh,
    get_points_in_square as pip_sh,
    generate_non_overlapping_square as gnop_sh,
)
from shapely.geometry import Polygon

def benchmark_point_in_poly():
    N_POINTS = 200_000
    # random RA/DEC in [0,1]
    ra = np.random.rand(N_POINTS)
    dec = np.random.rand(N_POINTS)
    # define a simple square polygon centered at (0.5,0.5)
    half = 0.1
    square_coords = np.array([
        [0.5-half, 0.5-half],
        [0.5+half, 0.5-half],
        [0.5+half, 0.5+half],
        [0.5-half, 0.5+half],
    ])
    polygon = Polygon(square_coords)

    # NumPy version
    t0 = time.perf_counter()
    idx_np = pip_np(ra, dec, square_coords)
    dt_np = time.perf_counter() - t0

    # Shapely version
    t0 = time.perf_counter()
    idx_sh = pip_sh(ra, dec, polygon)
    dt_sh = time.perf_counter() - t0

    print(f"Point‑in‑poly (NumPy):   {len(idx_np)} points in {dt_np:.3f}s")
    print(f"Point‑in‑poly (Shapely): {len(idx_sh)} points in {dt_sh:.3f}s\n")


def benchmark_square_generation():
    N_POINTS = 100_000
    N_SQUARES = 100
    ra = np.random.rand(N_POINTS)
    dec = np.random.rand(N_POINTS)
    existing_np = []
    existing_sh = []
    centre = np.array([0.5, 0.5])
    square_size = 0.1
    phi = 0.0

    # NumPy version
    t0 = time.perf_counter()
    for i in range(N_SQUARES):
        sq = gnop_np(ra, dec, centre.copy(), square_size, phi, existing_np)
        existing_np.append(sq)
    dt_np = time.perf_counter() - t0

    # Shapely version
    t0 = time.perf_counter()
    for i in range(N_SQUARES):
        # shapely version expects a Polygon ‘centre’ seed
        # we pass the same centre array but it rebuilds its own square
        poly = gnop_sh(ra, dec, centre.copy(), square_size, phi, existing_sh)
        existing_sh.append(poly)
    dt_sh = time.perf_counter() - t0

    print(f"Square‑gen (NumPy)  x {N_SQUARES}: {dt_np:.3f}s")
    print(f"Square‑gen (Shapely)x {N_SQUARES}: {dt_sh:.3f}s")


if __name__ == "__main__":
    print("\n=== Benchmark: point-in-polygon ===")
    benchmark_point_in_poly()
    print("=== Benchmark: square generation ===")
    benchmark_square_generation()
