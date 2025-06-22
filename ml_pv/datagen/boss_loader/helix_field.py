"""
Create a field of helices with a given number of points.
This can be used to generate parity violating datasets based on the
relative % of each handedness.
"""

import numpy as np
import plotly.graph_objects as go
from scipy.stats import special_ortho_group


def generate_helices(
    radius: float = 0.03,
    pitch: float = 0.1,
    num_points: int = 25,
    length: float = 0.3,
    num_helices: int = 200,
    parity_split: float = 0.5,
) -> np.ndarray:
    """
    Generate a field of helices with specified parameters.
    Parameters
    ----------
    radius : float
        Radius of the helices.
    pitch : float
        Pitch of the helices, which determines the vertical distance between turns.
    num_points : int
        Number of points in each helix.
    length : float
        Length of each helix, which determines how many turns the helix will have.
    num_helices : int
        Number of helices to generate.
    parity_split : float
        Fraction of helices that will be right-handed (0) vs left-handed (1).
    """
    # Use parameters to generate a helix
    t = np.linspace(-np.pi * length / pitch, np.pi * length / pitch, num_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = (pitch / (2 * np.pi)) * t

    # Combine x, y, z into a single array of points
    helix_points = np.vstack((x, y, z)).T

    # Generate random initial points for each helix
    initial_xy = np.random.rand(num_helices, 2) * 10  # Shape: (num_helices, 3)
    initial_z = np.random.rand(num_helices, 1) * 10
    initial_points = np.hstack((initial_xy, initial_z))  # Shape: (num_helices, 3)
    # Generate random rotation matrices for each helix
    rotation_matrices = np.array(
        [special_ortho_group.rvs(3) for _ in range(num_helices)]
    )  # Shape: (num_helices, 3, 3)

    handedness = np.random.choice(
        [0, 1], size=num_helices, p=[parity_split, 1 - parity_split]
    )  # Shape: (num_helices,)
    # Modify helix points based on handedness
    helix_points_modified = np.array(
        [helix_points if h == 0 else np.vstack((x, -y, z)).T for h in handedness]
    )

    # Apply the rotation matrices
    all_helix_points = np.einsum(
        "nij,nkj->nki", rotation_matrices, helix_points_modified
    )  # Shape: (num_helices, n, 3)
    all_helix_points += initial_points[:, np.newaxis, :]
    return all_helix_points


def plot_helices_3d(helices: np.ndarray) -> None:
    fig = go.Figure()
    for helix in helices:
        fig.add_trace(
            go.Scatter3d(
                x=helix[:, 0],
                y=helix[:, 1],
                z=helix[:, 2],
                mode="lines",
                line=dict(width=2),
                opacity=0.5,
            )
        )
    fig.update_layout(
        title="3D Helices from BOSS Data",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        width=800,
        height=800,
    )
    fig.show()
