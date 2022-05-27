"""ZettaAI Python plotting utilities."""
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt # type: ignore

import ztutils as zu


def render_vectors(
    vec: zu.zt_types.Array,
    grid_size: int = 50,
    alpha: float = 0.6,
    linewidth: float = 0.5,
    figsize: tuple[int, int] = (10, 10),
) -> npt.NDArray:
    """
    Render the given vector field to an RGB numpy image.

    Args:
        vec (zu.zt_types.Array): An array of shape [(1,) H, W, 2] representing
            a 2D vector field.
        grid_side (int): Number of arrows per side.

    Returns:
        npt.NDArray: RGB rendering of the vector field.
    """
    vec = zu.zt_types.to_np(vec).squeeze()
    if vec.shape[0] == 2:
        vec = np.transpose(vec, (1, 2, 0))

    assert vec.shape[0] == vec.shape[1]

    n = vec.shape[0]
    y, x = np.mgrid[0:n, 0:n]

    x_coords = [0, vec.shape[0]]
    y_coords = [0, vec.shape[1]]

    ex = (1) * vec[:, :, 0]
    ey = vec[:, :, 1]
    r = np.arctan2(ex, ey)

    interval = (x_coords[1] - x_coords[0]) // grid_size

    plt.figure(figsize=figsize)
    plt.quiver(
        x[x_coords[0] : x_coords[1] : interval, y_coords[0] : y_coords[1] : interval],
        y[x_coords[0] : x_coords[1] : interval, y_coords[0] : y_coords[1] : interval],
        ex[x_coords[0] : x_coords[1] : interval, y_coords[0] : y_coords[1] : interval],
        ey[x_coords[0] : x_coords[1] : interval, y_coords[0] : y_coords[1] : interval],
        r[x_coords[0] : x_coords[1] : interval, y_coords[0] : y_coords[1] : interval],
        alpha=alpha,
    )
    plt.quiver(
        x[x_coords[0] : x_coords[1] : interval, y_coords[0] : y_coords[1] : interval],
        y[x_coords[0] : x_coords[1] : interval, y_coords[0] : y_coords[1] : interval],
        ex[x_coords[0] : x_coords[1] : interval, y_coords[0] : y_coords[1] : interval],
        ey[x_coords[0] : x_coords[1] : interval, y_coords[0] : y_coords[1] : interval],
        edgecolor="k",
        facecolor="None",
        linewidth=linewidth,
    )
    plt.gca().invert_yaxis()

    return np.array([10, 10])
