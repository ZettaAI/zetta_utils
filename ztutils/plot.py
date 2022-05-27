import numpy as np


def visualize_residuals(res, figsize=(10,10), x_coords=None, y_coords=None, vec_grid=50):
    res = prepare_for_show(res)
    if res.shape[0] == 2:
        res = np.transpose(res, (1, 2, 0))

    assert res.shape[0] == res.shape[1]
    plt.figure(figsize=figsize)
    n = res.shape[0]
    y, x = np.mgrid[0:n, 0:n]

    if x_coords is None:
        x_coords = [0, res.shape[0]]
    if y_coords is None:
        y_coords = [0, res.shape[1]]

    ex = (1) * res[:, :, 0]
    ey = res[:, :, 1]
    r = np.arctan2(ex, ey)

    interval = (x_coords[1] - x_coords[0]) // vec_grid

    plt.quiver(  x[x_coords[0]:x_coords[1]:interval, y_coords[0]:y_coords[1]:interval],
                 y[x_coords[0]:x_coords[1]:interval, y_coords[0]:y_coords[1]:interval],
                ex[x_coords[0]:x_coords[1]:interval, y_coords[0]:y_coords[1]:interval],
                ey[x_coords[0]:x_coords[1]:interval, y_coords[0]:y_coords[1]:interval],
                 r[x_coords[0]:x_coords[1]:interval, y_coords[0]:y_coords[1]:interval], alpha=0.6)
    plt.quiver(x[x_coords[0]:x_coords[1]:interval, y_coords[0]:y_coords[1]:interval],
                 y[x_coords[0]:x_coords[1]:interval, y_coords[0]:y_coords[1]:interval],
                ex[x_coords[0]:x_coords[1]:interval, y_coords[0]:y_coords[1]:interval],
                ey[x_coords[0]:x_coords[1]:interval, y_coords[0]:y_coords[1]:interval], edgecolor='k', facecolor='None', linewidth=.5)
    plt.gca().invert_yaxis()

d
