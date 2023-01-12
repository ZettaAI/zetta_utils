# pylint: disable=missing-docstring
# type: ignore
"""ZettaAI Python plotting utilities. Most of the plotting functions are not included
in automated testing, as they're not meand for production use."""
from __future__ import annotations

import copy
import gc
import io

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typeguard import typechecked

from zetta_utils import tensor_ops
from zetta_utils.tensor_typing import Tensor

# matplotlib.use("Agg")
DEFAULT_COMMON_KWARGS = {
    "dpi": 80,
    "figsize": (8, 8),
}


@typechecked
class Renderer:  # pylint: disable=too-few-public-methods
    def __init__(  # pylint: disable=too-many-arguments
        self,
        common_kwargs=None,
        fld_kwargs=None,
        img_kwargs=None,
        msk_kwargs=None,
        seg_kwargs=None,
    ):  # pragma: no cover
        if common_kwargs is None:
            common_kwargs = copy.deepcopy(DEFAULT_COMMON_KWARGS)
        else:
            common_kwargs = DEFAULT_COMMON_KWARGS | common_kwargs

        if img_kwargs is None:
            img_kwargs = {}
        if msk_kwargs is None:
            msk_kwargs = {}
        if fld_kwargs is None:
            fld_kwargs = {}
        if seg_kwargs is None:
            seg_kwargs = {}

        self.img_kwargs = common_kwargs | img_kwargs
        self.msk_kwargs = common_kwargs | msk_kwargs
        self.fld_kwargs = common_kwargs | fld_kwargs
        self.seg_kwargs = common_kwargs | seg_kwargs

    def __call__(self, x):  # pragma: no cover
        x = tensor_ops.convert.to_np(x).squeeze()
        if len(x.shape) == 3:
            return render_fld(x, **self.fld_kwargs)
        if x.dtype == bool:
            return render_msk(x, **self.msk_kwargs)
        if np.issubdtype(x.dtype, np.integer):
            return render_seg(x, **self.seg_kwargs)
        # else:
        return render_img(x, **self.img_kwargs)


@typechecked
def get_img_from_fig(
    fig: matplotlib.figure.Figure,
    dpi: int,
) -> npt.NDArray:  # pragma: no cover
    """
    Render matplotlib image to an RGB numpy array.

    Args:
        fig (matplotlib.figure.Figure): Figure to be rendered.
        dpi (int): DPI of the rendered image.
    Returns:
        npt.NDArray: RGB image
    """
    plt.tight_layout()
    plt.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    # plt.pause(.1) #TODO: investigate the effects
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    # buf.close()
    img = cv2.imdecode(img_arr, 1)  # pylint: disable=E1101
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # pylint: disable=E1101

    return img


@typechecked
def render_img(
    img: Tensor,
    figsize: tuple[int, int] = (8, 8),
    dpi: int = 80,
    cmap: str = "gray",
) -> npt.NDArray:  # pragma: no cover
    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    plt.axis("off")
    plt.imshow(img, cmap=cmap)
    result = get_img_from_fig(fig, dpi=dpi)
    plt.cla()
    plt.clf()
    plt.close(fig)
    plt.close()
    del fig
    gc.collect()
    return result


render_msk = render_img
render_seg = render_img


@typechecked
def render_fld(  # pylint: disable=too-many-locals,too-many-arguments
    fld: Tensor,
    figsize: tuple[int, int] = (8, 8),
    dpi: int = 80,
    grid_size: int = 50,
    alpha: float = 0.6,
    linewidth: float = 0.5,
) -> npt.NDArray:  # pragma: no cover
    """
    Render the given vector field to an RGB numpy image.

    The field is assumed to be in a residual format and pixel units.

    :param fld: An array of shape [(1,) H, W, 2] representing a 2D vector field.
    :param grid_side: float of arrows per side.
    :return: RGB rendering of the vector field.
    """
    fld = tensor_ops.convert.to_np(fld).squeeze()
    if fld.shape[0] == 2:
        fld = np.transpose(fld, (1, 2, 0))

    assert fld.shape[0] == fld.shape[1]

    n = fld.shape[0]
    y, x = np.mgrid[0:n, 0:n]

    x_coords = [0, fld.shape[0]]
    y_coords = [0, fld.shape[1]]

    ex = (1) * fld[:, :, 0]
    ey = fld[:, :, 1]
    r = np.arctan2(ex, ey)

    interval = (x_coords[1] - x_coords[0]) // grid_size

    fig = plt.figure(figsize=figsize)
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

    result = get_img_from_fig(fig, dpi=dpi)
    plt.close()
    plt.close(fig)
    plt.cla()
    plt.clf()
    del fig
    gc.collect()
    return result
