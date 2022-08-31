# pylint: disable=missing-docstring
"""Jupyter widgets. Most widget code is excluded from automated testing."""
import ipywidgets as widgets  # type: ignore
from ipywidgets import interact
import matplotlib.pyplot as plt  # type: ignore

import zetta_utils as zu
from . import rendering


def entry_loader(
    entries, renderer, choice, grid_size, grid_x, grid_y
):  # pylint: disable=too-many-arguments # pragma: no cover
    entry = entries[choice].squeeze()
    entry = zu.tensor_ops.convert.to_np(entry)

    x_size = entry.shape[-2] // grid_size
    y_size = entry.shape[-1] // grid_size

    x_coords = (x_size * grid_x, x_size * (grid_x + 1))
    y_coords = (y_size * grid_y, y_size * (grid_y + 1))

    entry = entry[..., x_coords[0] : x_coords[1], y_coords[0] : y_coords[1]]
    entry_rendered = renderer(entry)
    plt.axis("off")
    plt.tight_layout()
    plt.imshow(entry_rendered)


def list_viz(entries, grid_size=1, grid_x=0, grid_y=0, renderer=None):  # pragma: no cover
    """Interactive visualizer for a list of images.

    :param entries: A list of images to be visualized. Images will be squeezed before being
        rendered.
    :param grid_size: The input image will be broken into a uniform grid, and only one cell of
        that grid will be shown at a time. ``grid_size`` determines how many pieces the image
        will be broken into long each dimension. When ``grid_size == 1`` the whole image will
        be shown, when ``grid_size == 2`` the image will be broken into 4 equally sized
        rectangles, etc.
    :param grid_x: Default position on the X axis of the grid.
    :param grid_y: Default position on the Y axis of the grid.

    """
    if renderer is None:
        renderer = rendering.Renderer()

    if isinstance(entries, list):
        entries = {i: entries[i] for i in range(len(entries))}

    button_choice = widgets.ToggleButtons(
        options=entries.keys(),
        description="Entry:",
        disabled=False,
        button_style="",
    )

    grid_size_selector = widgets.IntText(
        value=grid_size, description="Section Count:", disabled=False
    )
    grid_x_selector = widgets.IntText(value=grid_x, description="X section:", disabled=False)
    grid_y_selector = widgets.IntText(value=grid_y, description="Y section:", disabled=False)

    def this_entry_loader(**kwargs):
        return entry_loader(entries=entries, renderer=renderer, **kwargs)

    interact(
        this_entry_loader,
        choice=button_choice,
        grid_size=grid_size_selector,
        grid_x=grid_x_selector,
        grid_y=grid_y_selector,
    )
