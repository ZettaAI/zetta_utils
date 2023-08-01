import copy
from typing import Literal, Protocol, TypeVar, Union

import cc3d
import einops
import fastremap
import numpy as np
import torch
from kornia import morphology
from skimage.morphology import diamond, disk, square, star
from typeguard import typechecked
from typing_extensions import ParamSpec

from zetta_utils import builder
from zetta_utils.tensor_typing import Tensor, TensorTypeVar

from . import convert

MaskFilteringModes = Literal["keep_large", "keep_small"]

P = ParamSpec("P")


class TensorOp(Protocol[P]):
    """
    Protocol which defines what it means for a function to be a tensor_op:
    it must take a `data` argument of TensorTypeVar type, and return a
    tensor of the same type.
    """

    def __call__(self, data: TensorTypeVar, *args: P.args, **k: P.kwargs) -> TensorTypeVar:
        ...


OpT = TypeVar("OpT", bound=TensorOp)


def skip_on_empty_data(fn: OpT) -> OpT:
    """
    Decorator that ensures early exit for a tensor op when `data` is 0.
    """

    def wrapped(data: TensorTypeVar, *args: P.args, **kwargs: P.kwargs) -> TensorTypeVar:
        if (data != 0).sum() == 0:
            result = data
        else:
            result = fn(data, *args, **kwargs)
        return result

    return wrapped  # type: ignore


@builder.register("filter_cc")  # type: ignore # TODO: pyright
@skip_on_empty_data
@typechecked
def filter_cc(
    data: TensorTypeVar,
    mode: MaskFilteringModes = "keep_small",
    thr: int = 100,
) -> TensorTypeVar:
    """
    Remove connected components from the given input tensor_ops.

    Clustering is performed based on non-zero values.

    :param data: Input tensor (CXYZ).
    :param mode: Filtering mode.
    :param thr:  Pixel size threshold.
    :return: Tensor with the filtered clusters removed.
    """
    data_np = convert.to_np(data)

    data_np = einops.rearrange(data_np, "C X Y Z -> Z C X Y")
    assert data_np.shape[1] == 1

    result_raw = np.zeros_like(data_np)

    for z in range(data_np.shape[0]):
        if (data_np[z] != 0).sum() > 0:
            cc_labels = cc3d.connected_components(data_np[z] != 0)
            segids, counts = np.unique(cc_labels, return_counts=True)
            if mode == "keep_large":
                segids = [segid for segid, ct in zip(segids, counts) if ct > thr]
            else:
                segids = [segid for segid, ct in zip(segids, counts) if ct <= thr]

            filtered_mask = fastremap.mask_except(cc_labels, segids, in_place=True) != 0

            result_raw[z] = copy.copy(data_np[z])
            result_raw[z][filtered_mask == 0] = 0

    result_raw = einops.rearrange(result_raw, "Z C X Y -> C X Y Z")
    result = convert.astype(result_raw, data)
    return result


def _normalize_kernel(
    kernel: Union[Tensor, str], width: int, device: torch.types.Device
) -> torch.Tensor:
    if isinstance(kernel, str):
        if kernel == "square":
            return convert.to_torch(square(width), device=device)
        if kernel == "diamond":
            return convert.to_torch(diamond(width), device=device)
        if kernel == "disk":
            return convert.to_torch(disk(width), device=device)
        if kernel == "star":
            return convert.to_torch(star(width), device=device)
        else:
            raise ValueError(f"Unknown kernel type {kernel}")
    if kernel.ndim != 2:
        raise ValueError(f"Currently only 2D kernel supported, got {kernel.ndim}D")
    return convert.to_torch(kernel, device=device)


@builder.register("kornia_opening")  # type: ignore
@skip_on_empty_data
@typechecked
def kornia_opening(
    data: TensorTypeVar,
    kernel: Union[Tensor, str] = "square",
    device: torch.types.Device = None,
    width: int = 3,
    **kwargs,
) -> TensorTypeVar:
    """
    Close the given mask. Uses kornia.morphology.opening and supports a selection of
    skimage.morphology 2D footprints as structuring element.

    See https://kornia.readthedocs.io/en/latest/morphology.html#kornia.morphology.opening and
    https://scikit-image.org/docs/stable/auto_examples/numpy_operations/plot_structuring_elements.html
    for additional information.

    :param data: Input mask tensor (CXYZ).
    :param kernel: Either a 2D kernel, or one of "square", "diamond", "disk", "star",
                   defaults to "square".
    :param device: Target device for opening operation, defaults to None (using data.device)
    :param width: Follows skimage convention, defaults to 3, ignored if kernel is a `Tensor`.
    :param kwargs: Additional keyword arguments passed to kornia.morphology.opening
    :return: The opened mask, same type as input.
    """
    data_torch_cxyz = convert.to_torch(data, device=device)
    kernel_torch = _normalize_kernel(kernel, width, device=data_torch_cxyz.device)
    data_torch = einops.rearrange(data_torch_cxyz, "C X Y Z -> Z C X Y")

    result_torch = morphology.opening(
        data_torch,
        kernel=kernel_torch,
        max_val=kwargs.pop("max_val", kernel_torch.max()),
        **kwargs,
    )

    result_torch = result_torch.to(data_torch.dtype)
    result = convert.astype(einops.rearrange(result_torch, "Z C X Y -> C X Y Z"), data)
    return result


@builder.register("kornia_closing")  # type: ignore
@skip_on_empty_data
@typechecked
def kornia_closing(
    data: TensorTypeVar,
    kernel: Union[Tensor, str] = "square",
    device: torch.types.Device = None,
    width: int = 3,
    **kwargs,
) -> TensorTypeVar:
    """
    Close the given mask. Uses kornia.morphology.closing and supports a selection of
    skimage.morphology 2D footprints as structuring element.

    See https://kornia.readthedocs.io/en/latest/morphology.html#kornia.morphology.closing and
    https://scikit-image.org/docs/stable/auto_examples/numpy_operations/plot_structuring_elements.html
    for additional information.

    :param data: Input mask tensor (CXYZ).
    :param kernel: Either a 2D kernel, or one of "square", "diamond", "disk", "star",
                   defaults to "square".
    :param device: Target device for closing operation, defaults to None (using data.device)
    :param width: Follows skimage convention, defaults to 3, ignored if kernel is a `Tensor`.
    :param kwargs: Additional keyword arguments passed to kornia.morphology.closing
    :return: The closed mask, same type as input.
    """
    data_torch_cxyz = convert.to_torch(data, device=device)
    kernel_torch = _normalize_kernel(kernel, width, device=data_torch_cxyz.device)
    data_torch = einops.rearrange(data_torch_cxyz, "C X Y Z -> Z C X Y")

    result_torch = morphology.closing(
        data_torch,
        kernel=kernel_torch,
        max_val=kwargs.pop("max_val", kernel_torch.max()),
        **kwargs,
    )

    result_torch = result_torch.to(data_torch.dtype)
    result = convert.astype(einops.rearrange(result_torch, "Z C X Y -> C X Y Z"), data)
    return result


@builder.register("kornia_erosion")  # type: ignore
@skip_on_empty_data
@typechecked
def kornia_erosion(
    data: TensorTypeVar,
    kernel: Union[Tensor, str] = "square",
    device: torch.types.Device = None,
    width: int = 3,
    **kwargs,
) -> TensorTypeVar:
    """
    Erode the given mask. Uses kornia.morphology.erosion and supports a selection of
    skimage.morphology 2D footprints as structuring element.

    See https://kornia.readthedocs.io/en/latest/morphology.html#kornia.morphology.erosion and
    https://scikit-image.org/docs/stable/auto_examples/numpy_operations/plot_structuring_elements.html
    for additional information.

    :param data: Input mask tensor (CXYZ).
    :param kernel: Either a 2D kernel, or one of "square", "diamond", "disk", "star",
                   defaults to "square".
    :param device: Target device for erosion operation, defaults to None (using data.device)
    :param width: Follows skimage convention, defaults to 3, ignored if kernel is a `Tensor`.
    :param kwargs: Additional keyword arguments passed to kornia.morphology.erosion
    :return: The eroded mask, same type as input.
    """
    data_torch_cxyz = convert.to_torch(data, device=device)
    kernel_torch = _normalize_kernel(kernel, width, device=data_torch_cxyz.device)
    data_torch = einops.rearrange(data_torch_cxyz, "C X Y Z -> Z C X Y")

    result_torch = morphology.erosion(
        data_torch,
        kernel=kernel_torch,
        max_val=kwargs.pop("max_val", kernel_torch.max()),
        **kwargs,
    )

    result_torch = result_torch.to(data_torch.dtype)
    result = convert.astype(einops.rearrange(result_torch, "Z C X Y -> C X Y Z"), data)
    return result


@builder.register("kornia_dilation")  # type: ignore
@skip_on_empty_data
@typechecked
def kornia_dilation(
    data: TensorTypeVar,
    kernel: Union[Tensor, str] = "square",
    device: torch.types.Device = None,
    width: int = 3,
    **kwargs,
) -> TensorTypeVar:
    """
    Dilate the given mask. Uses kornia.morphology.dilation and supports a selection of
    skimage.morphology 2D footprints as structuring element.

    See https://kornia.readthedocs.io/en/latest/morphology.html#kornia.morphology.dilation and
    https://scikit-image.org/docs/stable/auto_examples/numpy_operations/plot_structuring_elements.html
    for additional information.

    :param data: Input mask tensor (CXYZ).
    :param kernel: Either a 2D kernel, or one of "square", "diamond", "disk", "star",
                   defaults to "square".
    :param device: Target device for dilation operation, defaults to None (using data.device)
    :param width: Follows skimage convention, defaults to 3, ignored if kernel is a `Tensor`.
    :param kwargs: Additional keyword arguments passed to kornia.morphology.dilation
    :return: The dilated mask, same type as input.
    """
    data_torch_cxyz = convert.to_torch(data, device=device)
    kernel_torch = _normalize_kernel(kernel, width, device=data_torch_cxyz.device)
    data_torch = einops.rearrange(data_torch_cxyz, "C X Y Z -> Z C X Y")

    result_torch = morphology.dilation(
        data_torch,
        kernel=kernel_torch,
        max_val=kwargs.pop("max_val", kernel_torch.max()),
        **kwargs,
    )

    result_torch = result_torch.to(data_torch.dtype)
    result = convert.astype(einops.rearrange(result_torch, "Z C X Y -> C X Y Z"), data)
    return result
