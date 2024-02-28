from typing import Protocol, TypeVar

import torch
from typing_extensions import ParamSpec

from zetta_utils import builder
from zetta_utils.tensor_ops import convert
from zetta_utils.tensor_typing import TensorTypeVar

from .mask import kornia_erosion

P = ParamSpec("P")


class MultiTensorOp(Protocol[P]):
    """
    Protocol which defines what it means for a function to be a MultiTensorOp:
    it must take a `data1` and 'data2' arguments of TensorTypeVar type, and return a
    tensor of the same type.
    """

    def __call__(
        self, data1: TensorTypeVar, data2: TensorTypeVar, *args: P.args, **k: P.kwargs
    ) -> TensorTypeVar:
        ...


OpT = TypeVar("OpT", bound=MultiTensorOp)


def skip_on_empty_datas(fn: OpT) -> OpT:
    """
    Decorator that ensures early exit for a tensor op when `data1` and 'data2' are both zeros.
    """

    def wrapped(
        data1: TensorTypeVar, data2: TensorTypeVar, *args: P.args, **kwargs: P.kwargs
    ) -> TensorTypeVar:

        if (data1 != 0).sum() == 0 and (data2 != 0).sum() == 0:
            result = data1
        else:
            result = fn(data1, data2, *args, **kwargs)
        return result

    return wrapped  # type: ignore


@builder.register("compute_pixel_error")  # type: ignore
@skip_on_empty_datas
def compute_pixel_error(
    data1: TensorTypeVar, data2: TensorTypeVar, erosion: int = 5, **kwargs
) -> TensorTypeVar:
    """
    Returns the symmetric pixel difference of two tensors in the area
    where two tensors overlap after erosion using a square kernel
    to exclude edge artifacts.
    :param data1: Input tensor (CXYZ).
    :param data2: Input tensor (CXYZ).
    :param erosion: Follows skimage convention, defaults to 5.
    :param kwargs: Additional keyword arguments passed to kornia_erosion.
    :return: The symmetric difference of the two input tensors.
    """
    data1_torch = convert.to_torch(data1)
    data2_torch = convert.to_torch(data2)
    dilated_mask = torch.logical_or(
        kornia_erosion(  # type: ignore
            data1_torch, kernel="square", device=data1_torch.device, width=erosion, **kwargs
        )
        == 0,
        kornia_erosion(  # type: ignore
            data2_torch, kernel="square", device=data2_torch.device, width=erosion, **kwargs
        )
        == 0,
    )
    zeros = torch.zeros_like(data1_torch)
    result = torch.where(
        dilated_mask,
        zeros,
        torch.minimum((data1_torch - data2_torch).abs(), (data2_torch - data1_torch).abs()),
    )
    return convert.astype(result, data1)


@builder.register("erode_combine")  # type: ignore
@skip_on_empty_datas
def erode_combine(
    data1: TensorTypeVar, data2: TensorTypeVar, erosion: int = 5, **kwargs
) -> TensorTypeVar:
    """
    Combines two tensors by taking values from each one where they do not overlap,
    and averaging the two where they do. The overlap is determined after erosion
    using a square kernel to exclude edge artifacts.
    :param data1: Input tensor (CXYZ).
    :param data2: Input tensor (CXYZ).
    :param erosion: Follows skimage convention, defaults to 5.
    :param kwargs: Additional keyword arguments passed to kornia_erosion.
    :return: The blended sum of two input tensors.
    """
    data1_torch = convert.to_torch(data1)
    data2_torch = convert.to_torch(data2)
    mask1 = (
        kornia_erosion(  # type: ignore
            data1_torch, kernel="square", device=data1_torch.device, width=erosion, **kwargs
        )
        != 0
    )
    mask2 = (
        kornia_erosion(  # type: ignore
            data2_torch, kernel="square", device=data2_torch.device, width=erosion, **kwargs
        )
        != 0
    )
    result = torch.where(
        torch.logical_and(mask1, mask2),
        (0.5 * data1_torch + 0.5 * data2_torch).to(data1_torch.dtype),
        torch.zeros_like(data1_torch),
    )
    result = torch.where(torch.logical_not(mask2), data1_torch, result)
    result = torch.where(torch.logical_not(mask1), data2_torch, result)

    return convert.astype(result, data1)
