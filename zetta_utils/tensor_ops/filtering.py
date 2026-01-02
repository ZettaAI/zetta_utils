"""Statistical filtering operations for tensors."""
import numpy as np
from scipy.stats import median_abs_deviation
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.tensor_ops import convert
from zetta_utils.tensor_ops.common import supports_dict
from zetta_utils.tensor_typing import TensorTypeVar


@builder.register("nanmedian_filter")
@supports_dict
@typechecked
def nanmedian_filter(
    data: TensorTypeVar,
    axis: int = 3,
    sigma: float | None = None,
    threshold: float | None = None,
    fill_value: float = np.nan,
) -> TensorTypeVar:
    """
    Replace values that deviate more than a robust multiple of MAD (`sigma`)
    and/or an absolute threshold from the nan-aware median along the given axis
    with `fill_value`. If both are provided, a value is considered an outlier if
    it violates *either* condition.

    :param data:       Input tensor (e.g. CXYZ).
    :param axis:       Axis along which to compute median/deviation (default 3).
    :param sigma:      Number of robust "standard deviations" (MAD scaled with
                       normal approximation) allowed. Default None.
    :param threshold:  Absolute difference from median allowed. Default None.
    :param fill_value: Value used to replace outliers (default NaN).
    :return:           Tensor with outliers replaced by `fill_value`.
    """
    if sigma is None and threshold is None:
        raise ValueError("Need to specify at least one of `sigma` or `threshold`.")

    data_np = convert.to_np(data)

    median = np.nanmedian(data_np, axis=axis, keepdims=True)
    mask_channel = np.zeros_like(data_np, dtype=bool)

    if sigma is not None:
        mad = median_abs_deviation(data_np, axis=axis, scale="normal", nan_policy="omit")
        mad = np.expand_dims(mad, axis=axis)
        lower = median - sigma * mad
        upper = median + sigma * mad
        mask_channel |= (data_np < lower) | (data_np > upper)

    if threshold is not None:
        lower = median - threshold
        upper = median + threshold
        mask_channel |= (data_np < lower) | (data_np > upper)

    mask_voxel = np.any(mask_channel, axis=0, keepdims=True).repeat(data_np.shape[0], axis=0)

    fill_value_arr = np.atleast_1d(fill_value).astype(data_np.dtype)
    if fill_value_arr.size == 1:
        fill_value_arr = np.repeat(fill_value_arr, data_np.shape[0])
    fill_value_arr = fill_value_arr[:, None, None, None]

    result_np = np.where(mask_voxel, fill_value_arr, data_np)
    return convert.astype(result_np, data)
