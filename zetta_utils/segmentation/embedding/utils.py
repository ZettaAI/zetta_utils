from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from typeguard import typechecked

from zetta_utils.tensor_ops import convert
from zetta_utils.tensor_typing import TensorTypeVar


@typechecked
def vec_to_pca(data: TensorTypeVar) -> TensorTypeVar:
    """
    Transform feature vectors into an RGB map with PCA dimensionality reduction.

    :param data: Feature vectors
    """
    assert (data.ndim == 5) and (data.shape[0] == 1)
    data_np = convert.to_np(data).astype(np.float32)

    # pylint: disable=invalid-name
    dim = data_np.shape[-4]
    data_tp = data_np.transpose(0, 2, 3, 4, 1)
    X = data_tp.reshape(-1, dim)  # (n_samples, n_features)
    pca = PCA(dim).fit_transform(X)
    pca_tp = pca.reshape(data_tp.shape)
    pca_np = pca_tp.transpose(0, 4, 1, 2, 3)
    result = convert.astype(pca_np, data)
    return result


@typechecked
def vec_to_rgb(data: TensorTypeVar) -> TensorTypeVar:
    """
    Transform feature vectors into an RGB map by slicing the first three
    channels and then rescale them.

    :param data: Feature vectors
    :return: RGB map
    """
    assert (data.ndim == 5) and (data.shape[0] == 1)
    data_np = convert.to_np(data).astype(np.float32)
    rgbmap = data_np[:, 0:3, ...]
    rgbmap -= np.min(rgbmap)
    rgbmap /= np.max(rgbmap)
    result = convert.astype(rgbmap, data)
    return result
