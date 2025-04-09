import attrs
import cc3d
import cv2
import einops
import fastremap
import numpy as np
import torch
from numpy import typing as npt
from typeguard import typechecked

from zetta_utils import builder, convnet
from zetta_utils.tensor_ops import convert


@builder.register("ResinDetector")
@typechecked
@attrs.mutable
class ResinDetector:
    # Input uint8 [   0 .. 255]
    # Output uint8 Prediction [0 .. 255]

    # Don't create the model during initialization for efficient serialization
    model_path: str
    tissue_filter_threshold: int = 10000
    resin_filter_threshold: int = 10000

    def __call__(self, src: npt.NDArray) -> npt.NDArray:
        if (src != 0).sum() == 0:
            return np.full_like(src, 1).astype(np.uint8)
        else:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

            # load model during the call _with caching_
            model = convnet.utils.load_model(self.model_path, device=device, use_cache=True)

            if src.dtype == np.uint8:
                data_in_np = src.astype(float) / 255.0  # [0.0 .. 1.0]
            else:
                raise ValueError(f"Unsupported src dtype: {src.dtype}")

            data_in = convert.to_torch(einops.rearrange(data_in_np, "C X Y Z -> Z C X Y")).float()
            data_in = data_in.to(device=device)
            with torch.no_grad():
                result = model(data_in)

            result = einops.rearrange(result, "Z C X Y -> C X Y Z")
            result = torch.sigmoid(result)
            pred = ((result > 128.0 / 255.0)).to(dtype=torch.uint8, device="cpu")

            # Background is resin
            pred[src == 0.0] = 1

            # Filter small islands of tissue
            tissue = (1 - pred).squeeze().numpy()
            tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            if self.tissue_filter_threshold > 0:
                # TODO: refactor logic with the below & test
                islands = cc3d.connected_components(tissue)
                uniq, counts = fastremap.unique(islands, return_counts=True)
                islands = fastremap.mask(
                    islands,
                    [lbl for lbl, cnt in zip(uniq, counts) if cnt < self.tissue_filter_threshold],
                )
                tissue[islands == 0] = 0

            # Filter small islands of resin
            resin = 1 - tissue
            if self.resin_filter_threshold > 0:
                islands = cc3d.connected_components(resin)
                uniq, counts = fastremap.unique(islands, return_counts=True)
                islands = fastremap.mask(
                    islands,
                    [lbl for lbl, cnt in zip(uniq, counts) if cnt < self.resin_filter_threshold],
                )
                resin[islands == 0] = 0

        return resin.reshape(pred.shape)
