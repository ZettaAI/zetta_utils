from __future__ import annotations

from typing import Sequence

import attrs
import torch

from zetta_utils import builder, convnet, mazepa
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.mazepa import semaphore
from zetta_utils.tensor_ops.common import crop


# TODO Fix image mask and output mask dimensions
@builder.register("run_affinities_inference_onnx")
def run_affinities_inference_onnx(
    image: torch.Tensor,
    model_path: str,
    image_mask: torch.Tensor | None = None,
    output_mask: torch.Tensor | None = None,
    myelin_mask_threshold: float = 1.0,
) -> torch.Tensor:  # pragma: no cover

    if image_mask is not None:
        image *= image_mask
    output = convnet.utils.load_and_run_model(path=model_path, data_in=image)

    aff = output[:, :3, ...]
    msk = torch.amax(output[:, 3:, ...], dim=-4, keepdim=True)
    output = torch.concatenate((aff, msk), dim=-4)
    output = output[0]
    output_aff = output[0:3, :, :, :]
    output_mye_mask = output[3:, :, :, :] < myelin_mask_threshold
    if output_mask is not None:
        output_aff *= output_mask * output_mye_mask
    else:
        output_aff *= output_mye_mask
    output = torch.permute(output_aff, (0, 2, 3, 1))

    return output.to(dtype=torch.float32)


@builder.register("AffinityInferenceOperation")
@mazepa.taskable_operation_cls
@attrs.frozen()
class AffinityInferenceOperation:
    crop_pad: Sequence[int] = (0, 0, 0)

    def get_operation_name(self):  # pylint: disable=no-self-use
        return "AffinityInferenceOperation"

    def get_input_resolution(  # pylint: disable=no-self-use
        self, dst_resolution: Vec3D[float]
    ) -> Vec3D[float]:
        return dst_resolution

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> AffinityInferenceOperation:
        return attrs.evolve(self, crop_pad=Vec3D(*self.crop_pad) + crop_pad)

    def __call__(  # pylint: disable=too-many-locals
        self,
        idx: VolumetricIndex,
        image: VolumetricLayer,
        model_path: str,
        dst: VolumetricLayer,
    ) -> None:
        idx_padded = idx.padded(self.crop_pad)
        idx_padded.resolution = self.get_input_resolution(idx_padded.resolution)
        # with semaphore("read"):
        img = torch.Tensor(image[idx_padded])
        img = torch.permute(img, (0, 3, 1, 2)).unsqueeze(0).float()
        with semaphore("cuda"):
            output = run_affinities_inference_onnx(image=img, model_path=model_path)
        output_cropped = crop(output, self.crop_pad).numpy()
        dst[idx] = output_cropped
