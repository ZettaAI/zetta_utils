import os
from typing import Sequence

import attrs
import fsspec
import numpy as np

from zetta_utils import builder, mazepa
from zetta_utils.geometry.vec import Vec3D
from zetta_utils.layer.volumetric.layer import VolumetricLayer


@builder.register("PartnerSVLookupHacky")
@mazepa.taskable_operation_cls
@attrs.mutable
class PartnerSVLookupHacky:
    chunk_size: Sequence[int]
    chunk_dir: str
    output_dir: str
    ws_layer: VolumetricLayer
    resolution: Sequence[float]
    pad: Sequence[int]

    def __call__(self, chunk_id: Sequence[int]):
        cleft_ids_path = os.path.join(
            self.chunk_dir, f"cleft_ids_{chunk_id[0]}_{chunk_id[1]}_{chunk_id[2]}.npy"
        )
        coords_vx_path = os.path.join(
            self.chunk_dir, f"coords_vx_{chunk_id[0]}_{chunk_id[1]}_{chunk_id[2]}.npy"
        )
        with fsspec.open(cleft_ids_path, "rb") as f:
            cleft_ids = np.load(f)
        with fsspec.open(coords_vx_path, "rb") as f:
            coords_vx = np.load(f)

        chunk_size_vec = Vec3D(*[int(e) for e in self.chunk_size])
        chunk_id_vec = Vec3D(*[int(e) for e in chunk_id])
        start_coord = chunk_size_vec * chunk_id_vec - Vec3D(*self.pad)
        end_coord = chunk_size_vec * (chunk_id_vec + 1) + Vec3D(*self.pad)
        chunk_data = self.ws_layer[
            Vec3D(*self.resolution),
            start_coord[0] : end_coord[0],
            start_coord[1] : end_coord[1],
            start_coord[2] : end_coord[2],
        ].squeeze()

        rows = []
        for i in range(cleft_ids.shape[0]):
            cleft_id = cleft_ids[i]
            presyn_vx = coords_vx[i, 0].astype(int)
            postsyn_vx = coords_vx[i, 1].astype(int)
            presyn_relative_coord = presyn_vx - start_coord
            presyn_sv_id = chunk_data[
                presyn_relative_coord[0], presyn_relative_coord[1], presyn_relative_coord[2]
            ]
            postsyn_relative_coord = postsyn_vx - start_coord
            postsyn_sv_id = chunk_data[
                postsyn_relative_coord[0], postsyn_relative_coord[1], postsyn_relative_coord[2]
            ]
            rows.append(f"{cleft_id},{presyn_sv_id},{postsyn_sv_id}")

        result_path = os.path.join(
            self.output_dir, f"{chunk_id[0]}_{chunk_id[1]}_{chunk_id[2]}.csv"
        )
        with fsspec.open(result_path, "w") as f:
            f.write("\n".join(rows))


@builder.register("partner_sv_lookup_hacky_flow")
@mazepa.flow_schema
def partner_sv_lookup_hacky_flow(
    nonempty_chunk_ids_path: str,
    chunk_size: Sequence[int],
    chunk_dir: str,
    output_dir: str,
    pad: Sequence[int],
    ws_layer: VolumetricLayer,
    resolution: Sequence[float],
):
    with open(nonempty_chunk_ids_path, "rb") as f:
        nonempty_chunk_ids = np.load(f)

    operation = PartnerSVLookupHacky(
        chunk_size=chunk_size,
        chunk_dir=chunk_dir,
        output_dir=output_dir,
        pad=pad,
        ws_layer=ws_layer,
        resolution=resolution,
    )

    tasks = [
        operation.make_task(nonempty_chunk_ids[i]) for i in range(nonempty_chunk_ids.shape[0])
    ]
    yield tasks
