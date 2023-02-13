from __future__ import annotations

from typing import Literal, Sequence

from zetta_utils import builder, mazepa, parsing
from zetta_utils.geometry import BBox3D, IntVec3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricLayer
from zetta_utils.typing import check_type

from ..common import build_write_flow


@mazepa.flow_schema_cls
class AnnotatedSectionCopyFlowSchema:
    def flow(  # pylint: disable=no-self-use
        self,
        dst: VolumetricLayer,
        src: VolumetricLayer,
        start_coord_xy: Sequence[float],
        end_coord_xy: Sequence[float],
        coord_resolution_xy: Sequence[float],
        fill_resolutions: Vec3D | Sequence[Vec3D],
        chunk_size_xy: Sequence[int],
        annotation_path: str,
        order: Literal["low_to_high", "high_to_low", "as_given", "concurrent"] = "low_to_high",
    ):
        annotations = parsing.ngl_state.read_remote_annotations(annotation_path)
        annotation_zs = []
        for e in annotations:
            if not check_type(e, Vec3D):
                raise ValueError(f"All given annotations must be of Point type. Got: {type(e)}")
            annotation_zs.append(e[-1])  # type: ignore # mypy doesn't undertand `check_type`

        if order == "low_to_high":
            annotation_zs.sort()
        elif order == "high_to_low":
            annotation_zs.sort(reverse=True)
        else:
            ...

        if not check_type(fill_resolutions, Vec3D):
            fill_resolutions_list = fill_resolutions
        else:
            fill_resolutions_list = [
                fill_resolutions  # type: ignore # mypy doesn't get check_type
            ]

        for z in annotation_zs:
            for fill_res in fill_resolutions_list:
                bbox = BBox3D.from_coords(
                    start_coord=Vec3D(start_coord_xy[0], start_coord_xy[1], z),
                    end_coord=Vec3D(end_coord_xy[0], end_coord_xy[1], z + fill_res[-1]),
                    resolution=Vec3D(coord_resolution_xy[0], coord_resolution_xy[1], 1),
                )
                flow = build_write_flow(
                    chunk_size=IntVec3D(chunk_size_xy[0], chunk_size_xy[1], 1),
                    bbox=bbox,
                    dst_resolution=fill_res,
                    src=src,
                    dst=dst,
                )
                yield flow

            if order != "concurrent":
                yield mazepa.Dependency()


@builder.register("build_annotated_section_copy_flow")
def build_annotated_section_copy_flow(
    dst: VolumetricLayer,
    src: VolumetricLayer,
    start_coord_xy: Sequence[float],
    end_coord_xy: Sequence[float],
    coord_resolution_xy: Sequence[float],
    fill_resolutions: Vec3D | Sequence[Vec3D],
    chunk_size_xy: Sequence[int],
    annotation_path: str,
    order: Literal["low_to_high", "high_to_low", "as_given", "concurrent"] = "low_to_high",
) -> mazepa.Flow:
    return AnnotatedSectionCopyFlowSchema()(
        dst=dst,
        src=src,
        start_coord_xy=start_coord_xy,
        end_coord_xy=end_coord_xy,
        coord_resolution_xy=coord_resolution_xy,
        fill_resolutions=fill_resolutions,
        chunk_size_xy=chunk_size_xy,
        annotation_path=annotation_path,
        order=order,
    )
