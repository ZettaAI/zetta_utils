# type: ignore
# pylint: skip-file
from __future__ import annotations

import math

from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox

from zetta_utils import log, mazepa
from zetta_utils.builder.built_in_registrations import efficient_parse_lambda_str
from zetta_utils.geometry.bbox import BBox3D
from zetta_utils.geometry.vec import Vec3D
from zetta_utils.layer.volumetric.cloudvol.build import build_cv_layer
from zetta_utils.layer.volumetric.tools import VolumetricIndexTranslator
from zetta_utils.mazepa_addons.configurations.execute_on_gcp_with_sqs import (
    execute_on_gcp_with_sqs,
)
from zetta_utils.mazepa_layer_processing.common import build_subchunkable_apply_flow
from zetta_utils.ng.link_builder import make_ng_link

logger = log.get_logger("zetta_utils")
log.set_verbosity("INFO")
log.configure_logger()

SOURCE_PATHS = {
    # "microns_pinky": {
    #     "src_res": [32, 32, 40],
    #     "path": "gs://neuroglancer/pinky100_v0/son_of_alignment_v15_rechunked",
    #     "bbox": [[6144, 5120, 17], [14336, 9216, 1201]],
    #     "n": 256,
    #     "stride": 1,
    #     # "chunk_size": [256, 256, 16],
    #     # Boss version downsamples are partially corrupt
    # },
    # "microns_basil": {
    #     "src_res": [32, 32, 40],
    #     "path": "https://s3.amazonaws.com/bossdb-open-data/iarpa_microns/basil/em",
    #     "bbox": [[2048, 2048, 0], [27648, 32768, 993]],
    #     "n": 10,
    #     "stride": 1,
    #     # "chunk_size": [128, 128, 64],
    # },
    # "microns_minnie": {
    #     "src_res": [32, 32, 40],
    #     "path": "gs://iarpa_microns/minnie/minnie65/em",
    #     "bbox": [[3072, 3072, 14832], [56320, 48128, 24464]],
    #     "n": 4,
    #     "stride": 2400,
    #     # "chunk_size": [64, 64, 64],
    #     # 14825-27904, S3 version has strong JPEG artifacts
    # },
    # "microns_interneuron": {
    #     "src_res": [8, 8, 40],
    #     "dst_res": [32, 32, 40],
    #     "path": "https://s3.amazonaws.com/bossdb-open-data/iarpa_microns/interneuron/em",
    #     "bbox": [[20480, 24576, 4683], [110592, 114688, 17286]],
    #     "n": 16,
    #     "stride": 787,
    #     # "chunk_size": [2048, 2048, 1],
    #     # 4005-17347, stronger JPEG artifacts at 16x16 and 32x32 due to low-contrast
    # },
    # "aibs_v1dd": {
    #     "src_res": [38.8, 38.8, 45],
    #     "path": "gs://v1dd_imagery/image/aligned_image",
    #     "bbox": [[9216, 5120, 0], [40960, 26624, 15708]],
    #     "n": 12,
    #     "stride": 1280,
    #     # "chunk_size": [64, 64, 64],
    #     # Bounding box way too large - try max: 40,960 x 32,768
    # },
    # "kim_n2da": {
    #     "src_res": [32, 32, 50],
    #     "path": "gs://zetta_jkim_001_n2da_1430/tests/corgie_tests/uint8_siftv11_newnets_onepass_m7m5m3_retry4_m55333/img/img_rendered",
    #     "bbox": [[0, 0, 1], [1024, 1024, 622]],
    #     "n": 8192,  # all 621 sections
    #     "stride": 1,
    #     # "chunk_size": [1024, 1024, 1],
    # },
    # "kim_pfc2022": {
    #     "src_res": [16, 16, 40],
    #     "dst_res": [32, 32, 40],
    #     "path": "gs://zetta_jkim_001_pfc2022_em/pfc/v1",
    #     "bbox": [[0, 0, 4], [14336, 12288, 1205]],
    #     "n": 183,
    #     "stride": 1,
    #     # "chunk_size": [256, 256, 16],
    #     # jpeg artifacts at 32x32
    # },
    # "kronauer_cra9": {
    #     "src_res": [4, 4, 42],
    #     "dst_res": [32, 32, 42],
    #     "path": "gs://dkronauer-ant-001-drop/cra9_inspection_4nm_sections2665-2680",
    #     "bbox": [[73728, 49152, 2665], [172032, 131072, 2679]],
    #     "n": 68,  # all 14 sections
    #     "stride": 1,
    #     # "chunk_size": [1024, 1024, 1],
    #     # 2665-2680, 4nm only
    # },
    # "kubota_001": {
    #     "src_res": [20, 20, 40],
    #     "path": "gs://zetta_kubota_001_alignment/v1",
    #     "bbox": [[1024, 1024, 0], [6144, 6144, 1191]],
    #     "n": 300,
    #     "stride": 1,
    #     # "chunk_size": [512, 512, 8],
    #     # suspicious of this resolution... 20 nm looks closer to 30-40nm
    # },
    # "lee_fanc": {
    #     "src_res": [34.4, 34.4, 45],
    #     "path": "gs://zetta_lee_fly_vnc_001_precomputed/vnc1_full_v3align_2/realigned_v1",
    #     "bbox": [[0, 0, 0], [10240, 27648, 4400]],
    #     "n": 30,
    #     "stride": 146,
    #     # "chunk_size": [1024, 1024, 1],
    # },
    # "lee_banc": {
    #     "src_res": [32, 32, 45],
    #     "path": "gs://zetta_lee_fly_cns_001_alignment/aligned/v0",
    #     "bbox": [[1024, 1024, 0], [26624, 32768, 7010]],
    #     "n": 10,
    #     "stride": 701,
    #     # "chunk_size": [2048, 2048, 1],
    # },
    # "lee_ppc": {
    #     "src_res": [8, 8, 40],
    #     "dst_res": [32, 32, 40],
    #     "path": "gs://zetta_lee_mouse_ppc_001_alignment/test_bbox/m7_sm2000_m5_sm2000_m3_sm2000_300iter/img/img/img_rendered",
    #     "bbox": [[144384, 39936, 12], [156672, 52224, 1241]],
    #     "n": 910,
    #     "stride": 1,
    #     # "chunk_size": [2048, 2048, 1],
    #     # Cutout: 144384, 39936, 12 - 156672, 52224, 1241 @ 8x8x40 only
    # },
    # "lee_mosquito": {
    #     "src_res": [16, 16, 40],
    #     "dst_res": [32, 32, 40],
    #     "path": "gs://zetta_lee_mosquito_001_raw_image/V1_aligned/raw",
    #     "bbox": [[0, 0, 3000], [44032, 28160, 4747]],
    #     "n": 28,
    #     "stride": 62,
    #     # "chunk_size": [1024, 1024, 1],
    #     # 16x16x40 only
    # },
    # "lichtman_zebrafish": {
    #     "src_res": [32, 32, 30],
    #     "path": "gs://zetta_jlichtman_zebrafish_001_alignment/fine_full_v2/img",
    #     "bbox": [[1024, 2048, 0], [10240, 14336, 4010]],
    #     "n": 76,
    #     "stride": 52,
    #     # "chunk_size": [2048, 2048, 1],
    # },
    # # "neitz_macaque": {
    # #   "src_res": [10,10,50],
    # #   "path": "gs://zetta_neitz_macaque_retina_001_alignment_temp/13846-17051_11069-14269_5-2354/image_stitch_decay140_z1230-2250_mip1"
    # #   # too small and render artifacts
    # # },
    # "prieto_godino_larva": {
    #     "src_res": [32, 32, 32],
    #     "path": "gs://zetta-prieto-godino-fly-larva-001-image/image-v1-iso",
    #     "bbox": [[0, 0, 0], [4218, 4531, 3442]],
    #     "n": 450,
    #     "stride": 1,
    #     # "chunk_size": [128, 128, 128],
    # },
    # "fafb_v15": {
    #     "src_res": [32, 32, 40],
    #     "path": "https://tigerdata.princeton.edu/sseung-test1/fafb-v15-alignment-temp/fine_final/z0_7063/v1/aligned/mip1",
    #     "bbox": [[2048, 2048, 0], [29696, 14336, 7063]],
    #     "n": 25,
    #     "stride": 280,
    #     # "chunk_size": [512, 512, 8],
    # },
    # "lichtman_h01": {
    #     "src_res": [8, 8, 33],
    #     "dst_res": [32, 32, 33],
    #     "path": "gs://h01-release/data/20210601/4nm_raw",
    #     "bbox": [[61440, 45056, 0], [491520, 286720, 5293]],
    #     "n": 3,
    #     "stride": 1760,
    #     # "chunk_size": [128, 128, 32],
    #     # 16x16 starts showing JPEG artifacts
    # },
    # # "janelia_hemibrain": {
    # #     "src_res": [32, 32, 32],
    # #     "path": "gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg",
    # #     "bbox": [[0, 1024, 0], [8606, 9216, 10240]],
    # #     "n": 102,
    # #     "stride": 1,
    # #     # "chunk_size": [64, 64, 64],
    # #     # Slab interfaces - need to use yz, requires manual transpose
    # # },
    # "janelia_manc": {
    #     "src_res": [32, 32, 32],
    #     # "path": "gs://flyem-vnc-2-26-213dba213ef26e094c16c860ae7f4be0/v3_emdata_clahe_xy/jpeg",
    #     "path": "https://storage.googleapis.com/flyem-vnc-2-26-213dba213ef26e094c16c860ae7f4be0/v3_emdata_clahe_xy/jpeg",
    #     "bbox": [[1024, 1024, 0], [9216, 12288, 20569]],
    #     "n": 93,
    #     "stride": 192,
    #     # "chunk_size": [64, 64, 64],
    #     # Slab interfaces - xy is good
    # },
    # "nguyen_thomas_2022": {
    #     "src_res": [4, 4, 40],
    #     "dst_res": [32, 32, 40],
    #     "path": "https://s3.amazonaws.com/bossdb-open-data/nguyen_thomas2022/cb2/em",
    #     "bbox": [[0, 0, 0], [249600, 230400, 1200]],
    #     "n": 10,
    #     "stride": 1,
    #     # "chunk_size": [1024, 1024, 25],
    #     # Corrupt downsamples - use 4x4x40
    # },
    # # "maher_briegel_2023": {
    # #   "src_res": [5,5,75],
    # #   "path": "https://s3.amazonaws.com/bossdb-open-data/MaherBriegel2023/Lgn200/sbem"
    # #   # Sections too thick
    # # },
    # # "mulcahy_2022_1h": {
    # #   "src_res": [16, 16, 30],
    # #   "path": "https://s3.amazonaws.com/bossdb-open-data/mulcahy2022/1h_L1/em",
    # #   # Poor alignment
    # # },
    # "mulcahy_2022_16h": {
    #     "src_res": [32, 32, 30],
    #     "path": "https://s3.amazonaws.com/bossdb-open-data/mulcahy2022/16h_L1/em",
    #     "bbox": [[0, 0, 0], [7616, 2304, 1051]],
    #     "n": 490,
    #     "stride": 1,
    #     # "chunk_size": [256, 256, 32],
    # },
    # "wildenberg_2021_vta_dat12a": {
    #     "src_res": [32, 32, 40],
    #     "path": "https://s3.amazonaws.com/bossdb-open-data/wildenberg2021/VTA_dat12a_saline_control_Dendrites_6nm_aligned/image",
    #     "bbox": [[0, 0, 0], [2565, 2662, 191]],
    #     "n": 1258,  # all 191 sections
    #     "stride": 1,
    #     # "chunk_size": [512, 512, 16],
    # },
    # "bumbarber_2013": {
    #     "src_res": [31.2, 31.2, 50],
    #     "path": "https://s3.amazonaws.com/bossdb-open-data/neurodata/bumbarger/bumbarger13/image",
    #     "bbox": [[512, 512, 0], [2560, 2560, 2762]],
    #     "n": 2048,
    #     "stride": 1,
    #     # "chunk_size": [512, 512, 16],
    # },
    # "wilson_2019_p3": {
    #     "src_res": [32, 32, 30],
    #     "path": "https://s3.amazonaws.com/bossdb-open-data/wilson2019/P3/em",
    #     "bbox": [[0, 0, 0], [5120, 7168, 1657]],
    #     "n": 234,
    #     "stride": 1,
    #     # "chunk_size": [512, 512, 16],
    # },
    # # "ishibashi_2021_em1": {
    # #     "src_res": [16, 16, 4],
    # #     "path": "https://s3.amazonaws.com/bossdb-open-data/Ishibashi2021/EM1/em",
    # #     "bbox": [[0,0,0], [1536, 1024, 1136]],
    # #     "n": 21845,  # all 142 sections
    # #     "stride": 1,
    # #     # "chunk_size": [512, 512, 16],
    # #     # 32x32x4 downsampling is corrupt, also should take every 8th slice
    # # },
    # # "ishibashi_2021_em2": {
    # #     "src_res": [16, 16, 4],
    # #     "path": "https://s3.amazonaws.com/bossdb-open-data/Ishibashi2021/EM2/em",
    # #     "bbox": [[0,0,0], [1664, 1152, 1344]],
    # #     "n": 13443,  # all 168 sections
    # #     "stride": 1,
    # #     # "chunk_size": [512, 512, 16],
    # #     # 32x32x4 downsampling is corrupt, also should take every 8th slice
    # # },
    # "templier_2019_wafer1": {
    #     "src_res": [32, 32, 50],
    #     "path": "https://s3.amazonaws.com/bossdb-open-data/neurodata/templier/Wafer1/C1_EM",
    #     "bbox": [[0, 0, 0], [9216, 7168, 514]],
    #     "n": 130,
    #     "stride": 1,
    #     # "chunk_size": [64, 64, 64],
    # },
    # "templier_2019_wafer3": {
    #     "src_res": [32, 32, 50],
    #     "path": "https://s3.amazonaws.com/bossdb-open-data/neurodata/templier/Wafer3/EM",
    #     "bbox": [[0, 0, 0], [7168, 6144, 204]],
    #     "n": 195,
    #     "stride": 1,
    #     # "chunk_size": [64, 64, 64],
    # },
    "lichtman_octopus2022": {
        "src_res": [32, 32, 30],
        "path": "gs://octopus-connectomes/vertical_lobe/img",
        "bbox": [[2048, 1024, 0], [9216, 12288, 892]],
        "n": 106,
        "stride": 1,
        # "chunk_size": [64, 64, 64],
    }
}


for k, v in SOURCE_PATHS.items():
    cv = CloudVolume(v["path"], v["src_res"], use_https=True)
    bbox = Bbox(v["bbox"][0], v["bbox"][1])
    total_sections = (min(v["n"], int(bbox.size3()[2])) * int(cv.chunk_size[2])) / math.ceil(
        (int(cv.chunk_size[2]) / v["stride"])
    )
    total_chunks = math.ceil(total_sections / int(cv.chunk_size[2]))
    print(
        "Download: ",
        int(total_chunks),
        "chunks",
        k,
        int(bbox.size3()[0])
        * int(bbox.size3()[1])
        * int(cv.chunk_size[2])
        * total_chunks
        / 1024
        / 1024
        / 1024,
        "GiB",
    )
    print(cv.chunk_size)


flows = []
for k, v in SOURCE_PATHS.items():
    print(v["path"])
    if k in ["kronauer_cra9", "lichtman_zebrafish"]:
        continue
    # Check for src chunk size
    cv = CloudVolume(v["path"], v["src_res"], use_https=True)
    chunk_size_z = cv.chunk_size[2]
    bbox = Bbox(*v["bbox"])
    src_res = v["src_res"]
    dst_res = v.get("dst_res", src_res)
    scale_factor = int(round(dst_res[0] / src_res[0]))
    chunk_size_xy_adjust = 2 ** math.ceil(math.log(math.sqrt(chunk_size_z), 2))

    if v["stride"] == 1:
        # Copy continuous chunk, match processing chunk size to src chunk size
        processing_chunk_size = [
            max(1024, 32768 // (scale_factor * chunk_size_xy_adjust)),
            max(1024, 32768 // (scale_factor * chunk_size_xy_adjust)),
            chunk_size_z,
        ]
        size_z = int(min(v["n"], bbox.size3()[2]))
        start_z = int(
            max(
                bbox.minpt[2],
                bbox.minpt[2] + math.ceil(bbox.size3()[2] / 2) - math.ceil(size_z / 2),
            )
        )
        end_z = start_z + size_z + 1
        src_bboxes = [
            BBox3D.from_coords(
                start_coord=[int(bbox.minpt[0]), int(bbox.minpt[1]), start_z],
                end_coord=[int(bbox.maxpt[0]), int(bbox.maxpt[1]), end_z],
                resolution=v["src_res"],
            )
        ]
        dst_bboxes = [
            BBox3D.from_coords(
                start_coord=[0, 0, 0],
                end_coord=[
                    int(bbox.maxpt[0]) - int(bbox.minpt[0]),
                    int(bbox.maxpt[1]) - int(bbox.minpt[1]),
                    size_z,
                ],
                resolution=v["src_res"],
            )
        ]
    else:
        size_z = int(2 * v["n"])
        processing_chunk_size = [
            max(1024, 32768 // (scale_factor * chunk_size_xy_adjust)),
            max(1024, 32768 // (scale_factor * chunk_size_xy_adjust)),
            2,
        ]
        src_bboxes = [
            BBox3D.from_coords(
                start_coord=[int(bbox.minpt[0]), int(bbox.minpt[1]), start_z],
                end_coord=[int(bbox.maxpt[0]), int(bbox.maxpt[1]), start_z + 2],
                resolution=v["src_res"],
            )
            for start_z in range(int(bbox.minpt[2]), int(bbox.maxpt[2]), v["stride"])
        ]
        dst_bboxes = [
            BBox3D.from_coords(
                start_coord=[0, 0, start_z],
                end_coord=[
                    int(bbox.maxpt[0]) - int(bbox.minpt[0]),
                    int(bbox.maxpt[1]) - int(bbox.minpt[1]),
                    start_z + 2,
                ],
                resolution=v["src_res"],
            )
            for start_z in range(0, int(bbox.maxpt[2]) - int(bbox.minpt[2]), 2)
        ]

    flow = mazepa.concurrent_flow(
        [
            build_subchunkable_apply_flow(
                dst=build_cv_layer(
                    "gs://zetta-research-nico/encoder/datasets/" + k,
                    info_reference_path=v["path"],
                    on_info_exists="overwrite",
                    info_field_overrides={
                        "type": "image",
                        "num_channels": 1,
                        "data_type": "uint8",
                        "scales": [
                            {
                                "chunk_sizes": [[1024, 1024, 1]],
                                "resolution": dst_res,
                                "encoding": "raw",
                                "key": f"{dst_res[0]}_{dst_res[1]}_{dst_res[2]}",
                                "voxel_offset": [0, 0, 0],
                                "size": [
                                    int(bbox.size3()[0] // scale_factor),
                                    int(bbox.size3()[1] // scale_factor),
                                    size_z,
                                ],
                            }
                        ],
                    },
                    cv_kwargs={"delete_black_uploads": True},
                ),
                fn=efficient_parse_lambda_str(lambda_str="lambda src: src", name=f"Transfer {k}"),
                skip_intermediaries=True,
                dst_resolution=dst_res,
                processing_chunk_sizes=[processing_chunk_size],
                op_kwargs={
                    "src": build_cv_layer(
                        v["path"],
                        data_resolution=v["src_res"],
                        interpolation_mode="img",
                        cv_kwargs={"use_https": True},
                        index_procs=[
                            VolumetricIndexTranslator(
                                offset=[
                                    10
                                    * (
                                        src_bbox.start[0] - dst_bbox.start[0]
                                    ),  # Hack for decimal resolutions
                                    10
                                    * (
                                        src_bbox.start[1] - dst_bbox.start[1]
                                    ),  # Hack for decimal resolutions
                                    src_bbox.start[2] - dst_bbox.start[2],
                                ],
                                resolution=[0.1, 0.1, 1],  # Hack for decimal resolutions
                            )
                        ],
                    )
                },
                bbox=dst_bbox,
            )
            for (src_bbox, dst_bbox) in zip(src_bboxes, dst_bboxes)
        ]
    )
    flows.append(flow)


for k in SOURCE_PATHS.keys():
    cv = CloudVolume("precomputed://gs://zetta-research-nico/encoder/datasets/" + k)
    make_ng_link(
        layers=[(k, "image", "precomputed://gs://zetta-research-nico/encoder/datasets/" + k)],
        title=k,
        position=Vec3D(*cv.bounds.center().round()),
        scale_bar_nm=5000,
    )


import json
import os

os.environ["ZETTA_RUN_SPEC"] = json.dumps("")
execute_on_gcp_with_sqs(
    worker_image="us.gcr.io/zetta-research/zetta_utils:nico_py3.9_20230728_7",
    worker_resources={"memory": "27560Mi"},
    worker_replicas=50,
    worker_cluster_name="zutils-x3",
    worker_cluster_region="us-east1",
    worker_cluster_project="zetta-research",
    checkpoint_interval_sec=60,
    do_dryrun_estimation=True,
    # checkpoint="gs://zetta_utils_runs/nkem/exec-nice-sepia-wren-of-jest/2023-07-29_152159_7246.zstd",
    local_test=False,
    batch_gap_sleep_sec=0.1,
    target=mazepa.seq_flow(flows),
)
