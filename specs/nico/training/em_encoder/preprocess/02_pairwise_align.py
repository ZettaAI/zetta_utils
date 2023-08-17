from __future__ import annotations

import math
from functools import partial

from cloudvolume import CloudVolume


from zetta_utils.api.v0 import *

SOURCE_PATHS = {
    # # # "microns_pinky": {"contiguous": True},
    # # # "microns_basil": {"contiguous": True},
    # # # "microns_minnie": {"contiguous": False},
    # # # "microns_interneuron": {"contiguous": False},
    "aibs_v1dd": {"contiguous": False},
    # # # "kim_n2da": {"contiguous": True},
    # # # "kim_pfc2022": {"contiguous": True},
    # # # "kronauer_cra9": {"contiguous": True},
    # # # "kubota_001": {"contiguous": True},
    "lee_fanc": {"contiguous": False},
    # # # "lee_banc": {"contiguous": False},
    # # # "lee_ppc": {"contiguous": True},
    # # # "lee_mosquito": {"contiguous": False},
    # # # "lichtman_zebrafish": {"contiguous": False},
    # # # "prieto_godino_larva": {"contiguous": True},
    # # # "fafb_v15": {"contiguous": False},
    # # # "lichtman_h01": {"contiguous": False},
    # # # "janelia_hemibrain": {"contiguous": True},
    # # # "janelia_manc": {"contiguous": False},
    # # # "nguyen_thomas_2022": {"contiguous": True},
    # # # "mulcahy_2022_16h": {"contiguous": True},
    # # # "wildenberg_2021_vta_dat12a": {"contiguous": True},
    # # # "bumbarber_2013": {"contiguous": True},
    # # # "wilson_2019_p3": {"contiguous": True},
    # # # "ishibashi_2021_em1": {"contiguous": True},
    # # # "ishibashi_2021_em2": {"contiguous": True},
    # # # "templier_2019_wafer1": {"contiguous": True},
    # # # "templier_2019_wafer3": {"contiguous": True},
    # # # "lichtman_octopus2022": {"contiguous": True},
}


BASE_PATH = "gs://zetta-research-nico/encoder/"
IMG_SRC_PATH = BASE_PATH + "datasets/"
DST_PATH = BASE_PATH + "pairwise_aligned/"
TMP_FIELD_PATH = BASE_PATH + "tmp/pairwise_aligned_fields/"


concurrent_cf_flows = []
concurrent_warp_flows = []
tasks_count = {}
for k, v in SOURCE_PATHS.items():
    src_img_path = IMG_SRC_PATH + k
    dst_field_path = DST_PATH + k + "/field"
    dst_img_path = DST_PATH + k + "/warped_img"
    dst_enc_path = DST_PATH + k + "/warped_enc"

    cv_src_img = CloudVolume(src_img_path, progress=False)


    bounds = cv_src_img.bounds
    resolution = cv_src_img.resolution.tolist()
    minpt = bounds.minpt.tolist()
    maxpt = bounds.maxpt.tolist()
    size = bounds.size3().tolist()

    field_ref = cv_src_img.info
    field_ref["data_type"] = "float32"
    field_ref["num_channels"] = 2
    for i in range(3):
        field_ref["scales"][i].update(
            {
                "encoding": "zfpc",
                "zfpc_correlated_dims": [True, True, False, False],
                "zfpc_tolerance": 1 / 512,
            }
        )

    # ds_pyramid = []
    # src_res = resolution
    # dst_res = [src_res[0] * 2, src_res[1] * 2, src_res[2]]

    # for i in range(2):
    #     ds_pyramid.append(
    #         build_subchunkable_apply_flow(
    #             dst=build_cv_layer(IMG_SRC_PATH + k, cv_kwargs={"delete_black_uploads": True}),
    #             fn=efficient_parse_lambda_str(
    #                 lambda_str="lambda src: src", name=f"Downsample {k}"
    #             ),
    #             skip_intermediaries=True,
    #             dst_resolution=dst_res,
    #             processing_chunk_sizes=[[8192, 8192, 1]],
    #             op_kwargs={
    #                 "src": build_cv_layer(
    #                     IMG_SRC_PATH + k,
    #                     data_resolution=src_res,
    #                     interpolation_mode="img",
    #                 )
    #             },
    #             bbox=BBox3D.from_coords(
    #                 start_coord=[0, 0, 0],
    #                 end_coord=[size[0], size[1], size[2]],
    #                 resolution=dst_res,
    #             ),
    #             expand_bbox_processing=True,
    #         )
    #     )
    #     src_res = dst_res
    #     dst_res = [src_res[0] * 2, src_res[1] * 2, src_res[2]]
    #     size = [size[0] // 2, size[1] // 2, size[2]]

    # concurrent_flows.append(mazepa.seq_flow(ds_pyramid))

    if v["contiguous"]:
        z_ranges = [(minpt[2], maxpt[2] + 1)]
    else:
        z_ranges = [(z, z + 1) for z in range(minpt[2], maxpt[2], 2)]

    tasks_count[k] = {"cf": [0, 0, 0], "warp": 0}
    for z_start, z_end in z_ranges[1:]:
        # tasks_count[k]["cf"][0] += (16384 * math.ceil(0.25 * size[0] / 16384.0) * 16384 * math.ceil(0.25 * size[1] / 16384.0) * (z_end - z_start)) / (16384 * 16384)
        # tasks_count[k]["cf"][1] += (16384 * math.ceil(0.5 * size[0] / 16384.0) * 16384 * math.ceil(0.5 * size[1] / 16384.0) * (z_end - z_start)) / (16384 * 16384)
        # tasks_count[k]["cf"][2] += (16384 * math.ceil(size[0] / 16384.0) * 16384 * math.ceil(size[1] / 16384.0) * (z_end - z_start)) / (16384 * 16384)
        # tasks_count[k]["warp"]  += (16384 * math.ceil(size[0] / 16384.0) * 16384 * math.ceil(size[1] / 16384.0) * (z_end - z_start)) / (16384 * 16384)
        lvl0_sizes = [
            [
                min(8192, 2048 * math.ceil(size[0] / 2048)),
                min(8192, 2048 * math.ceil(size[1] / 2048)),
                1
            ],
            [
                min(8192, 2048 * math.ceil(size[0] / 2048)),
                min(8192, 2048 * math.ceil(size[1] / 2048)),
                1
            ],
            [
                min(8192, 2048 * math.ceil(size[0] / 2048)),
                min(8192, 2048 * math.ceil(size[1] / 2048)),
                1
            ],
            [
                min(16384, 2048 * math.ceil(size[0] / 2048)),
                min(16384, 2048 * math.ceil(size[1] / 2048)),
                1
            ],
        ]


        compute_field_flow = build_compute_field_multistage_flow(
            stages=[
                ComputeFieldStage(
                    fn=partial(align_with_online_finetuner, sm=10, num_iter=300, lr=0.1),
                    dst_resolution=resolution,
                    processing_chunk_sizes=[lvl0_sizes[0], [2048, 2048, 1]],
                    processing_crop_pads=[[0, 0, 0], [64, 64, 0]],
                    expand_bbox_processing=True,
                ),
                ComputeFieldStage(
                    fn=partial(align_with_online_finetuner, sm=10, num_iter=300, lr=0.1),
                    dst_resolution=resolution,
                    processing_chunk_sizes=[lvl0_sizes[1], [2048, 2048, 1]],
                    processing_crop_pads=[[0, 0, 0], [64, 64, 0]],
                    expand_bbox_processing=True,
                ),
                ComputeFieldStage(
                    fn=partial(align_with_online_finetuner, sm=10, num_iter=200, lr=0.1),
                    dst_resolution=resolution,
                    processing_chunk_sizes=[lvl0_sizes[2], [2048, 2048, 1]],
                    processing_crop_pads=[[0, 0, 0], [64, 64, 0]],
                    expand_bbox_processing=True,
                )
            ],
            bbox=BBox3D.from_coords(
                start_coord=[minpt[0], minpt[1], z_start],
                end_coord=[maxpt[0], maxpt[1], z_end],
                resolution=resolution
            ),
            src_offset=[0, 0, 1],
            tgt_offset=[0, 0, 0],
            offset_resolution=resolution,
            src=build_cv_layer(src_img_path),
            tgt=build_cv_layer(src_img_path),
            dst=build_cv_layer(
                DST_PATH + k + "/field",
                info_field_overrides=field_ref,
            ),
            tmp_layer_dir=TMP_FIELD_PATH + k,
            tmp_layer_factory=partial(build_cv_layer, info_field_overrides=field_ref)
        )
        concurrent_cf_flows.append(compute_field_flow)


        warp_img_flow = build_subchunkable_apply_flow(
            dst=build_cv_layer(
                dst_img_path,
                cv_kwargs={"delete_black_uploads": True},
                info_reference_path=src_img_path,

            ),
            op=WarpOperation(mode="img"),
            skip_intermediaries=True,
            dst_resolution=resolution,
            processing_chunk_sizes=[lvl0_sizes[3], [2048, 2048, 1]],
            processing_crop_pads=[[0, 0, 0], [256, 256, 0]],
            op_kwargs={
                "src": build_cv_layer(
                    src_img_path,
                    index_procs=[
                        VolumetricIndexTranslator(
                            offset=[0, 0, 1],
                            resolution=resolution
                        )
                    ],
                ),
                "field": build_cv_layer(dst_field_path)
            },
            bbox=BBox3D.from_coords(
                start_coord=[minpt[0], minpt[1], z_start],
                end_coord=[maxpt[0], maxpt[1], z_end],
                resolution=resolution,
            ),
            expand_bbox_processing=True,
        )
        concurrent_warp_flows.append(warp_img_flow)


import json
import os

os.environ["ZETTA_RUN_SPEC"] = json.dumps("")
execute_on_gcp_with_sqs(
    worker_image="us.gcr.io/zetta-research/zetta_utils:nico_py3.9_20230809",
    worker_resources={"memory": "17560Mi", "nvidia.com/gpu": 1},
    worker_replicas=300,
    worker_cluster_name="zutils-x3",
    worker_cluster_region="us-east1",
    worker_cluster_project="zetta-research",
    checkpoint_interval_sec=60,
    do_dryrun_estimation=True,
    local_test=False,
    batch_gap_sleep_sec=0.1,
    # checkpoint="gs://zetta_utils_runs/nkem/exec-smart-caracara-of-strange-progress/2023-08-04_235232_9768.zstd",
    target=concurrent_flow(concurrent_cf_flows),
)

execute_on_gcp_with_sqs(
    worker_image="us.gcr.io/zetta-research/zetta_utils:nico_py3.9_20230809", #03_2
    worker_resources={"memory": "13000Mi"},
    worker_replicas=50,
    worker_cluster_name="zutils-x3",
    worker_cluster_region="us-east1",
    worker_cluster_project="zetta-research",
    checkpoint_interval_sec=60,
    do_dryrun_estimation=True,
    local_test=False,
    batch_gap_sleep_sec=0.1,
    target=concurrent_flow(concurrent_warp_flows),
)
