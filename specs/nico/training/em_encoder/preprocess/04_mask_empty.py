# type: ignore
# pylint: skip-file
from __future__ import annotations

import json
import math
import os
from functools import partial

from cloudvolume import CloudVolume

from zetta_utils.api.v0 import *

SOURCE_PATHS = {
    # "microns_pinky": {"contiguous": True},
    # "microns_basil": {"contiguous": True},
    # "microns_minnie": {"contiguous": False},
    # "microns_interneuron": {"contiguous": False},
    # "aibs_v1dd": {"contiguous": False},
    "kim_n2da": {"contiguous": True},
    # "kim_pfc2022": {"contiguous": True},
    # "kronauer_cra9": {"contiguous": True},
    # "kubota_001": {"contiguous": True},
    # "lee_fanc": {"contiguous": False},
    # "lee_banc": {"contiguous": False},
    # "lee_ppc": {"contiguous": True},
    # "lee_mosquito": {"contiguous": False},
    # "lichtman_zebrafish": {"contiguous": False},
    # "prieto_godino_larva": {"contiguous": True},
    # "fafb_v15": {"contiguous": False},
    # "lichtman_h01": {"contiguous": False},
    # "janelia_hemibrain": {"contiguous": True},
    # "janelia_manc": {"contiguous": False},
    # "nguyen_thomas_2022": {"contiguous": True},
    "mulcahy_2022_16h": {"contiguous": True},
    # "wildenberg_2021_vta_dat12a": {"contiguous": True},
    "bumbarber_2013": {"contiguous": True},
    # "wilson_2019_p3": {"contiguous": True},
    # "ishibashi_2021_em1": {"contiguous": True},
    # "ishibashi_2021_em2": {"contiguous": True},
    # "templier_2019_wafer1": {"contiguous": True},
    # "templier_2019_wafer3": {"contiguous": True},
    # "lichtman_octopus2022": {"contiguous": True},
}

BASE_PATH = "gs://zetta-research-nico/encoder/"

concurrent_mask_flows = []
for k, v in SOURCE_PATHS.items():
    img_tgt_path = BASE_PATH + "datasets/" + k
    img_src_path = BASE_PATH + "pairwise_aligned/" + k + "/warped_img"
    misd_mask_thr_path = BASE_PATH + "pairwise_aligned/" + k + "/misd_mask_thr"

    cv_src_img = CloudVolume(img_tgt_path, progress=False)
    bounds = cv_src_img.bounds
    resolution = cv_src_img.resolution.tolist()
    minpt = bounds.minpt.tolist()
    maxpt = bounds.maxpt.tolist()
    size = bounds.size3().tolist()

    if v["contiguous"]:
        z_ranges = [(minpt[2], maxpt[2] + 1)]
    else:
        z_ranges = [(z, z + 1) for z in range(minpt[2], maxpt[2], 2)]

    for z_start, z_end in z_ranges:
        mask_flow = build_subchunkable_apply_flow(
            dst=build_cv_layer(
                misd_mask_thr_path,
                write_procs=[
                    partial(to_uint8),
                ],
            ),
            fn=efficient_parse_lambda_str(
                lambda_str="lambda src: (src['src']==0) | (src['tgt']==0) | (src['misd']!=0)",
                name=f"Downsample Warped Mask",
            ),
            skip_intermediaries=True,
            dst_resolution=[resolution[0] * 1024, resolution[1] * 1024, resolution[2]],
            processing_chunk_sizes=[[math.ceil(size[0] / 1024.0), math.ceil(size[1] / 1024.0), 1]],
            processing_crop_pads=[[0, 0, 0]],
            op_kwargs={
                "src": build_layer_set(
                    {
                        "src": build_cv_layer(
                            img_src_path,
                            data_resolution=[resolution[0] * 4, resolution[1] * 4, resolution[2]],
                            interpolation_mode="mask",
                        ),
                        "tgt": build_cv_layer(
                            img_tgt_path,
                            data_resolution=[resolution[0] * 4, resolution[1] * 4, resolution[2]],
                            interpolation_mode="mask",
                        ),
                        "misd": build_cv_layer(
                            misd_mask_thr_path,
                        ),
                    }
                )
            },
            bbox=BBox3D.from_coords(
                start_coord=[minpt[0], minpt[1], z_start],
                end_coord=[maxpt[0], maxpt[1], z_end],
                resolution=resolution,
            ),
            expand_bbox_processing=True,
            expand_bbox_resolution=True,
        )
        concurrent_mask_flows.append(mask_flow)


os.environ["ZETTA_RUN_SPEC"] = json.dumps("")
execute_on_gcp_with_sqs(
    worker_image="us.gcr.io/zetta-research/zetta_utils:nico_py3.9_20230816",
    worker_resources={"memory": "13000Mi"},
    worker_replicas=50,
    worker_cluster_name="zutils-x3",
    worker_cluster_region="us-east1",
    worker_cluster_project="zetta-research",
    checkpoint_interval_sec=60,
    do_dryrun_estimation=True,
    local_test=False,
    batch_gap_sleep_sec=0.1,
    target=concurrent_flow([concurrent_mask_flows]),
)
