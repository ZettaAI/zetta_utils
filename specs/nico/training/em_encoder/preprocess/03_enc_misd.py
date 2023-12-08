# type: ignore
# pylint: skip-file
from __future__ import annotations

import json
import math
import os
from copy import deepcopy
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
ENCODER_MODEL = "gs://zetta-research-nico/training_artifacts/base_enc_zfish/1.4.0_M3_M3_unet4_lr0.0001_equi0.5_post1.6-1.6_fmt0.8_zfish/last.ckpt.model.spec.json"
MISD_MODEL = "gs://zetta-research-nico/training_artifacts/aced_misd_cns_zfish/thr5.0_lr0.00001_zfish_finetune_2/last.ckpt.static-2.0.0+cu117-model.jit"

concurrent_enc_flows = []
concurrent_misd_flows = []
concurrent_img_ds_flows = []
concurrent_mask_ds_flows = []
for k, v in SOURCE_PATHS.items():
    img_tgt_path = BASE_PATH + "datasets/" + k
    img_src_path = BASE_PATH + "pairwise_aligned/" + k + "/warped_img"
    enc_tgt_path = BASE_PATH + "pairwise_aligned/" + k + "/tgt_enc"
    enc_src_path = BASE_PATH + "pairwise_aligned/" + k + "/warped_enc"
    misd_mask_path = BASE_PATH + "pairwise_aligned/" + k + "/misd_mask"
    misd_mask_thr_path = BASE_PATH + "pairwise_aligned/" + k + "/misd_mask_thr"

    cv_src_img = CloudVolume(img_tgt_path, progress=False)
    bounds = cv_src_img.bounds
    resolution = cv_src_img.resolution.tolist()
    minpt = bounds.minpt.tolist()
    maxpt = bounds.maxpt.tolist()
    size = bounds.size3().tolist()

    enc_ref = cv_src_img.info
    enc_ref["data_type"] = "int8"
    enc_ref["scales"] = enc_ref["scales"][:1]

    mask_ref = deepcopy(enc_ref)
    mask_ref["data_type"] = "uint8"

    mask_thresh_ref = deepcopy(mask_ref)
    mask_thresh_ref["scales"][0]["size"] = [math.ceil(maxpt[0] / 2.0**10), math.ceil(maxpt[1] / 2.0**10), maxpt[2]]
    mask_thresh_ref["scales"][0]["chunk_sizes"] = [[math.ceil(maxpt[0] / 2.0**10), math.ceil(maxpt[1] / 2.0**10), 1]]
    mask_thresh_ref["scales"][0]["resolution"] = [resolution[0] * 2**10, resolution[1] * 2**10, resolution[2]]
    mask_thresh_ref["scales"][0]["key"] = f"{resolution[0] * 2**10}_{resolution[1] * 2**10}_{resolution[2]}"

    if v["contiguous"]:
        z_ranges = [(minpt[2], maxpt[2] + 1)]
    else:
        z_ranges = [(z, z + 1) for z in range(minpt[2], maxpt[2], 2)]

    superchunk_size = [
        min(8192, 2048 * math.ceil(size[0] / 2048)),
        min(8192, 2048 * math.ceil(size[1] / 2048)),
        1
    ]

    for z_start, z_end in z_ranges:
        # for img_path, enc_path in [(img_src_path, enc_src_path), (img_tgt_path, enc_tgt_path)]:
        #     enc_flow = build_subchunkable_apply_flow(
        #         dst=build_cv_layer(
        #             enc_path,
        #             cv_kwargs={"delete_black_uploads": True},
        #             info_field_overrides=enc_ref,
        #         ),
        #         op=VolumetricCallableOperation(
        #             fn=BaseEncoder(ENCODER_MODEL),
        #             crop_pad=[0, 0, 0],
        #             res_change_mult=[1, 1, 1],
        #         ),
        #         skip_intermediaries=True,
        #         dst_resolution=resolution,
        #         processing_chunk_sizes=[superchunk_size, [2048, 2048, 1]],
        #         processing_crop_pads=[[0, 0, 0], [32, 32, 0]],
        #         op_kwargs={
        #             "src": build_cv_layer(
        #                 img_path,
        #             ),
        #         },
        #         bbox=BBox3D.from_coords(
        #             start_coord=[minpt[0], minpt[1], z_start],
        #             end_coord=[maxpt[0], maxpt[1], z_end],
        #             resolution=resolution,
        #         ),
        #         expand_bbox_processing=True,
        #     )
        #     concurrent_enc_flows.append(enc_flow)

        # misd_flow = build_subchunkable_apply_flow(
        #     dst=build_cv_layer(
        #         misd_mask_path,
        #         cv_kwargs={"delete_black_uploads": True},
        #         info_field_overrides=mask_ref,
        #     ),
        #     op=VolumetricCallableOperation(
        #         fn=MisalignmentDetector(MISD_MODEL),
        #     ),
        #     skip_intermediaries=True,
        #     dst_resolution=resolution,
        #     processing_chunk_sizes=[superchunk_size, [2048, 2048, 1]],
        #     processing_crop_pads=[[0, 0, 0], [64, 64, 0]],
        #     op_kwargs={
        #         "src": build_cv_layer(
        #             enc_src_path,
        #         ),
        #         "tgt": build_cv_layer(
        #             enc_tgt_path,
        #         ),
        #     },
        #     bbox=BBox3D.from_coords(
        #         start_coord=[minpt[0], minpt[1], z_start],
        #         end_coord=[maxpt[0], maxpt[1], z_end],
        #         resolution=resolution,
        #     ),
        #     expand_bbox_processing=True,
        # )
        # concurrent_misd_flows.append(misd_flow)


        # seq_ds_flows = []
        # for src_res in [[resolution[0] * 2**factor, resolution[1] * 2**factor, resolution[2]] for factor in range(0, 2)]:
        #     dst_res = [2 * src_res[0], 2 * src_res[1], src_res[2]]
        #     ds_flow = build_subchunkable_apply_flow(
        #         dst=build_cv_layer(
        #             img_src_path,
        #             cv_kwargs={"delete_black_uploads": True},
        #         ),
        #         fn=efficient_parse_lambda_str(lambda_str="lambda src: src", name=f"Downsample Warped Img"),
        #         skip_intermediaries=True,
        #         dst_resolution=dst_res,
        #         processing_chunk_sizes=[superchunk_size, [2048, 2048, 1]],
        #         processing_crop_pads=[[0, 0, 0], [0, 0, 0]],
        #         op_kwargs={
        #             "src": build_cv_layer(
        #                 img_src_path,
        #                 data_resolution=src_res,
        #                 interpolation_mode="img",
        #             ),
        #         },
        #         bbox=BBox3D.from_coords(
        #             start_coord=[minpt[0], minpt[1], z_start],
        #             end_coord=[maxpt[0], maxpt[1], z_end],
        #             resolution=resolution,
        #         ),
        #         expand_bbox_resolution=True,
        #     )
        #     seq_ds_flows.append(ds_flow)
        # concurrent_img_ds_flows.append(sequential_flow(seq_ds_flows))

        seq_ds_flows = []
        dst_res = [resolution[0] * 2**10, resolution[1] * 2**10, resolution[2]]
        ds_flow = build_subchunkable_apply_flow(
            dst=build_cv_layer(
                misd_mask_thr_path,
                info_field_overrides=mask_thresh_ref,
                data_resolution=dst_res,
                interpolation_mode="nearest",
            ),
            fn=efficient_parse_lambda_str(lambda_str="lambda src: src", name=f"Downsample Warped Mask"),
            skip_intermediaries=True,
            dst_resolution=resolution,
            processing_chunk_sizes=[[math.ceil(maxpt[0]/1024.0)*1024, math.ceil(maxpt[1]/1024.0)*1024, 1]],
            processing_crop_pads=[[0, 0, 0]],
            op_kwargs={
                "src": build_cv_layer(
                    misd_mask_path,
                    read_procs=[
                        partial(rearrange, pattern="C X Y Z -> Z C X Y"),
                        partial(compare, mode=">=", value=32, binarize=True),
                        partial(to_float32),
                        partial(interpolate, scale_factor=[1.0/2**10, 1.0/2**10], mode="area", unsqueeze_input_to=4),
                        partial(compare, mode=">", value=0.1, binarize=True),
                        partial(to_uint8),
                        partial(interpolate, scale_factor=[2**10, 2**10], mode="nearest", unsqueeze_input_to=4),
                        partial(rearrange, pattern="Z C X Y -> C X Y Z"),
                    ],
                ),
            },
            bbox=BBox3D.from_coords(
                start_coord=[minpt[0], minpt[1], z_start],
                end_coord=[maxpt[0], maxpt[1], z_end],
                resolution=resolution,
            ),
            expand_bbox_processing=True,
            expand_bbox_resolution=True,
        )
        concurrent_mask_ds_flows.append(ds_flow)



os.environ["ZETTA_RUN_SPEC"] = json.dumps("")
# execute_on_gcp_with_sqs(
#     worker_image="us.gcr.io/zetta-research/zetta_utils:nico_py3.9_20230809",
#     worker_resources={"memory": "17560Mi", "nvidia.com/gpu": 1},
#     worker_replicas=30,
#     worker_cluster_name="zutils-x3",
#     worker_cluster_region="us-east1",
#     worker_cluster_project="zetta-research",
#     checkpoint_interval_sec=60,
#     do_dryrun_estimation=True,
#     local_test=False,
#     batch_gap_sleep_sec=0.1,
#     target=sequential_flow([
#         concurrent_flow(concurrent_enc_flows),
#         concurrent_flow(concurrent_misd_flows),
        
#     ])
# )

# breakpoint()
# execute_on_gcp_with_sqs(
#     worker_image="us.gcr.io/zetta-research/zetta_utils:nico_py3.9_20230809",
#     worker_resources={"memory": "13000Mi"},
#     worker_replicas=50,
#     worker_cluster_name="zutils-x3",
#     worker_cluster_region="us-east1",
#     worker_cluster_project="zetta-research",
#     checkpoint_interval_sec=60,
#     do_dryrun_estimation=True,
#     local_test=True,
#     batch_gap_sleep_sec=0.1,
#     target=concurrent_flow([
#         concurrent_mask_ds_flows
#     ]),
# )

for k in SOURCE_PATHS.keys():
    cv = CloudVolume("precomputed://gs://zetta-research-nico/encoder/datasets/" + k)
    link = make_ng_link(
        layers=[
            ("tgt", "image", f"precomputed://gs://zetta-research-nico/encoder/datasets/{k}"),
            ("src", "image", f"precomputed://gs://zetta-research-nico/encoder/pairwise_aligned/{k}/warped_img"),
            ("misd", "image", f"precomputed://gs://zetta-research-nico/encoder/pairwise_aligned/{k}/misd_mask"),
            ("bad_chunks", "segmentation", f"precomputed://gs://zetta-research-nico/encoder/pairwise_aligned/{k}/misd_mask_thr"),
            (f"CREATE:zetta-research-nico/encoder/pairwise_aligned/{k}", "annotation", None),
        ],
        title=k,
        position=Vec3D(*(cv.bounds.center().round()[:2]), 0),
        scale_bar_nm=30000,
        print_to_logger=False
    )

    print(f"{k}: {link}")