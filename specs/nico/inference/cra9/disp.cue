import "math"

import "list"

#TARGET_ENC_PATH:         "gs://tmp_2w/nico/cra8/enc/z0_cns"
#ALIGNED_FIELD_PATH:      "gs://tmp_2w/nico/cra8/field_fwd_align_cns"
#MISALIGNED_FIELD_PATH:   "gs://tmp_2w/nico/cra8/bad_field_fwd_align_cns"
#DST_FIELD_PATH:          "gs://zetta-research-nico/encoder/misd/misalignment_fields_cns/"

#DST_INFO_CHUNK_SIZE: [2048, 2048, 1] // Will automatically get truncated if dataset becomes too small
#MAX_TASK_SIZE: [8192, 8192, 1]
#HIGH_RES: [32, 32, 42]

#DATASET_BOUNDS: [
	[0 * #HIGH_RES[0], 12800 * #HIGH_RES[0]],
	[0 * #HIGH_RES[1], 12032 * #HIGH_RES[1]],
	[0 * #HIGH_RES[2], 6112 * #HIGH_RES[2]],
]

#ROI_BOUNDS: [
	[0 * #HIGH_RES[0], 12800 * #HIGH_RES[0]],
	[0 * #HIGH_RES[1], 12032 * #HIGH_RES[1]],
	[4000 * #HIGH_RES[2], 4020 * #HIGH_RES[2]],
]

"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-research"
worker_cluster_name:    "zutils-x3"
worker_image:           "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:nico_py3.9_20231212"
worker_resource_requests: {
	memory: "10000Mi"
}
worker_replicas: 100
local_test:      false
target: {
	"@type": "mazepa.sequential_flow"
	stages: [
		{
			"@type": "mazepa.concurrent_flow"
			stages: [
				for i in [2] for path in [#ALIGNED_FIELD_PATH] {
					#DOWNSAMPLE_FIELD_TEMPLATE & {
						_path:          path +  "/z\(i)_composed"
						src_resolution: #HIGH_RES
					}
				},
			]
		},
		{
			"@type": "mazepa.concurrent_flow"
			stages: [
				for i in [2] for res in [#HIGH_RES, [#HIGH_RES[0] * 2, #HIGH_RES[1] * 2, #HIGH_RES[2]]] {
					#FIELD_DIFF_TEMPLATE & {
						_z_offset:      i
						dst_resolution: res
					}
				},
			]
		},
	]
}

#DOWNSAMPLE_FIELD_TEMPLATE: {
	_path:     string

	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1]-#ROI_BOUNDS[0][0])/#DST_INFO_CHUNK_SIZE[0]/dst_resolution[0]) * #DST_INFO_CHUNK_SIZE[0]]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1]-#ROI_BOUNDS[1][0])/#DST_INFO_CHUNK_SIZE[1]/dst_resolution[1]) * #DST_INFO_CHUNK_SIZE[1]]),
		1,
	]

	"@type": "build_interpolate_flow"
	mode:    "field"
	src_resolution: [number, number, number]
	dst_resolution: [src_resolution[0] * 2, src_resolution[1] * 2, src_resolution[2]]
	chunk_size: max_chunk_size
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [#ROI_BOUNDS[0][0], #ROI_BOUNDS[1][0], #ROI_BOUNDS[2][0]]
		end_coord: [#ROI_BOUNDS[0][1], #ROI_BOUNDS[1][1], #ROI_BOUNDS[2][1]]
	}

	src: {
		"@type": "build_cv_layer"
		path:    _path
	}
	dst: {
		"@type": "build_cv_layer"
		path:    _path
	}

}

#FIELD_DIFF_TEMPLATE: {
	_z_offset: int

	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1]-#ROI_BOUNDS[0][0])/#DST_INFO_CHUNK_SIZE[0]/dst_resolution[0]) * #DST_INFO_CHUNK_SIZE[0]]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1]-#ROI_BOUNDS[1][0])/#DST_INFO_CHUNK_SIZE[1]/dst_resolution[1]) * #DST_INFO_CHUNK_SIZE[1]]),
		1,
	]

	"@type": "build_subchunkable_apply_flow"
	fn: {
		"@type": "torch.sub", "@mode": "partial"
	}
	processing_chunk_sizes: [max_chunk_size]
	processing_crop_pads: [[0, 0, 0]]
	dst_resolution:         _
	expand_bbox_resolution: true
	skip_intermediaries:    true
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [#ROI_BOUNDS[0][0], #ROI_BOUNDS[1][0], #ROI_BOUNDS[2][0]]
		end_coord: [#ROI_BOUNDS[0][1], #ROI_BOUNDS[1][1], #ROI_BOUNDS[2][1]]
	}
	op_kwargs: {
		input: {
			"@type": "build_cv_layer"
			path:    #ALIGNED_FIELD_PATH + "/z\(_z_offset)_composed"
		}
		other: {
			"@type": "build_cv_layer"
			path:    #MISALIGNED_FIELD_PATH + "/z\(_z_offset)"
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_FIELD_PATH + "kronauer_cra9_defects/displacements/z\(_z_offset)"
		info_reference_path: #TARGET_ENC_PATH
		info_field_overrides: {
			data_type: "uint8"
		}
		on_info_exists: "overwrite"
		write_procs: [
			{
				"@type":    "lambda"
				lambda_str: "lambda data: (data.norm(dim=0, keepdim=True)*10.0).round().clamp(0, 255).byte()"
			},
		]
	}
}
