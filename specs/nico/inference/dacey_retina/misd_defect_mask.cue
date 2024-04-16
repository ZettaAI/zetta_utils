import "math"

import "list"


#ALIGNED_MISD_Z1_PATH:                 "gs://dacey-human-retina-001-alignment-temp/pairwise/misd_raw/z-1"
#ALIGNED_MISD_Z2_PATH:                 "gs://dacey-human-retina-001-alignment-temp/pairwise/misd_raw/z-2"
#ALIGNED_MISD_GARBAGE_PATH:            "gs://dacey-human-retina-001-alignment-temp/pairwise/misd_raw/garbage"

#DST_INFO_CHUNK_SIZE: [2048, 2048, 1] // Will automatically get truncated if dataset becomes too small
#MAX_TASK_SIZE: [8192, 8192, 1]

#HIGH_RES: [40, 40, 50]

#DATASET_BOUNDS: [
	[0 * #HIGH_RES[0], 8192 * #HIGH_RES[0]],
	[0 * #HIGH_RES[1], 8192 * #HIGH_RES[1]],
	[1 * #HIGH_RES[2], 3030 * #HIGH_RES[2]],
]

#ROI_BOUNDS: [
	[0 * #HIGH_RES[0], 8192 * #HIGH_RES[0]],
	[0 * #HIGH_RES[1], 8192 * #HIGH_RES[1]],
	[1 * #HIGH_RES[2], 3030 * #HIGH_RES[2]],
]

"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_cluster_region:  "us-east1"
worker_cluster_project: "dacey-human-retina-001"
worker_cluster_name:    "zutils"
worker_image:           "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:nico_py3.9_20240122"
// worker_resources: {
// 	"nvidia.com/gpu": "1"
// }
worker_resource_requests: {
	memory: "21000Mi"       // sized for n1-highmem-4
}
worker_replicas: 100
local_test:      true
debug: true

target: {
	#CREATE_GARBAGE_MASK & {
		dst_resolution: [640,640,50]
	}
}

#CREATE_GARBAGE_MASK: {
	let max_cv_chunk_size = [
		list.Min([#DST_INFO_CHUNK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1] - #ROI_BOUNDS[0][0]) / dst_resolution[0])]),
		list.Min([#DST_INFO_CHUNK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1] - #ROI_BOUNDS[1][0]) / dst_resolution[1])])
	]
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1]-#ROI_BOUNDS[0][0])/max_cv_chunk_size[0]/dst_resolution[0]) * max_cv_chunk_size[0]]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1]-#ROI_BOUNDS[1][0])/max_cv_chunk_size[1]/dst_resolution[1]) * max_cv_chunk_size[1]]),
		1,
	]

	"@type": "build_subchunkable_apply_flow"
	fn: {
		"@type": "lambda"
		lambda_str: "lambda **kw: kw['z1'] * kw['z2'] * kw['z1_1'] * kw['z2_2']"
	}
	dst_resolution: _
	processing_chunk_sizes: [[max_chunk_size[0], max_chunk_size[1], 3029]]
	processing_crop_pads: [[0, 0, 0]]
	skip_intermediaries:    true
	expand_bbox_processing: true
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [#ROI_BOUNDS[0][0], #ROI_BOUNDS[1][0], #ROI_BOUNDS[2][0]]
		end_coord: [#ROI_BOUNDS[0][1], #ROI_BOUNDS[1][1], #ROI_BOUNDS[2][1]]
	}
	op_kwargs: {
		// "@type": "build_layer_set"
		// layers: {
			"z1": {
				"@type": "build_cv_layer"
				path:    #ALIGNED_MISD_Z1_PATH
				read_procs: [
					{
						"@type": "compare"
						"@mode": "partial"
						mode:    ">="
						value:   127
					},
					{
						"@type": "to_uint8"
						"@mode": "partial"
					},
				]
			},
			"z1_1": {
				"@type": "build_cv_layer"
				path:    #ALIGNED_MISD_Z1_PATH
				index_procs: [{
					"@type": "VolumetricIndexTranslator"
					offset: [0, 0, 1]
					resolution: dst_resolution
				}]
				read_procs: [
					{
						"@type": "compare"
						"@mode": "partial"
						mode:    ">="
						value:   127
					},
					{
						"@type": "to_uint8"
						"@mode": "partial"
					},
				]
			},
			"z2": {
				"@type": "build_cv_layer"
				path:    #ALIGNED_MISD_Z2_PATH
				read_procs: [
					{
						"@type": "compare"
						"@mode": "partial"
						mode:    ">="
						value:   127
					},
					{
						"@type": "to_uint8"
						"@mode": "partial"
					},
				]
			},
			"z2_2": {
				"@type": "build_cv_layer"
				path:    #ALIGNED_MISD_Z2_PATH
				index_procs: [{
					"@type": "VolumetricIndexTranslator"
					offset: [0, 0, 2]
					resolution: dst_resolution
				}]
				read_procs: [
					{
						"@type": "compare"
						"@mode": "partial"
						mode:    ">="
						value:   127
					},
					{
						"@type": "to_uint8"
						"@mode": "partial"
					},
				]
			}
		// }
	}
	dst: {
		"@type":     "build_cv_layer"
		path:        #ALIGNED_MISD_GARBAGE_PATH
		info_add_scales:     [dst_resolution]
		info_add_scales_mode: "replace"
		info_reference_path: #ALIGNED_MISD_Z1_PATH
		on_info_exists:      "overwrite"
		write_procs: [
			{
				"@type": "to_uint8"
				"@mode": "partial"
			},
			{
				"@type": "torch.mul"
				"@mode": "partial"
				other: 255
			}
		]
	}
}