import "math"
import "list"

#UNALIGNED_IMG_PATH: "gs://dacey-human-retina-001-montaging/montage_prod_384croprender_128cropenc_final/img_warped/final"
#UNALIGNED_DEFECT_PATH: "gs://dacey-human-retina-001-montaging/montage_prod_384croprender_128cropenc_final/img_warped/defect_mask"
// #UNALIGNED_ENC_PATH: "gs://dacey-human-retina-001-alignment-temp/unaligned/enc"
#RIGID_TRANSFORM_PATH: "gs://dacey-human-retina-001-alignment-temp/rigid_transforms_3x3_LMedS/absolute"
#RIGID_FIELD_PATH: "gs://dacey-human-retina-001-alignment-temp/rigid/field"
#RIGID_IMG_PATH: "gs://dacey-human-retina-001-alignment-temp/rigid/img"
#RIGID_DEFECT_PATH: "gs://dacey-human-retina-001-alignment-temp/rigid/defect_mask"
// #RIGID_ENC_PATH: "gs://dacey-human-retina-001-alignment-temp/rigid/enc"

#DST_INFO_CHUNK_SIZE: [2048, 2048, 1]
#MAX_TASK_SIZE: [8192, 8192, 1]

#REFERENCE_RES: [5, 5, 50]

#DATASET_BOUNDS: [
	[0 * #REFERENCE_RES[0], 65536 * #REFERENCE_RES[0]],
	[0 * #REFERENCE_RES[1], 65536 * #REFERENCE_RES[1]],
	[1 * #REFERENCE_RES[2], 3030 * #REFERENCE_RES[2]],
]

#ROI_BOUNDS: [
	[0 * #REFERENCE_RES[0], 65536 * #REFERENCE_RES[0]],
	[0 * #REFERENCE_RES[1], 65536 * #REFERENCE_RES[1]],
	[1 * #REFERENCE_RES[2], 3030 * #REFERENCE_RES[2]],
]


"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_cluster_region:  "us-east1"
worker_cluster_project: "dacey-human-retina-001"
worker_cluster_name:    "zutils"
worker_image:           "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:nico_py3.9_20240120"
worker_resources: {
	// memory: "10000Mi"
	"nvidia.com/gpu": "1"
}
worker_replicas: 100
local_test:      true
debug: true

target: {
	"@type": "mazepa.sequential_flow"
	stages: [
		// #GEN_AFFINE_TEMPLATE & {
		// 	_bounds: #ROI_BOUNDS
		// 	dst_resolution: [40, 40, 50]
		// 	dst: path: #RIGID_FIELD_PATH
		// },
		for z in [297,534,700,759,864,928,1159,1218,1800,1801,1829,1965,2020,2061,2310,2830,2879] {
			#WARP_IMG_TEMPLATE & {
				_bounds: [
					[0 * #REFERENCE_RES[0], 65536 * #REFERENCE_RES[0]],
					[0 * #REFERENCE_RES[1], 65536 * #REFERENCE_RES[1]],
					[z * #REFERENCE_RES[2], (z+1) * #REFERENCE_RES[2]],
				]
				dst_resolution: [320, 320, 50]
			}
		}
	]
},

#FIELD_INFO_OVERRIDE: {
	_dst_resolution: _
	type: "image"
	data_type: "float32",
	num_channels: 2,
	scales: [
		{
			let vx_res = _dst_resolution
			let ds_offset = [ for j in [0, 1, 2] {
				#DATASET_BOUNDS[j][0] / _dst_resolution[j]  // technically should be floor
			}]
			let ds_size = [ for j in [0, 1, 2] {
				math.Ceil((#DATASET_BOUNDS[j][1] - #DATASET_BOUNDS[j][0]) / _dst_resolution[j])
			}]

			chunk_sizes: [[ for j in [0, 1, 2] {list.Min([#DST_INFO_CHUNK_SIZE[j], ds_size[j]])}]]
			resolution:   vx_res
			encoding:     "zfpc"
			zfpc_correlated_dims: [true, true, false, false]
			zfpc_tolerance: 0.001953125
			key:          "\(vx_res[0])_\(vx_res[1])_\(vx_res[2])"
			voxel_offset: ds_offset
			size:         ds_size
		}
	],
	
}

#GEN_AFFINE_TEMPLATE: {
	_bounds: _
	let vx_res = dst_resolution
	let x_shape = math.Ceil(((_bounds[0][1] - _bounds[0][0]) / vx_res[0]))
	let y_shape = math.Ceil(((_bounds[1][1] - _bounds[1][0]) / vx_res[1]))
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type": "FieldFromTransform2D"
			shape: [x_shape, y_shape]
		}
		crop_pad: [0, 0, 0]
	}
	dst_resolution: _
	skip_intermediaries: true
	processing_chunk_sizes: [[x_shape, y_shape, 1]]
	processing_crop_pads:   [[0, 0, 0]]
	expand_bbox_resolution: true
	bbox: {
		"@type": "BBox3D.from_coords",
		start_coord: [_bounds[0][0], _bounds[1][0], _bounds[2][0]]
		end_coord: [_bounds[0][1], _bounds[1][1], _bounds[2][1]]
	}
	op_kwargs: {
		mat: {
			"@type": "build_cv_layer"
			path:    #RIGID_TRANSFORM_PATH
			index_procs: [{
				"@type": "VolumetricIndexOverrider"
				override_offset: [null, null, null]
				override_size: [2, 3, 1]
				override_resolution: [640, 640, 50]
			}]
		}
	}
	dst: {
		"@type":              "build_cv_layer"
		path:                 _
		info_field_overrides: #FIELD_INFO_OVERRIDE & {
			_dst_resolution: dst_resolution
		}
		on_info_exists:       "overwrite"
	}
}

#WARP_IMG_TEMPLATE: {
	_bounds: _ | *#ROI_BOUNDS
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((_bounds[0][1]-_bounds[0][0])/#DST_INFO_CHUNK_SIZE[0]/dst_resolution[0]) * #DST_INFO_CHUNK_SIZE[0]]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((_bounds[1][1]-_bounds[1][0])/#DST_INFO_CHUNK_SIZE[1]/dst_resolution[1]) * #DST_INFO_CHUNK_SIZE[1]]),
		1,
	]

	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    "mask"
		crop_pad: [256, 256, 0]
	}
	dst_resolution: _
	processing_chunk_sizes: [max_chunk_size]
	processing_crop_pads: [[0, 0, 0]]
	skip_intermediaries:    true
	expand_bbox_processing: true
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [_bounds[0][0], _bounds[1][0], _bounds[2][0]]
		end_coord: [_bounds[0][1], _bounds[1][1], _bounds[2][1]]
	}
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #UNALIGNED_DEFECT_PATH
		}
		field: {
			"@type": "build_cv_layer"
			path:    #RIGID_FIELD_PATH
			data_resolution: [40, 40, 50]
			interpolation_mode: "field"
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #RIGID_DEFECT_PATH
		info_reference_path: op_kwargs.src.path
		info_add_scales: [dst_resolution]
		info_add_scales_mode: "replace"
		info_chunk_size: [1024, 1024, 1]
		on_info_exists:       "overwrite"
		write_procs: [{
			"@type": "lambda"
			"lambda_str": "lambda x: x*255"
		}]
	}
}
