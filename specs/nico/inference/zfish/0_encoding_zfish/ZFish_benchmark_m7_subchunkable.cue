import "math"

import "list"

#SRC_PATH: "gs://zetta_jlichtman_zebrafish_001_alignment/fine_full_v1/img"
#DST_PATH: "gs://tmp_2w/nico/zfish_enc_bench_subchunk_crop"

#DST_INFO_CHUNK_SIZE: [2048, 2048, 1] // Will automatically get truncated if dataset becomes too small
#TASK_SIZE: [2048, 2048, 1]

#DATASET_BOUNDS: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 0]
	end_coord: [12288, 16384, 4014]
	resolution: [32, 32, 30]
}

#ROI_BOUNDS: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 1000]
	end_coord: [int, int, int] | *[12288, 16384, 1005]
	resolution: [32, 32, 30]
}

#SCALES: [
	for i in list.Range(0, 6, 1) {
		let ds_factor = [math.Pow(2, i), math.Pow(2, i), 1]
		let vx_res = [ for j in [0, 1, 2] {#DATASET_BOUNDS.resolution[j] * ds_factor[j]}]
		let ds_offset = [ for j in [0, 1, 2] {
			__div(#DATASET_BOUNDS.start_coord[j], ds_factor[j])
		}]
		let ds_size = [ for j in [0, 1, 2] {
			__div((#DATASET_BOUNDS.end_coord[j] - ds_offset[j]), ds_factor[j])
		}]

		chunk_sizes: [[ for j in [0, 1, 2] {list.Min([#DST_INFO_CHUNK_SIZE[j], ds_size[j]])}]]
		resolution:   vx_res
		encoding:     "raw"
		key:          "\(vx_res[0])_\(vx_res[1])_\(vx_res[2])"
		voxel_offset: ds_offset
		size:         ds_size
	},
]

"@type":              "mazepa.execute"
do_dryrun_estimation: true
target: {
	let trunc_tasksize = [ for j in [0, 1, 2] {list.Min([#SCALES[5].size[j], #TASK_SIZE[j]])}]
	let roi_pad = [ for j in [0, 1, 2] {
		__mod((trunc_tasksize[j] - (__mod(#SCALES[5].size[j], trunc_tasksize[j]))), trunc_tasksize[j])
	}]

	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type":    "BaseCoarsenerSubchunkable"
			model_path: "gs://zetta-research-nico/training_artifacts/base_enc_zfish/2.6.0_M3_M8_conv6_lr0.00005_equi0.5_post1.2-1.2_fmt0.8_zfish/last.ckpt.model.spec.json"
			ds_factor:  32
		}
		res_change_mult: [32, 32, 1]
	}
	dst_resolution: [1024, 1024, 30]
	processing_chunk_sizes: [trunc_tasksize, [32, 32, 1]]
	processing_crop_pads: [[0, 0, 0], [16, 16, 0]]
	max_reduction_chunk_sizes: trunc_tasksize
	level_intermediaries_dirs: ["file://~/.zutil/tmp", "file://~/.zutil/tmp"]
	bbox: #ROI_BOUNDS & {
		end_coord: [ for j in [0, 1, 2] {#ROI_BOUNDS.end_coord[j] + roi_pad[j]*32}]
	}
	op_kwargs: {
		src: {
			"@type": "build_ts_layer"
			path:    #SRC_PATH
		}
	}
	dst: {
		"@type": "build_cv_layer"
		path:    #DST_PATH
		info_field_overrides: {
			type:         "image"
			num_channels: 1
			data_type:    "int8"
			scales:       #SCALES
		}
	}
}
