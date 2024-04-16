import "math"

import "list"

#TGT_PATH:        "gs://tmp_2w/nico/cra8/enc/z0_masked_cns"
#SRC_PATH_PREFIX: "gs://tmp_2w/nico/cra8/enc_fwd_align_thick_render_thin/"
#DST_PATH_PREFIX: "gs://tmp_2w/nico/cra8/misd/thick_sm25/"

#DST_INFO_CHUNK_SIZE: [2048, 2048, 1] // Will automatically get truncated if dataset becomes too small
#TASK_SIZE: [2048, 2048, 1]

#DATASET_BOUNDS: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 0]
	// end_coord: [32768, 32768, 7010]
	end_coord: [12800, 12032, 6112]
	resolution: [32, 32, 42]
}

#ROI_BOUNDS: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 80]
	// end_coord: [32768, 32768, 3001]
	end_coord: [int, int, int] | *[12800, 12032, 90]
	resolution: [32, 32, 42]
}

#SCALES: [
	for i in list.Range(0, 2, 1) {
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

#CHANNEL_COUNT: 1
#MODELS: {
	"thr1.5": {
		"z1": {
			path: "gs://zetta-research-nico/training_artifacts/aced_misd_general/3.2.0_dsfactor2_thr1.5_lr0.0001_z1/epoch=44-step=11001-backup.ckpt.model.spec.json"
			//path: "gs://zetta-research-nico/training_artifacts/aced_misd_general/1.3.1_dsfactor2_thr1.5_lr0.0001_z1/epoch=77-step=19422-backup.ckpt.model.spec.json"
			// path: "gs://zetta-research-nico/training_artifacts/aced_misd_general/1.2.0_dsfactor2_thr1.5_lr0.0001_z1/last.ckpt.model.spec.json"
		}
		"z2": {
			path: "gs://zetta-research-nico/training_artifacts/aced_misd_general/1.2.3_dsfactor2_thr1.5_lr0.0001_z2_fp32/last.ckpt.model.spec.json"
		}
		"z1_2": {
			path: "gs://zetta-research-nico/training_artifacts/aced_misd_general/1.2.0_dsfactor2_thr1.5_lr0.0001_z1_2/last.ckpt.model.spec.json"
		}
	},
	"thr2.5": {
		"z1": {
			path: "gs://zetta-research-nico/training_artifacts/aced_misd_general/1.2.0_dsfactor2_thr2.5_lr0.0001_z1/last.ckpt.model.spec.json"
		}
		"z2": {
			path: "gs://zetta-research-nico/training_artifacts/aced_misd_general/3.2.0_dsfactor2_thr2.0_lr0.0001_z2/epoch=74-step=18730-backup.ckpt.model.spec.json"
			//path: "gs://zetta-research-nico/training_artifacts/aced_misd_general/1.3.3_dsfactor2_thr2.0_lr0.0001_z2_fp32/last.ckpt.model.spec.json"
			//path: "gs://zetta-research-nico/training_artifacts/aced_misd_general/1.2.0_dsfactor2_thr2.5_lr0.0001_z2/epoch=34-step=8652-backup.ckpt.model.spec.json"
		}
		"z1_2": {
			path: "gs://zetta-research-nico/training_artifacts/aced_misd_general/1.2.0_dsfactor2_thr2.5_lr0.0001_z1_2/last.ckpt.model.spec.json"
		}
	}
}

// #MODELS: {
// 	for thr in ["3.0"] {
// 		"thr\(thr)": {
// 			"z1": {
// 				path: "gs://zetta-research-nico/training_artifacts/aced_misd_general/1.1.0_dsfactor1_thr\(thr)_lr0.00002_z1/last.ckpt.model.spec.json"
// 			}
// 		}
// 	}
// }



#FLOW_TEMPLATE: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "VolumetricCallableOperation"
		fn:      {
			"@type":    "MisalignmentDetector"
			model_path: string
			apply_sigmoid: true
		}
		crop_pad: [16, 16, 0]
	}
	dst_resolution: [int, int, int]
	processing_chunk_sizes: [[int, int, int]]
	processing_crop_pads: [[0, 0, 0]]
	bbox: #ROI_BOUNDS
	skip_intermediaries: true
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    _
			index_procs?: _
		}
		tgt: {
			"@type": "build_cv_layer"
			path:    #TGT_PATH
		}
	}
	dst: {
		"@type":               "build_cv_layer"
		path:                  _
		info_field_overrides?: _
		on_info_exists:        "overwrite"
	}
}

"@type": "mazepa.execute"
target: {
	for i in list.Range(1, 2, 1) {
		let res_mult = [math.Pow(2, i), math.Pow(2, i), 1]
		let dst_res = [ for j in [0, 1, 2] {#DATASET_BOUNDS.resolution[j] * res_mult[j]}]
		let trunc_tasksize = [ for j in [0, 1, 2] {list.Min([#SCALES[i].size[j], #TASK_SIZE[j]])}]
		let roi_pad = [ for j in [0, 1, 2] {
			__mod((trunc_tasksize[j] - (__mod(#SCALES[i].size[j], trunc_tasksize[j]))), trunc_tasksize[j])
		}]
		"@type": "mazepa.concurrent_flow"
		stages: [
			for src_z in ["z2"] {
				"@type": "mazepa.concurrent_flow"
				stages: [
					for thr in ["thr2.5"] {
						"@type": "mazepa.concurrent_flow"
						stages: [
							#FLOW_TEMPLATE & {
								op: fn: {
									model_path:  #MODELS[thr][src_z].path
								}
								dst_resolution: dst_res
								processing_chunk_sizes: [trunc_tasksize]
								dst: info_field_overrides: {
									type:         "image"
									num_channels: 1
									data_type:    "uint8"
									scales:       #SCALES
								}
								bbox: #ROI_BOUNDS & {
									end_coord: [ for j in [0, 1, 2] {#ROI_BOUNDS.end_coord[j] + roi_pad[j]*math.Pow(2, i)}]
								}
								op_kwargs: src: path: "\(#SRC_PATH_PREFIX)\(src_z)"

								dst: path: "\(#DST_PATH_PREFIX)\(thr)/\(src_z)/model_\(src_z)"
							},
							// #FLOW_TEMPLATE & {
							// 	op: fn: {
							// 		model_path:  #MODELS[thr]["z1_2"].path
							// 	}
							// 	dst_resolution: dst_res
							// 	processing_chunk_sizes: [trunc_tasksize]
							// 	dst: info_field_overrides: {
							// 		type:         "image"
							// 		num_channels: 1
							// 		data_type:    "uint8"
							// 		scales:       #SCALES
							// 	}
							// 	bbox: #ROI_BOUNDS & {
							// 		end_coord: [ for j in [0, 1, 2] {#ROI_BOUNDS.end_coord[j] + roi_pad[j]*math.Pow(2, i)}]
							// 	}
							// 	op_kwargs: src: path: #TGT_PATH //"\(#SRC_PATH_PREFIX)\(src_z)"
							// 	dst: path: "\(#DST_PATH_PREFIX)\(thr)/\(src_z)/model_z1_2"
							// }
						]
					}
				]
			}
		]
	}
}
