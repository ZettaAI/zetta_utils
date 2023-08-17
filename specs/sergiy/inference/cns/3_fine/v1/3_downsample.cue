#BASE_FOLDER: "gs://zetta_lee_fly_cns_001_alignment_temp/aced/fine_x0/from_med_x0/v1"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 429]
	end_coord: [2048, 2048, 600]
	resolution: [512, 512, 45]
}

#FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	//expand_bbox_processing: true
	shrink_processing_chunk: true
	processing_chunk_sizes:  _
	dst_resolution:          _
	op: {
		"@type":         "InterpolateOperation"
		mode:            _
		res_change_mult: _ | *[2, 2, 1]
	}
	bbox: #BBOX
	src: {
		"@type":    "build_ts_layer"
		path:       _
		read_procs: _ | *[]
	}
	dst: {
		"@type": "build_cv_layer"
		path:    src.path
	}

}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x134"
worker_resources: {
	memory:           "18560Mi"
	"nvidia.com/gpu": "1"
}
worker_replicas:     250
batch_gap_sleep_sec: 0.05
local_test:          false
target: {
	"@type": "mazepa.sequential_flow"
	stages: [
		for res in [32, 64, 128, 256, 512, 1024] {
			#FLOW_TMPL & {
				processing_chunk_sizes: [[1024 * 4, 1024 * 4, 1]]

				op: mode:  "img"
				src: path: "\(#BASE_FOLDER)/raw_img" //"gs://zetta_lee_fly_cns_001_alignment_temp/aced/fine_x0/from_med_x0/img_aligned_try_x1_iter400_rig2000"
				dst_resolution: [res, res, 45]
			}
		},
	]
}
