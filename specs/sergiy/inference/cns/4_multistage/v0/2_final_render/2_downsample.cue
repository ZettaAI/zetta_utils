#FOLDER:   "gs://zetta_lee_fly_cns_001_alignment_temp/aced/med_x1/try_x0"
#SUFFIX:   "_x0"
#IMG_PATH: "\(#FOLDER)/img_aligned_final\(#SUFFIX)"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 2701]
	end_coord: [2048, 2048, 3155]
	resolution: [512, 512, 45]
}

#FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	//expand_bbox: true
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
	memory: "18560Mi"
	//"nvidia.com/gpu": "1"
}
worker_cluster_name:    "zutils-cns"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-lee-fly-vnc-001"
worker_replicas:        250
local_test:             false
target: {
	"@type": "mazepa.seq_flow"
	stages: [
		for res in [32, 64, 128, 256, 512, 1024] {
			#FLOW_TMPL & {
				src: path: #IMG_PATH
				processing_chunk_sizes: [[1024 * 4, 1024 * 4, 1]]

				op: mode: "img"
				dst_resolution: [res, res, 45]
			}
		},
	]
}
