#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 3500]
	end_coord: [2048, 2048, 3510]
	resolution: [512, 512, 45]
}

#FLOW_TMPL: {
	"@type":                "build_subchunkable_apply_flow"
	expand_bbox:            true
	processing_chunk_sizes: _
	dst_resolution:         _
	op: {
		"@type":         "InterpolateOperation"
		mode:            _
		res_change_mult: [_, _, _] | *[2, 2, 1.0]
	}
	bbox: #BBOX
	src: {
		"@type":    "build_cv_layer"
		path:       _
		read_procs: _ | *[]
	}
	dst: src
}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x184"
worker_resources: {
	memory:           "18560Mi"
	"nvidia.com/gpu": "1"
}
worker_replicas:        40
worker_cluster_project: "zetta-lee-fly-vnc-001"
worker_cluster_region:  "us-east1"
worker_cluster_name:    "zutils-cns"
local_test:             false
target: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		{
			"@type": "mazepa.seq_flow"
			stages: [
				for res in [64, 128, 256, 512, 1024] {
					#FLOW_TMPL & {
						op: mode:  "img"
						src: path: "gs://zetta_lee_fly_cns_001_alignment_temp/cns/rigid_x0/raw_img"
						dst_resolution: [res, res, 45]
						processing_chunk_sizes: [[4096, 4096, 1]]
					}
				},
			]
		},
		{
			"@type": "mazepa.seq_flow"
			stages: [
				for res in [128, 256, 512, 1024, 2048, 4096, 8192] {
					#FLOW_TMPL & {
						op: mode: "mask"
						src: read_procs: [
							{"@type": "filter_cc", "@mode": "partial", mode: "keep_large", thr: 20},
						]
						src: path: "gs://zetta_lee_fly_cns_001_alignment_temp/cns/rigid_x0/defect_mask"
						dst_resolution: [res, res, 45]
						processing_chunk_sizes: [[4096, 4096, 1]]
					}
				},
			]
		},
		#FLOW_TMPL & {
			op: mode: "mask"
			op: res_change_mult: [0.5, 0.5, 1]
			src: path: "gs://zetta_lee_fly_cns_001_alignment_temp/cns/rigid_x0/defect_mask"
			dst_resolution: [32, 32, 45]
			processing_chunk_sizes: [[4096, 4096, 1]]
		},
	]
}
