#FIELD_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/rigid_to_elastic/v1/field"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 2000]
	end_coord: [2048, 2048, 2500]
	resolution: [512, 512, 45]
}
#BIG_BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 2000]
	end_coord: [1024 * 8, 1024 * 8, 2500]
	resolution: [512, 512, 45]
}

#FLOW_TMPL: {
	"@type":        "build_interpolate_flow"
	chunk_size:     _
	src_resolution: _
	dst_resolution: _
	mode:           _
	bbox:           _ | *#BBOX
	src: {
		"@type":    "build_cv_layer"
		path:       _
		read_procs: _ | *[]
	}

}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x53"
worker_resources: {
	memory: "18560Mi"
}
worker_replicas:     100
batch_gap_sleep_sec: 0.05
local_test:          false
target: {
	"@type": "mazepa.concurrent_flow"
	stages: [

		{
			"@type": "mazepa.seq_flow"
			stages: [
				#FLOW_TMPL & {
					mode: "mask"
					src: path: "gs://sergiy_exp/aced/demo_x0/rigid_to_elastic/defect_mask"
					src_resolution: [64, 64, 45]
					dst_resolution: [32, 32, 45]
					chunk_size: [1024 * 4, 1024 * 4, 1]
				},
				for res in [128, 256, 512] {
					#FLOW_TMPL & {
						mode: "mask"
						src: path: "gs://sergiy_exp/aced/demo_x0/rigid_to_elastic/defect_mask"
						src: read_procs: [
							{"@type": "filter_cc", "@mode": "partial", mode: "keep_large", thr: 20},
						]
						src_resolution: [res / 2, res / 2, 45]
						dst_resolution: [res, res, 45]
					}
					chunk_size: [1024 * 2, 1024 * 2, 1]
				},
				for res in [1024, 2048, 4096] {
					#FLOW_TMPL & {
						mode: "mask"
						bbox: #BIG_BBOX
						src: path: "gs://sergiy_exp/aced/demo_x0/rigid_to_elastic/defect_mask"
						src: read_procs: [
							{"@type": "filter_cc", "@mode": "partial", mode: "keep_large", thr: 5},
						]
						src_resolution: [res / 2, res / 2, 45]
						dst_resolution: [res, res, 45]
						chunk_size: [1024, 1024, 1]
					}
				},
			]
		},
		{
			"@type": "mazepa.seq_flow"
			stages: [
				for res in [64, 128] {
					#FLOW_TMPL & {
						chunk_size: [1024 * 4, 1024 * 4, 1]
						mode: "img"
						src: path: "gs://sergiy_exp/coarse_x0/raw_img"
						src_resolution: [res / 2, res / 2, 45]
						dst_resolution: [res, res, 45]
					}
				},
				for res in [256, 512] {
					#FLOW_TMPL & {
						chunk_size: [1024 * 2, 1024 * 2, 1]
						mode: "img"
						src: path: "gs://sergiy_exp/coarse_x0/raw_img"
						src_resolution: [res / 2, res / 2, 45]
						dst_resolution: [res, res, 45]
					}
				},
				for res in [1024, 2048, 4096] {
					#FLOW_TMPL & {
						mode: "img"
						bbox: #BIG_BBOX
						chunk_size: [1024, 1024, 1]
						src: path: "gs://sergiy_exp/coarse_x0/raw_img"
						src_resolution: [res / 2, res / 2, 45]
						dst_resolution: [res, res, 45]
					}
				},
			]
		},
	]
}
