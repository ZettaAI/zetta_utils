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
	fn: {
		"@type": "apply_mask_fn"
		"@mode": "partial"
	}
	bbox: #BBOX
	src: {
		"@type":    "build_cv_layer"
		path:       _
		read_procs: _ | *[]
	}
	masks: [
		{
			"@type":    "build_cv_layer"
			path:       "gs://zetta_lee_fly_cns_001_alignment_temp/cns/rigid_x0/defect_mask"
			read_procs: _ | *[]
		},
	]
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: src.path
		info_chunk_size: [1024, 1024, 1]
		on_info_exists: "overwrite"
	}
}

"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_image:           "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x184"
worker_cluster_name:    "zutils-cns"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-lee-fly-vnc-001"
worker_resources: {
	memory:           "18560Mi"
	"nvidia.com/gpu": "1"
}
worker_replicas: 100
local_test:      false
target: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		for res in [32, 64] {
			#FLOW_TMPL & {
				src: path: "gs://zetta_lee_fly_cns_001_alignment_temp/cns/rigid_x0/encodings"
				dst: path: "gs://zetta_lee_fly_cns_001_alignment_temp/cns/rigid_x0/encodings_masked"
				dst_resolution: [res, res, 45]
				processing_chunk_sizes: [[6 * 1024, 6 * 1024, 1]]
			}
		},
		for res in [128, 256, 512, 1024] {
			#FLOW_TMPL & {
				src: path: "gs://zetta_lee_fly_cns_001_alignment_temp/cns/rigid_x0/encodings"
				dst: path: "gs://zetta_lee_fly_cns_001_alignment_temp/cns/rigid_x0/encodings_masked"
				dst_resolution: [res, res, 45]
				processing_chunk_sizes: [[1024, 1024, 1]]
			}
		},
		for res in [2048, 4096, 8192] {
			#FLOW_TMPL & {
				src: path: "gs://zetta_lee_fly_cns_001_alignment_temp/cns/rigid_x0/encodings"
				dst: path: "gs://zetta_lee_fly_cns_001_alignment_temp/cns/rigid_x0/encodings_masked"
				dst_resolution: [res, res, 45]
				processing_chunk_sizes: [[1024, 1024, 1]]
			}
		},

	]
}
