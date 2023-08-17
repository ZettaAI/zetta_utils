#SRC_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/rigid_to_elastic/v1/field"
#DST_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/rigid_to_elastic/v1/field_inv"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 3370]
	end_coord: [2048, 2048, 3390]
	resolution: [512, 512, 45]
}
#CROP_PAD: 512
#CHUNK:    1024 * 2

#FLOW: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type": "invert_field"
			"@mode": "partial"
		}
	}
	processing_chunk_sizes: [ [2048, 2048, 1]]
	processing_crop_pads: [[64, 64, 0]]

	dst_resolution: [256, 256, 45]
	bbox: #BBOX
	level_intermediaries_dirs: ["gs://tmp_2w/tmp_dirs"]
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: src.path
	}
}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x55"
worker_resources: {
	memory: "18560Mi"
}
worker_replicas:     100
batch_gap_sleep_sec: 0.05
local_test:          true
target:              #FLOW
