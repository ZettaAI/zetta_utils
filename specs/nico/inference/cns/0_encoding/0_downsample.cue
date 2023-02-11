#SRC_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/elastic_low_res"

"@type":         "mazepa.execute_on_gcp_with_sqs"
worker_image:    "us.gcr.io/zetta-research/zetta_utils:py3.9_torch_1.13.1_cu11.7_zu20230131_unet_pow"
worker_replicas: 10
worker_resources: {
	memory: "18560Mi"
}
batch_gap_sleep_sec: 0.1

local_test: false

target: {
	"@type": "build_interpolate_flow"
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [0, 0, 0]
		end_coord: [4096, 4096, 7010]
		resolution: [512, 512, 45]
	}
	dst_resolution: [512, 512, 45]
	src_resolution: [256, 256, 45]
	chunk_size: [2 * 1024, 2 * 1024, 1]
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
	mode: "img"
}