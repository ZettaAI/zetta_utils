#SRC_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/fine_v4/M7_500xSM200_M6_500xSM200_M5_500xSM200_M4_250xSM200_M3_250xSM200_VV3_CT2.5_BS10/mip1/img/img_rendered"

#CHUNK_SIZE: [2048, 2048, 1]

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 2700]
	end_coord: [32768, 32768, 3200]
	resolution: [128, 128, 45]
}

"@type":         "mazepa.execute_on_gcp_with_sqs"
worker_image:    "us.gcr.io/zetta-research/zetta_utils:py3.9_torch_1.13.1_cu11.7_zu20230131_unet_pow"
worker_replicas: 50
worker_resources: {
	memory: "18560Mi"
}

target: {
	"@type": "build_interpolate_flow"
	bbox: #BBOX
	dst_resolution: [512, 512, 45]
	src_resolution: [256, 256, 45]
	chunk_size: #CHUNK_SIZE
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
	mode: "img"
}