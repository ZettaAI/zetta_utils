import "math"
import "list"

#BASE_PATH: "gs://zetta-research-nico/encoder/"
// #TGT_IMG_PATH: #BASE_PATH + "datasets/" // + k
#ORIGINAL_WARPED_SRC_IMG_PATH: #BASE_PATH + "pairwise_aligned/" // + k + "/warped_img"
// #TGT_ENC_PATH: #BASE_PATH + "pairwise_aligned/" // + k + "/tgt_enc_2023"
// #WARPED_SRC_ENC_PATH: #BASE_PATH + "pairwise_aligned/" // + k + "/warped_enc_2023"
#PERLIN_FIELD_PATH: #BASE_PATH + "misd/misalignment_fields/" // + k + "/raw_perlin"
// #DST_FIELD_PATH: #BASE_PATH + "misd/misalignment_fields/" // + k + "/optimized_perlin" | "/no_perlin" + "/z\(_z_offset)"

#DST_WARPED_SRC_IMG_PATH: #BASE_PATH + "misd/img/" // + k + "/good_alignment" | "/bad_alignment" + "/z\(_z_offset)"
// #DST_WARPED_SRC_ENC_PATH: #BASE_PATH + "misd/enc/" // + k + "/good_alignment" | "/bad_alignment" + "/z\(_z_offset)"


#DATASETS: {
	"microns_pinky": {
		"contiguous": true
		"bounds": [[0, 262144], [0, 131072], [0, 10240]]
		"resolution": [32, 32, 40]
	}
	"microns_basil": {
		"contiguous": true
		"bounds": [[0, 819200], [0, 983040], [0, 400]]
		"resolution": [32, 32, 40]
	},
	// // "microns_minnie": {
	// // 	"contiguous": false
	// // 	"bounds": [[0, 1703936], [0, 1441792], [0, 320]]
	// // 	"resolution": [32, 32, 40]
	// // },
	// // "microns_interneuron": {
	// // 	"contiguous": false
	// // 	"bounds": [[0, 720896], [0, 720896], [0, 1280]]
	// // 	"resolution": [32, 32, 40]
	// // },
	// // "aibs_v1dd": {
	// // 	"contiguous": false
	// // 	"bounds": [[0.0, 1231667.2], [0.0, 834355.2], [0.0, 1080.0]]
	// // 	"resolution": [38.8, 38.8, 45.0]
	// // },
	"kim_n2da": {
		"contiguous": true
		"bounds": [[0, 32768], [0, 32768], [0, 31050]]
		"resolution": [32, 32, 50]
	},
	"kim_pfc2022": {
		"contiguous": true
		"bounds": [[0, 229376], [0, 196608], [0, 7320]]
		"resolution": [32, 32, 40]
	},
	"kronauer_cra9": {
		"contiguous": true
		"bounds": [[0, 393216], [0, 327680], [0, 588]]
		"resolution": [32, 32, 42]
	},
	"kubota_001": {
		"contiguous": true
		"bounds": [[0, 204800], [0, 204800], [0, 12000]]
		"resolution": [40, 40, 40]
	},
	// // "lee_fanc": {
	// // 	"contiguous": false
	// // 	"bounds": [[0.0, 352256.0], [0.0, 951091.2], [0.0, 2700.0]]
	// // 	"resolution": [34.4, 34.4, 45.0]
	// // },
	// // "lee_banc": {
	// // 	"contiguous": false
	// // 	"bounds": [[0, 819200], [0, 1015808], [0, 900]]
	// // 	"resolution": [32, 32, 45]
	// // },
	"lee_ppc": {
		"contiguous": true
		"bounds": [[0, 98304], [0, 98304], [0, 36400]]
		"resolution": [32, 32, 40]
	},
	// // "lee_mosquito": {
	// // 	"contiguous": false
	// // 	"bounds": [[0, 704512], [0, 450560], [0, 2240]]
	// // 	"resolution": [32, 32, 40]
	// // },
	// // "lichtman_zebrafish": {
	// // 	"contiguous": false
	// // 	"bounds": [[0, 294912], [0, 393216], [0, 4560]]
	// // 	"resolution": [32, 32, 30]
	// // },
	"prieto_godino_larva": {
		"contiguous": true
		"bounds": [[0, 134976], [0, 144992], [0, 14400]]
		"resolution": [32, 32, 32]
	},
	// // "fafb_v15": {
	// // 	"contiguous": false
	// // 	"bounds": [[0, 884736], [0, 393216], [0, 2000]]
	// // 	"resolution": [32, 32, 40]
	// // },
	// // "lichtman_h01": {
	// // 	"contiguous": false
	// // 	"bounds": [[0, 3440640], [0, 1933312], [0, 198]]
	// // 	"resolution": [32, 32, 33]
	// // },
	"janelia_hemibrain": {
		"contiguous": true
		"bounds": [[0, 317824], [0, 331168], [0, 3296]]
		"resolution": [32, 32, 32]
	},
	// // "janelia_manc": {
	// // 	"contiguous": false
	// // 	"bounds": [[0, 262144], [0, 360448], [0, 5952]]
	// // 	"resolution": [32, 32, 32]
	// // },
	"nguyen_thomas_2022": {
		"contiguous": true
		"bounds": [[0, 998400], [0, 921600], [0, 400]]
		"resolution": [32, 32, 40]
	},
	"mulcahy_2022_16h": {
		"contiguous": true
		"bounds": [[0, 243712], [0, 73728], [0, 14700]]
		"resolution": [32, 32, 30]
	},
	"wildenberg_2021_vta_dat12a": {
		"contiguous": true
		"bounds": [[0, 82080], [0, 85184], [0, 7640]]
		"resolution": [32, 32, 40]
	},
	"bumbarber_2013": {
		"contiguous": true
		"bounds": [[0.0, 63897.6], [0.0, 63897.6], [0.0, 102400.0]]
		"resolution": [31.2, 31.2, 50.0]
	},
	"wilson_2019_p3": {
		"contiguous": true
		"bounds": [[0, 163840], [0, 229376], [0, 7020]]
		"resolution": [32, 32, 30]
	},
	"ishibashi_2021_em1": {
		"contiguous": true
		"bounds": [[0, 24576], [0, 16384], [0, 4544]]
		"resolution": [32, 32, 32]
	},
	"ishibashi_2021_em2": {
		"contiguous": true
		"bounds": [[0, 26624], [0, 18432], [0, 5376]]
		"resolution": [32, 32, 32]
	},
	"templier_2019_wafer1": {
		"contiguous": true
		"bounds": [[0, 294912], [0, 229376], [0, 6500]]
		"resolution": [32, 32, 50]
	},
	"templier_2019_wafer3": {
		"contiguous": true
		"bounds": [[0, 229376], [0, 196608], [0, 9750]]
		"resolution": [32, 32, 50]
	},
	"lichtman_octopus2022": {
		"contiguous": true
		"bounds": [[0, 229376], [0, 360448], [0, 3180]]
		"resolution": [32, 32, 30]
	}
}

#DST_INFO_CHUNK_SIZE: [2048, 2048, 1]
#MAX_TASK_SIZE: [8192, 8192, 1]

#WARP_IMG_TEMPLATE: {
	_bounds: _
	_high_resolution: [number, number, number]
	_z_offset: _
	_layer_name: _
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((_bounds[0][1] - _bounds[0][0]) / #DST_INFO_CHUNK_SIZE[0] / dst_resolution[0]) * #DST_INFO_CHUNK_SIZE[0]]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((_bounds[1][1] - _bounds[1][0]) / #DST_INFO_CHUNK_SIZE[1] / dst_resolution[1]) * #DST_INFO_CHUNK_SIZE[1]]),
		1
	]

	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode: "img"
	}
	dst_resolution: [number, number, number]
	processing_chunk_sizes: [max_chunk_size, [1024, 1024, 1]]
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	expand_bbox_resolution: true
	skip_intermediaries: true
	bbox: {
		"@type": "BBox3D.from_coords",
		start_coord: [_bounds[0][0], _bounds[1][0], _bounds[2][0]]
		end_coord: [_bounds[0][1], _bounds[1][1], _bounds[2][1]]
	}
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #ORIGINAL_WARPED_SRC_IMG_PATH + _layer_name + "/warped_img"
			index_procs: [{
				"@type": "VolumetricIndexTranslator"
				offset: [0, 0, _z_offset - 1]  // src is already offset by 1
				resolution: dst_resolution
			}]
		}
		field: {
			"@type":            "build_cv_layer"
			path:               #PERLIN_FIELD_PATH  + _layer_name + "/raw_perlin_32nm"
			// data_resolution:    [dst_resolution[0] * 8, dst_resolution[1] * 8, dst_resolution[2]]
			// interpolation_mode: "field"
		}
	}
	dst: {
		"@type": "build_cv_layer"
		path:    #DST_WARPED_SRC_IMG_PATH + _layer_name + "/perlin32_only_img/z\(_z_offset)"
		info_reference_path: op_kwargs.src.path
		on_info_exists: "overwrite"
	}
}



"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:nico_py3.9_20231213"
worker_resources: {
	"nvidia.com/gpu":     1
}
worker_replicas:      200
batch_gap_sleep_sec:  0.1
do_dryrun_estimation: true
local_test:           false
worker_cluster_project: "zetta-research"
worker_cluster_region: "us-east1"
worker_cluster_name: "zutils-x3"
target: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		for key, dataset in #DATASETS {
			"@type": "mazepa.concurrent_flow"
			stages: [
				for z_offset in list.Range(1, 3, 1) {
					"@type": "mazepa.concurrent_flow"
					stages: [
						#WARP_IMG_TEMPLATE & {
							_bounds: dataset.bounds
							_z_offset: z_offset
							_layer_name: key
							dst_resolution: dataset.resolution
						}
					]
				}
			]
		}
	]
}