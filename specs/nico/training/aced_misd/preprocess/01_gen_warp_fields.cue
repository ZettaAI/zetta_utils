import "math"
import "list"

#BASE_PATH: "gs://zetta-research-nico/encoder/"
#TGT_IMG_PATH: #BASE_PATH + "datasets/"
#WARPED_SRC_IMG_PATH: #BASE_PATH + "pairwise_aligned/" // + k + "warped_enc/"
#PERLIN_FIELD_PATH: #BASE_PATH + "misd/misalignment_fields/"

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
	"microns_minnie": {
	    "contiguous": false
	    "bounds": [[0, 1703936], [0, 1441792], [0, 320]]
	    "resolution": [32, 32, 40]
	},
	"microns_interneuron": {
	    "contiguous": false
	    "bounds": [[0, 720896], [0, 720896], [0, 1280]]
	    "resolution": [32, 32, 40]
	},
	"aibs_v1dd": {
	    "contiguous": false
	    "bounds": [[0.0, 1231667.2], [0.0, 834355.2], [0.0, 1080.0]]
	    "resolution": [38.8, 38.8, 45.0]
	},
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
	"lee_fanc": {
	    "contiguous": false
	    "bounds": [[0.0, 352256.0], [0.0, 951091.2], [0.0, 2700.0]]
	    "resolution": [34.4, 34.4, 45.0]
	},
	"lee_banc": {
	    "contiguous": false
	    "bounds": [[0, 819200], [0, 1015808], [0, 900]]
	    "resolution": [32, 32, 45]
	},
	"lee_ppc": {
	    "contiguous": true
	    "bounds": [[0, 98304], [0, 98304], [0, 36400]]
	    "resolution": [32, 32, 40]
	},
	"lee_mosquito": {
	    "contiguous": false
	    "bounds": [[0, 704512], [0, 450560], [0, 2240]]
	    "resolution": [32, 32, 40]
	},
	"lichtman_zebrafish": {
	    "contiguous": false
	    "bounds": [[0, 294912], [0, 393216], [0, 4560]]
	    "resolution": [32, 32, 30]
	},
	"prieto_godino_larva": {
	    "contiguous": true
	    "bounds": [[0, 134976], [0, 144992], [0, 14400]]
	    "resolution": [32, 32, 32]
	},
	"fafb_v15": {
	    "contiguous": false
	    "bounds": [[0, 884736], [0, 393216], [0, 2000]]
	    "resolution": [32, 32, 40]
	},
	"lichtman_h01": {
	    "contiguous": false
	    "bounds": [[0, 3440640], [0, 1933312], [0, 198]]
	    "resolution": [32, 32, 33]
	},
	"janelia_hemibrain": {
	    "contiguous": true
	    "bounds": [[0, 317824], [0, 331168], [0, 3296]]
	    "resolution": [32, 32, 32]
	},
	"janelia_manc": {
	    "contiguous": false
	    "bounds": [[0, 262144], [0, 360448], [0, 5952]]
	    "resolution": [32, 32, 32]
	},
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
#PERLIN_FIELD_DS_FACTOR: math.Pow(2, 3)
#FIELD_INFO_OVERRIDE: {
	_dataset_bounds: _
	_dst_resolution: _
	type: "image"
	data_type: "float32",
	num_channels: 2,
	scales: [
		{
			let vx_res = _dst_resolution
			let ds_offset = [ for j in [0, 1, 2] {
				_dataset_bounds[j][0] / _dst_resolution[j]  // technically should be floor
			}]
			let ds_size = [ for j in [0, 1, 2] {
				math.Ceil((_dataset_bounds[j][1] - _dataset_bounds[j][0]) / _dst_resolution[j])
			}]

			chunk_sizes: [[ for j in [0, 1, 2] {list.Min([#DST_INFO_CHUNK_SIZE[j], ds_size[j]])}]]
			resolution:   vx_res
			encoding:     "zfpc"
			zfpc_correlated_dims: [true, true, false, false]
			zfpc_tolerance: 0.001953125
			key:          "\(vx_res[0])_\(vx_res[1])_\(vx_res[2])"
			voxel_offset: ds_offset
			size:         ds_size
		}
	],
	
}



#MAX_DISP: 20
#MEDIAN_DISP: 7.5
#PERLIN_NOISE_TEMPLATE: {
	_bounds: _
	let vx_res = dst_resolution
	let x_mult = math.Ceil(((_bounds[0][1] - _bounds[0][0]) / vx_res[0]) / 2048)
	let y_mult = math.Ceil(((_bounds[1][1] - _bounds[1][0]) / vx_res[1]) / 2048)
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type": "gen_biased_perlin_noise_field"
			"@mode": "partial"
			shape: [2, x_mult * 2048, y_mult * 2048, 1]
			res:   [   x_mult * 2,    y_mult * 2      ]
			max_displacement_px: #MAX_DISP / #PERLIN_FIELD_DS_FACTOR
			field_magn_thr_px: #MEDIAN_DISP / #PERLIN_FIELD_DS_FACTOR
			octaves: 8
			device: "cpu"
		}
		crop_pad: [0, 0, 0]
	}
	dst_resolution: _
	skip_intermediaries: true
	processing_chunk_sizes: [[x_mult * 2048, y_mult * 2048, 1]]
	processing_crop_pads:   [[0, 0, 0]]
	expand_bbox_resolution: true
	bbox: {
		"@type": "BBox3D.from_coords",
		start_coord: [_bounds[0][0], _bounds[1][0], _bounds[2][0]]
		end_coord: [_bounds[0][1], _bounds[1][1], _bounds[2][1]]
	}
	dst: {
		"@type":              "build_cv_layer"
		path:                 _
		info_field_overrides: #FIELD_INFO_OVERRIDE & {
			_dataset_bounds: _bounds
			_dst_resolution: dst_resolution
		}
	}
}


"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:nico_py3.9_20231118"
worker_resources: {
	memory:           "10560Mi"
}
worker_replicas:      100
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
			#PERLIN_NOISE_TEMPLATE & {
				_bounds: dataset.bounds,
				dst: path: #PERLIN_FIELD_PATH + key + "/raw_perlin"

				let ds_factor = [#PERLIN_FIELD_DS_FACTOR, #PERLIN_FIELD_DS_FACTOR, 1]
				let res = [ for j in [0, 1, 2] {dataset.resolution[j] * ds_factor[j]} ]
				dst_resolution: res
			}
		}
	]
}