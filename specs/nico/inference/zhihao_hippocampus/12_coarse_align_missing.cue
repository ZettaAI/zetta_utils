import "math"

import "list"

import "strconv"

#Z_MAP: {
    "1417": 1416,
    "1418": 1419,

    "1420": 1419,
    "1421": 1419,
    "1422": 1423,

    "1431": 1430,

    "1433": 1432,

    "1440": 1439,
    "1441": 1439,
    "1442": 1439,
    "1443": 1446,
    "1444": 1446,
    "1445": 1446,

    "1447": 1446,

    "1449": 1448,
    "1450": 1451,

    // "1452": 1451,
    "1453": 1451,
    "1454": 1451,
    "1455": 1451,
    "1456": 1451,
    "1457": 1451,
    "1458": 1464,
    "1459": 1464,
    "1460": 1464,
    "1461": 1464,
    "1462": 1464,
    "1463": 1464,

    "1465": 1464,
    "1466": 1464,
    "1467": 1464,
    "1468": 1464,
    "1469": 1464,
    "1470": 1464,
    "1471": 1464,
    "1472": 1479,
    "1473": 1479,
    "1474": 1479,
    "1475": 1479,
    "1476": 1479,
    "1477": 1479,
    "1478": 1479,

    "1480": 1479,
    "1481": 1479,
    "1482": 1479,
    "1483": 1479,
    "1484": 1479,
    "1485": 1479,
    "1486": 1479,
    "1487": 1479,
    "1488": 1479,
    "1489": 1479,
    "1490": 1479,
    "1491": 1501,
    "1492": 1501,
    "1493": 1501,
    "1494": 1501,
    "1495": 1501,
    "1496": 1501,
    "1497": 1501,
    "1498": 1501,
    "1499": 1501,
    // "1500": 1501,

    "1551": 1557,
    "1552": 1557,
    "1553": 1557,
    "1554": 1557,
    "1555": 1557,
    "1556": 1557,

    "1559": 1558,

    // "1561": 1560,
    "1562": 1564,
    "1563": 1564,

    "1565": 1564,
    "1566": 1564,
    "1567": 1564,
    "1568": 1564,
    "1569": 1564,
    "1570": 1575,
    "1571": 1575,
    "1572": 1575,
    "1573": 1575,
    "1574": 1575,

    "1576": 1575,
    "1577": 1578,

    "1579": 1578,
    "1580": 1578,
    // "1581": 1578,
    "1582": 1585,
    "1583": 1585,
    "1584": 1585,

    "1586": 1585,
    "1587": 1585,
    "1588": 1585,
    "1589": 1585,
    "1590": 1594,
    "1591": 1594,
    "1592": 1594,
    // "1593": 1594,

    "1595": 1594,
    "1596": 1594,
    "1597": 1599,

    "1721": 1720,
    "1722": 1720,
    "1723": 1720,
    "1724": 1720,
    "1725": 1720,
    "1726": 1730,
    "1727": 1730,
    "1728": 1730,
    "1729": 1730,

    "1731": 1730,
    "1732": 1730,
    "1733": 1730,
    "1734": 1730,
    "1735": 1739,
    "1736": 1739,
    "1737": 1739,
    "1738": 1739,

    "1746": 1745,
    "1747": 1745,
    "1748": 1750,
    "1749": 1750,

    "1752": 1751,
    "1753": 1751,
    "1754": 1751,
    "1755": 1751,
    "1756": 1751,
    "1757": 1751,
    "1758": 1751,
    "1759": 1751,
    "1760": 1751,
    "1761": 1751,
    "1762": 1751,
    "1763": 1751,
    "1764": 1751,
    "1765": 1751,
    "1766": 1751,
    "1767": 1781,
    "1768": 1781,
    "1769": 1781,
    "1770": 1781,
    "1771": 1781,
    "1772": 1781,
    "1773": 1781,
    "1774": 1781,
    "1775": 1781,
    "1776": 1781,
    "1777": 1781,
    "1778": 1781,
    "1779": 1781,
    "1780": 1781,

    "1786": 1785,

    "1788": 1787,

    "1790": 1789,

    "1793": 1792,
    "1794": 1792,
    "1795": 1792,
    "1796": 1792,
    "1797": 1800,
    "1798": 1800,
    "1799": 1800,
}

_ALIGN_PARAMS: {
	"24576": {channel: -1, sm: 100, num_iter: 700, lr: 0.001, src_zeros_sm_mult: 1.0, tgt_zeros_sm_mult: 1.0},
	"12288": {channel: 3, sm: 300, num_iter: 700, lr: 0.001, src_zeros_sm_mult: 0.1, tgt_zeros_sm_mult: 0.1},
	"6144": {channel: 3, sm: 300, num_iter: 700, lr: 0.0025, src_zeros_sm_mult: 0.1, tgt_zeros_sm_mult: 0.1},
}


// INPUT
// #UNALIGNED_IMG_PATH:    "https://td.princeton.edu/sseung-test1/ca3-alignment-temp/full_section_imap4"
#UNALIGNED_ENC_PATH:    "gs://zetta-research-nico/hippocampus/low_res_enc_c4"

#TARGET_PATH:           "gs://zetta-research-nico/hippocampus/coarse_final/low_res_enc_c4"
#AFFINE_ENC_PATH:       "gs://zetta-research-nico/hippocampus/redo_missing/affine_inv/low_res_enc_c4"
#AFFINE_FIELD_PATH:     "gs://zetta-research-nico/hippocampus/redo_missing/affine/field"
#AFFINE_INV_FIELD_PATH: "gs://zetta-research-nico/hippocampus/redo_missing/affine_inv/field"

// OUTPUT
#COARSE_WO_AFFINE_FIELD_PATH: "gs://zetta-research-nico/hippocampus/redo_missing/coarse_wo_affine/field"
#COARSE_FIELD_PATH:     "gs://zetta-research-nico/hippocampus/redo_missing/coarse/field"
#COARSE_INV_FIELD_PATH: "gs://zetta-research-nico/hippocampus/redo_missing/coarse_inv/field"
#COARSE_ENC_PATH:       "gs://zetta-research-nico/hippocampus/redo_missing/coarse_inv/low_res_enc_c4"

#DST_INFO_CHUNK_SIZE: [2048, 2048, 1] // Will automatically get truncated if dataset becomes too small
#MAX_TASK_SIZE: [8192, 8192, 1]
#REFERENCE_RES: [3, 3, 45]

#DATASET_BOUNDS: [
    [0 * #REFERENCE_RES[0], 524288 * #REFERENCE_RES[0]],  // [0 * #REFERENCE_RES[0], 1474560 * #REFERENCE_RES[0]],
	[0 * #REFERENCE_RES[1], 524288 * #REFERENCE_RES[1]],  // [0 * #REFERENCE_RES[1], 1474560 * #REFERENCE_RES[1]],
	[24 * #REFERENCE_RES[2], 3994 * #REFERENCE_RES[2]],
]

#ROI_BOUNDS: [
	[0 * #REFERENCE_RES[0], 524288 * #REFERENCE_RES[0]],
	[0 * #REFERENCE_RES[1], 524288 * #REFERENCE_RES[1]],
	[24 * #REFERENCE_RES[2], 3994 * #REFERENCE_RES[2]],
]

"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-research"
worker_cluster_name:    "zutils-x3"
worker_image:           "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:nico_py3.9_20240306"
worker_resources: {
	"nvidia.com/gpu": "1"
}
worker_replicas: 20
local_test:      false
debug: false

target: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		// {
		// 	"@type": "mazepa.concurrent_flow"
		// 	stages: [
		// 		for key, tgt_z in #Z_MAP {
		// 			let z = strconv.Atoi(key)
		// 			let bounds = [
		// 				[#ROI_BOUNDS[0][0], #ROI_BOUNDS[0][1]],
		// 				[#ROI_BOUNDS[1][0], #ROI_BOUNDS[1][1]],
		// 				[z * #REFERENCE_RES[2], (z+1) * #REFERENCE_RES[2]],
		// 			]
		// 			#COMPUTE_FIELD_TEMPLATE & {
		// 				_z_offset: tgt_z - z
		// 				_bounds: bounds
		// 			}
		// 		},
		// 	]
		// },
		for key, tgt_z in #Z_MAP {
			let z = strconv.Atoi(key)
			let bounds = [
				[#ROI_BOUNDS[0][0], #ROI_BOUNDS[0][1]],
				[#ROI_BOUNDS[1][0], #ROI_BOUNDS[1][1]],
				[z * #REFERENCE_RES[2], (z+1) * #REFERENCE_RES[2]],
			]
			"@type": "mazepa.sequential_flow"
			stages: [
				#COMPOSE_FIELD_TEMPLATE & {
					_bounds: bounds
					dst_resolution: [6144, 6144, 45]
				},
				#INVERT_FIELD_TEMPLATE & {
					_bounds: bounds
					dst_resolution: [6144, 6144, 45]
				},
				#WARP_ENC_TEMPLATE & {
					_bounds: bounds
					dst_resolution: [6144, 6144, 45]
				},
				#WARP_ENC_TEMPLATE & {
					_bounds: bounds
					dst_resolution: [12288, 12288, 45]
				}
			]
		},
	]
}




#STAGE_TMPL: {
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1]-#ROI_BOUNDS[0][0])/dst_resolution[0])]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1]-#ROI_BOUNDS[1][0])/dst_resolution[1])]),
		1,
	]
	"@type":        "ComputeFieldStage"
	dst_resolution: _
	processing_chunk_sizes: [max_chunk_size]
	processing_crop_pads: [[0, 0, 0]]
	fn: {
		"@type":  "align_with_online_finetuner"
		"@mode":  "partial"
		channel:  int
		sm:       int
		num_iter: int
		lr:       float
		src_zeros_sm_mult: number
		tgt_zeros_sm_mult: number
	}
}

#COMPUTE_FIELD_TEMPLATE: {
	_bounds: _ | *#ROI_BOUNDS
	_z_offset: int

	"@type": "build_compute_field_multistage_flow"
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [_bounds[0][0], _bounds[1][0], _bounds[2][0]]
		end_coord: [_bounds[0][1], _bounds[1][1], _bounds[2][1]]
	}
	stages: [
		// #STAGE_TMPL & {  // 24576
		// 	dst_resolution: [24576, 24576, #REFERENCE_RES[2]]
		// 	fn: #STAGE_TMPL.fn & _ALIGN_PARAMS["24576"]
		// },
		#STAGE_TMPL & {  // 12288
			dst_resolution: [12288, 12288, #REFERENCE_RES[2]]
			fn: #STAGE_TMPL.fn & _ALIGN_PARAMS["12288"]
		},
		#STAGE_TMPL & {  // 6144
			dst_resolution: [6144, 6144, #REFERENCE_RES[2]]
			fn: #STAGE_TMPL.fn & _ALIGN_PARAMS["6144"]
		},
	]

	tgt_offset: [0, 0, _z_offset]
	offset_resolution: #REFERENCE_RES

	src: {
		"@type": "build_cv_layer"
		path:    #AFFINE_ENC_PATH
		read_procs: [
			// {"@type": "lambda", "lambda_str": "lambda x: torch.where(x[1:2,:,:,:] == -103, 0, x)"},
			// {"@type": "lambda", "lambda_str": "lambda x: torch.where(x[0:1,:,:,:] == -120, 0, x)"},
			// {"@type": "lambda", "lambda_str": "lambda x: torch.where(x[0:1,:,:,:] == 112, 0, x)"}
		]
	}
	tgt: {
		"@type": "build_cv_layer"
		path:    #TARGET_PATH
		read_procs: [
			// {"@type": "lambda", "lambda_str": "lambda x: torch.where(x[1:2,:,:,:] == -103, 0, x)"},
			// {"@type": "lambda", "lambda_str": "lambda x: torch.where(x[0:1,:,:,:] == -120, 0, x)"},
			// {"@type": "lambda", "lambda_str": "lambda x: torch.where(x[0:1,:,:,:] == 112, 0, x)"}
		]
	}
	dst: {
		"@type": "build_cv_layer"
		path:    #COARSE_WO_AFFINE_FIELD_PATH
		info_reference_path: #AFFINE_FIELD_PATH
		info_add_scales: [[12288, 12288, 45]]
		info_add_scales_mode: "merge"
		on_info_exists: "overwrite"
	}
	tmp_layer_dir: #COARSE_WO_AFFINE_FIELD_PATH + "/tmp"
	tmp_layer_factory: {
		"@type":              "build_cv_layer"
		"@mode":              "partial"
		info_reference_path:  #AFFINE_FIELD_PATH
		info_add_scales:     [[12288, 12288, 45]]
		info_add_scales_mode: "merge"
		on_info_exists:       "overwrite"
	}
}

#COMPOSE_FIELD_TEMPLATE: {
	_bounds: _ | *#ROI_BOUNDS
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((_bounds[0][1]-_bounds[0][0])/dst_resolution[0])]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((_bounds[1][1]-_bounds[1][0])/dst_resolution[1])]),
		1,
	]

	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    "field"
		crop_pad: [256, 256, 0]
	}
	dst_resolution: _
	processing_chunk_sizes: [max_chunk_size]
	processing_crop_pads: [[0, 0, 0]]
	skip_intermediaries:    true
	expand_bbox_processing: true
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [_bounds[0][0], _bounds[1][0], _bounds[2][0]]
		end_coord: [_bounds[0][1], _bounds[1][1], _bounds[2][1]]
	}
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #AFFINE_INV_FIELD_PATH
		}
		field: {
			"@type": "build_cv_layer"
			path:    #COARSE_WO_AFFINE_FIELD_PATH
			data_resolution: [6144, 6144, 45]
			interpolation_mode: "field"
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #COARSE_INV_FIELD_PATH
		info_reference_path: op_kwargs.src.path
		on_info_exists:      "overwrite"
	}
}


#INVERT_FIELD_TEMPLATE: {
	_bounds: _ | *#ROI_BOUNDS
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((_bounds[0][1]-_bounds[0][0])/dst_resolution[0])]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((_bounds[1][1]-_bounds[1][0])/dst_resolution[1])]),
		1,
	]

	"@type": "build_subchunkable_apply_flow"
	fn: {"@type": "invert_field", "@mode": "partial"}
	dst_resolution: _
	processing_chunk_sizes: [max_chunk_size]
	processing_crop_pads: [[0, 0, 0]]
	skip_intermediaries:    true
	expand_bbox_processing: true
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [_bounds[0][0], _bounds[1][0], _bounds[2][0]]
		end_coord: [_bounds[0][1], _bounds[1][1], _bounds[2][1]]
	}
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #COARSE_INV_FIELD_PATH
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #COARSE_FIELD_PATH
		info_reference_path: op_kwargs.src.path
		info_add_scales:     [[6144, 6144, 45]]
		info_add_scales_mode: "replace"
		on_info_exists:      "overwrite"
	}
}

#WARP_ENC_TEMPLATE: {
	_bounds: _ | *#ROI_BOUNDS
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((_bounds[0][1]-_bounds[0][0])/dst_resolution[0])]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((_bounds[1][1]-_bounds[1][0])/dst_resolution[1])]),
		1,
	]

	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    "image"
		crop_pad: [256, 256, 0]
	}
	dst_resolution: _
	processing_chunk_sizes: [max_chunk_size]
	processing_crop_pads: [[0, 0, 0]]
	skip_intermediaries:    true
	expand_bbox_processing: true
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [_bounds[0][0], _bounds[1][0], _bounds[2][0]]
		end_coord: [_bounds[0][1], _bounds[1][1], _bounds[2][1]]
	}
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #UNALIGNED_ENC_PATH
		}
		field: {
			"@type": "build_cv_layer"
			path:    #COARSE_INV_FIELD_PATH
			data_resolution: [6144, 6144, 45]
			interpolation_mode: "field"
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #COARSE_ENC_PATH
		info_reference_path: op_kwargs.src.path
		info_add_scales:     [[6144, 6144, 45], [12288, 12288, 45]]
		info_add_scales_mode: "replace"
		on_info_exists:      "overwrite"
	}
}
