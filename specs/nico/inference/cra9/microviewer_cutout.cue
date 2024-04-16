MICROVIEWER_LINK: "{{microviewer_link}}"
#SRC_PATH:        "{{src_path}}"
#DST_PATH:        "{{dst_path}}"
#BBOX_SIZE:        [256, 256, 20]
#BBOX_CENTER:      [256, 256, 20]
#RESOLUTION:       [256, 256, 20]
#BBOX_RESOLUTION: [256, 256, 20]
#START_COORD: [
	#BBOX_CENTER[0] - #BBOX_SIZE[0] div 2,
	#BBOX_CENTER[1] - #BBOX_SIZE[1] div 2,
	#BBOX_CENTER[2] - #BBOX_SIZE[2] div 2,
]
#END_COORD: [
	#START_COORD[0] + #BBOX_SIZE[0],
	#START_COORD[1] + #BBOX_SIZE[1],
	#START_COORD[2] + #BBOX_SIZE[2],
]

#BBOX: {
	"@type":     "BBox3D.from_coords"
	start_coord: #START_COORD
	end_coord:   #END_COORD
	resolution:  #BBOX_RESOLUTION
}

"@type": "mazepa.execute"
target: {
	"@type":        "build_subchunkable_apply_flow"
	bbox:           #BBOX
	dst_resolution: #RESOLUTION
	processing_chunk_sizes: [#BBOX_SIZE]
	skip_intermediaries: true

	fn: {
		"@type":    "lambda"
		lambda_str: "lambda src: {'img': src, 'lbl': torch.zeros_like(src)}"
	}

	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #SRC_PATH
		}
	}

	dst: {
		"@type": "build_volumetric_layer_set"
		layers: {
			"img": {
				"@type": "build_cv_layer"
				path:    #DST_PATH + "/image"
				info_field_overrides: {
					"type":         "image"
					"data_type":    "uint8"
					"num_channels": 1
					"scales": [
						{
							"encoding":     "jpeg"
							"key":          "\(#RESOLUTION[0])_\(#RESOLUTION[1])_\(#RESOLUTION[2])"
							"resolution":   #RESOLUTION
							"voxel_offset": #START_COORD
							"size":         #BBOX_SIZE
							"chunk_sizes": [#BBOX_SIZE]
						},
					]
				},
				cv_kwargs: {
					"compress": false,
					"cdn_cache":   false,
					"delete_black_uploads": false,
				}
			}
			"lbl": {
				"@type": "build_cv_layer"
				path:    #DST_PATH + "/labels"
				info_field_overrides: {
					"type":         "segmentation"
					"data_type":    "uint8"
					"num_channels": 1
					"scales": [
						{
							"encoding":     "png"
							"key":          "initial"
							"resolution":   #RESOLUTION
							"voxel_offset": #START_COORD
							"size":         #BBOX_SIZE
							"chunk_sizes": [#BBOX_SIZE]
						},
					]
				}
				cv_kwargs: {
					"compress": false,
					"cdn_cache":   false,
					"delete_black_uploads": false,
				}
			}
		}
	}
}
