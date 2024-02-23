#SRC_PATH: "https://storage.googleapis.com/fafb_v15_aligned/v0/img/img_norm"
#DST_PATH: "gs://tmp_2w/tmp_data_x0-sharded"
#BBOX: {
   "@type": "BBox3D.from_coords"
   start_coord: [29696, 16384, 2000]
   end_coord: [29696 + 2048, 16384 + 2048, 2000 + 16]
   resolution: [16, 16, 40]
}

"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-research"
worker_cluster_name:    "zutils-x3"
worker_image:           "us-east1-docker.pkg.dev/zetta-research/containers-test/zetta_utils:tri-test-231107a"
worker_resources: {memory: "18560Mi"}
worker_replicas: 1
local_test:      true

target: {
   "@type": "build_subchunkable_apply_flow"
   bbox: #BBOX
   dst_resolution: [16, 16, 40]
   processing_chunk_sizes: [[1024, 1024, 16]]   // create 16M shards
   // We don't need intermediaries when https://github.com/ZettaAI/zetta_utils/issues/644 is fixed
   // skip_intermediaries: true
   level_intermediaries_dirs: ["file://~/.zetta_utils/tmp/"]
   fn: {
      "@type":    "lambda"
      lambda_str: "lambda src: src"
   }
   op_kwargs: {
      src: {
         "@type": "build_cv_layer"
         path:    #SRC_PATH
      }
   }
   dst: {
      // "@type":             "build_cv_layer"
      // cv_kwargs: {compress: false}  // need this if using CV
      "@type":             "build_ts_layer"
      path:                #DST_PATH
      info_reference_path: #SRC_PATH
      on_info_exists:      "overwrite"
      info_chunk_size: [256, 256, 4]
      info_sharding: {
         // "@type": "neuroglancer_uint64_sharded_v1"  // not needed - zutils will add this if missing
         hash: "identity"
         minishard_index_encoding: "gzip"
         data_encoding: "gzip"

         // Best practice:
         // `processing_chunk_sizes` need to contain whole shards
         //    -> processing_chunk_sizes is multiples of (chunk_size << preshift_bits)
         // Minishard index should be < 32KB. Which means <= 1500 chunks, each 24B, per minishard
         //    -> preshift_bits <= 10
         // Shard index should be < 8KB, so up to 512 minishards (each 16B)
         //    -> minishard_bits <= 9
         // Uncompressed size at different preshift_bits+minishard_bits when chunks are 256K:
         //    3b = 2M, 6b = 16M, 9b = 128M shards

         preshift_bits: 6     // [256, 256, 4] << preshift_bits = [1024, 1024, 16]
         minishard_bits: 0    // No need to have more than 1 minishard since preshift_bits is < 10
         shard_bits: 21       // Just need preshift_bits+minishard_bits+shard_bits <= 64
      }
      // remove unneeded scales
      info_add_scales: [dst_resolution]
      info_add_scales_mode: "replace"
   }
}
