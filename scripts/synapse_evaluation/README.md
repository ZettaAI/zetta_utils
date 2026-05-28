# Synapse detection + assignment pipeline

End-to-end: EM image → per-synapse pre/post partner assignments →
evaluation against ground truth → Neuroglancer link.

## Pipeline (CUE specs)

Four chained specs under [`specs/examples/inference/synapses/`](../../specs/examples/inference/synapses/),
run in order:

| # | Spec                            | Op                          | Reads                          | Writes                                        |
|---|---------------------------------|-----------------------------|--------------------------------|-----------------------------------------------|
| 1 | `inference_predictions.cue`     | `ModelInferencer`           | EM image                       | float32 `predictions` volume                  |
| 2 | `inference_segmentation.cue`    | `CCEdgeClear`               | predictions                    | uint `synseg` (one ID per CC)                 |
| 3 | `inference_assignment.cue`      | `AssignSynapsesOp`          | image + synseg + cellseg       | merged synseg + parquet metadata + lines      |
| 4 | `inference_scoring.cue`         | `SynapseScoreOp`            | predictions + synseg + parquet | parquet with `mean_score` / `median_score`    |

All four share the same `#DST_PREFIX`, `#RESOLUTION`, and `#BBOX`. Swap
the placeholders at the top of each file (image path, cellseg path,
model paths, destination bucket, and stage 1's `#INTERMEDIARIES_DIR`
scratch path) for your own.

```bash
zetta run specs/examples/inference/synapses/inference_predictions.cue
zetta run specs/examples/inference/synapses/inference_segmentation.cue
zetta run specs/examples/inference/synapses/inference_assignment.cue
zetta run specs/examples/inference/synapses/inference_scoring.cue
```

For non-trivial bboxes, point the `mazepa.execute_on_gcp_with_sqs`
target at a worker cluster — see [`specs/examples/inference/copy_data.cue`](../../specs/examples/inference/copy_data.cue)
for the worker-resource boilerplate.

**CAVE / Graphene cellseg:** if your cellseg is a Graphene layer (cells
are agglomerations of watershed supervoxels) and you want eval to compare
against proofread root IDs, use
[`inference_assignment_cave.cue`](../../specs/examples/inference/synapses/inference_assignment_cave.cue)
in place of `inference_assignment.cue` — same op, plus a `src_watershed`
layer wired up. The eval scripts then resolve anchor SVs to root IDs via
`--watershed-path` + `--cave-datastack`.

**TensorRT:** `ModelInferencer.tensorrt_enabled` defaults to `False`
because `tensorrt` is not a hard dep of this package. If your environment
has it installed and you want the speed-up, set
`tensorrt_enabled: true` in `inference_predictions.cue`.

## Detector / assignment variants

Pre/post is one symmetric axis — which side the detector predicts just
flips one parameter in the assignment spec.

| Detector predicts            | `synapse_type` | Typical `assign_type`         |
|------------------------------|----------------|-------------------------------|
| Synaptic cleft               | `cleft`        | `max`                         |
| Postsyn terminal (PST)       | `postsyn`      | `max`                         |
| PST with multi-pre input     | `postsyn`      | `pre_thresh`                  |
| Presyn ribbon (multi-post)   | `presyn`       | `post_thresh`                 |
| Presyn vesicle cloud         | `presyn`       | `max`                         |
| Any presyn/postsyn, no GPU   | `presyn`/`postsyn` | `dilate_nearest` (baseline) |

`dilate_nearest` is a no-network baseline: partner cells are the top-N
closest to the synapse mask in physical nm-space. Useful for
sanity-checking whether the assignment net learns anything beyond
geometric proximity.

For ribbons / vesicle clouds, also bump `candidate_dilation_xy` to ~5–10
voxels — the mask sits deep inside the presyn terminal and the default
(2) doesn't reach across the cleft to the partner cell.

## Evaluation (requires GT line annotations)

GT JSON: line annotations with `pointA` = presyn voxel, `pointB` =
postsyn voxel (Neuroglancer's standard line-annotation export).

- **[`sweep_threshold.py`](sweep_threshold.py)** — run stages 2–4 + eval
  at each of several segmentation thresholds; pick the best by F-beta.
- **[`eval_synapses.py`](eval_synapses.py)** — evaluate a single run
  (detection + assignment, Hungarian + key-based dedup); writes TP/FP/FN
  JSON annotation layers, optionally uploads a Neuroglancer state.
- **[`eval_detection.py`](eval_detection.py)** —
  assignment-invariant detection-only metric, reading centroids directly
  from the synseg layer.
- **[`eval_multi_partner.py`](eval_multi_partner.py)** — per-ribbon
  partner-set evaluation (set comparison, not Hungarian 1:1) for
  multi-partner synapses.
- **[`compare_runs.py`](compare_runs.py)** — 9-way TP/FP/FN split between
  two `eval_multi_partner.py` runs (e.g. network vs `dilate_nearest`).

Example end-to-end (flat segmentation, no CAVE):

```bash
# Pick the best threshold by running stages 2-4 + eval at each candidate
python scripts/synapse_evaluation/sweep_threshold.py \
    --gt-json   /path/to/gt_lines.json \
    --pred-path gs://YOUR-BUCKET/synapse-pipeline-example/predictions \
    --image-path   gs://zetta_lee_fly_cns_001_alignment/v1_sharded \
    --cellseg-path gs://zetta_lee_fly_cns_001_kisuk/final/v2/seg_semantic_split_membrane \
    --model-path   gs://YOUR-BUCKET/models/synapse-assignment.onnx \
    --output-prefix gs://YOUR-BUCKET/synapse-pipeline-example/sweep \
    --output-dir   ./sweep_out \
    --synapse-type postsyn
# --assignment-crop-pad defaults to [96, 96, 16] (voxels at --resolution).
# Don't reduce it without checking edge-blob scores: the assignment net
# loads its window from an image crop, and without padding the edge blobs
# see truncated windows and produce poor scores that get dropped at dedup.

# The sweep prints the eval command for the best threshold — run that with
# --upload-state to get a Neuroglancer link with TP/FP/FN annotation layers.
python scripts/synapse_evaluation/eval_synapses.py \
    --gt-json   /path/to/gt_lines.json \
    --pred-metadata gs://YOUR-BUCKET/synapse-pipeline-example/sweep/thr_<BEST>/assignment/metadata_scored \
    --segmentation-path gs://zetta_lee_fly_cns_001_kisuk/final/v2/seg_semantic_split_membrane \
    --synapse-type postsyn \
    --output-dir   ./eval_out \
    --upload-state \
    --image-path gs://zetta_lee_fly_cns_001_alignment/v1_sharded
```

For CAVE / ChunkedGraph datasets pass `--watershed-path` + `--cave-datastack`
instead of `--cellseg-path` / `--segmentation-path`. The eval scripts then
resolve watershed supervoxel IDs to proofread root IDs via CAVEclient.
