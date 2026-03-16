#!/usr/bin/env python3
"""Visualize filter stats from SegContactOp collect_filter_stats_only mode.

Interactive dashboard: 4x2+ grid of histograms with threshold sliders,
GT label coloring (stacked bars), and clickable neuroglancer links.

Usage:
    python scripts/visualize_filter_stats.py \
        --path gs://martin_exp/.../contacts_x2 \
        --output filter_histograms.html
"""

import argparse
import base64
import gzip
import json
import math
from string import Template

import fsspec
import numpy as np
import pandas as pd
from tqdm import tqdm


# (column, title, log_x, filter_direction, x_label, param_names)
# "min" = keep if >= thr, "max" = keep if <= thr, "minmax" = keep if min <= val <= max
# Order: 2-column grid, left=voxel/count, right=fraction/ratio
# None = empty cell placeholder
METRIC_CONFIGS = [
    ("min_size_vx", "Min Segment Size (vx)", True, "min", "seg_size_vx", ["min_seg_size_vx"]),
    "mesh",  # special: mesh availability panel
    ("min_best_overlap_vx", "Min Best Overlap (vx)", True, "min", "overlap_vx", ["min_overlap_vx"]),
    ("mean_affinity", "Mean Affinity", False, "minmax", "mean_affinity", ["min_mean_affinity", "max_mean_affinity"]),
    ("contact_count", "Contact Count (faces)", True, "minmax", "contact_vx", ["min_contact_vx", "max_contact_vx"]),
    ("interface_gt_fraction", "Interface GT Fraction", False, "min", "interface_gt_fraction", ["min_interface_gt_fraction"]),
    ("max_offtarget_vx", "Max Off-target (vx)", True, "max", "offtarget_vx", ["max_offtarget_vx"]),
    ("max_offtarget_fraction", "Max Off-target Fraction", False, "max", "offtarget_fraction", ["max_offtarget_fraction"]),
    ("max_unclaimed_vx", "Max Unclaimed (vx)", True, "max", "unclaimed_vx", ["max_unclaimed_vx"]),
    ("max_unclaimed_fraction", "Max Unclaimed Fraction", False, "max", "unclaimed_fraction", ["max_unclaimed_fraction"]),
]

N_SAMPLES_PER_BIN = 20

# Per-contact columns for the sample info table: (df_column, format, header_label)
TABLE_COLS = [
    ("min_size_vx", "d", "min_sz"),
    ("min_best_overlap_vx", "d", "min_ovlp"),
    ("max_offtarget_fraction", "f", "offtgt%"),
    ("max_unclaimed_fraction", "f", "unclm%"),
    ("contact_count", "d", "faces"),
    ("mean_affinity", "f", "aff"),
    ("interface_gt_fraction", "f", "iface_gt"),
]

# Extra columns appended to v array for filtering (not shown in sample tables)
EXTRA_V_COLS = [
    ("max_offtarget_vx", "d"),
    ("max_unclaimed_vx", "d"),
]

# Column name -> index in v array
_COL_TO_VI = {col: i for i, (col, _, _) in enumerate(TABLE_COLS)}
_COL_TO_VI.update({col: len(TABLE_COLS) + i for i, (col, _) in enumerate(EXTRA_V_COLS)})


def read_all_stats(path):
    stats_dir = f"{path}/filter_stats"
    fs, fs_path = fsspec.core.url_to_fs(stats_dir)
    parquet_files = [f for f in fs.ls(fs_path) if f.endswith(".parquet")]
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {stats_dir}")
    dfs = [pd.read_parquet(f, filesystem=fs) for f in tqdm(parquet_files, desc="Reading parquets")]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Read {len(dfs)} parquet files, {len(df)} total contacts")
    return df


def read_info(path):
    info_path = f"{path}/info"
    fs, fs_path = fsspec.core.url_to_fs(info_path)
    with fs.open(fs_path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def compute_bins(values, nbins, log_x):
    valid = ~np.isnan(values)
    v = values[valid]
    if len(v) == 0:
        return None

    if log_x:
        v_t = np.log10(v + 1)
        all_t = np.log10(values + 1)
    else:
        v_t, all_t = v, values

    edges = np.linspace(v_t.min(), v_t.max(), nbins + 1)
    bin_idx = np.full(len(values), -1, dtype=np.int32)
    bin_idx[valid] = np.clip(
        np.searchsorted(edges, all_t[valid], side="right") - 1, 0, nbins - 1
    )

    x_centers = ((edges[:-1] + edges[1:]) / 2).tolist()

    tick_vals = tick_texts = None
    if log_x:
        tick_vals, tick_texts = [], []
        # Always include 0 tick if in range
        t0 = np.log10(0 + 1)  # = 0
        if edges[0] <= t0 <= edges[-1]:
            tick_vals.append(float(t0))
            tick_texts.append("0")
        # Power-of-10 ticks with 1,2,5 subdivisions
        lo_pow = int(np.floor(np.log10(max(np.power(10, edges[0]) - 1, 1))))
        hi_pow = int(np.ceil(np.log10(max(np.power(10, edges[-1]) - 1, 1))))
        for p in range(lo_pow, hi_pow + 1):
            for mult in [1, 2, 5]:
                orig = mult * 10**p
                val = np.log10(orig + 1)
                if edges[0] <= val <= edges[-1]:
                    tick_vals.append(float(val))
                    tick_texts.append(f"{orig:g}")

    counts = [0] * nbins
    for b in bin_idx:
        if b >= 0:
            counts[b] += 1

    return {
        "edges": edges.tolist(),
        "x_centers": x_centers,
        "tick_vals": tick_vals,
        "tick_texts": tick_texts,
        "unfiltered_counts": counts,
        "bin_indices": bin_idx.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize filter stats")
    parser.add_argument("--path", required=True, help="Seg_contact layer path")
    parser.add_argument("--output", default="filter_histograms.html", help="Output HTML")
    parser.add_argument("--nbins", type=int, default=50, help="Number of bins")
    args = parser.parse_args()

    df = read_all_stats(args.path)
    info = read_info(args.path)

    # Map chunk coords to integer indices
    chunk_coords = df["chunk_coord"].values if "chunk_coord" in df.columns else None
    chunk_labels = []
    chunk_id_map = {}
    if chunk_coords is not None:
        for cc in chunk_coords:
            if cc not in chunk_id_map:
                chunk_id_map[cc] = len(chunk_id_map)
            chunk_labels.append(chunk_id_map[cc])
    n_chunks = len(chunk_id_map)

    contacts_js = []
    for i, (_, row) in tqdm(enumerate(df.iterrows()), total=len(df), desc="Building contacts"):
        gt = str(row.get("gt_merge_label", "unknown")) if pd.notna(row.get("gt_merge_label")) else "unknown"
        entry = {
            "a": str(int(row["seg_a"])),
            "b": str(int(row["seg_b"])),
            "c": [round(float(row["com_x"]), 1), round(float(row["com_y"]), 1), round(float(row["com_z"]), 1)],
            "g": {"merge": 0, "no_merge": 1, "unknown": 2}.get(gt, 2),
            "k": chunk_labels[i] if chunk_labels else 0,
            "cc": str(row["chunk_coord"]) if "chunk_coord" in df.columns else "",
        }
        vals = []
        for col, fmt, _ in TABLE_COLS:
            if col in df.columns and pd.notna(row.get(col)):
                v = row[col]
                vals.append(int(v) if fmt == "d" else round(float(v), 3))
            else:
                vals.append(None)
        for col, fmt in EXTRA_V_COLS:
            if col in df.columns and pd.notna(row.get(col)):
                v = row[col]
                vals.append(int(v) if fmt == "d" else round(float(v), 3))
            else:
                vals.append(None)
        entry["v"] = vals
        if "both_meshes" in df.columns:
            entry["m"] = 1 if row.get("both_meshes") else 0
        if "contact_faces_nm" in df.columns and pd.notna(row.get("contact_faces_nm")):
            entry["f"] = json.loads(row["contact_faces_nm"])
        if "gt_refs_a" in df.columns and pd.notna(row.get("gt_refs_a")):
            entry["ra"] = [str(x) for x in json.loads(row["gt_refs_a"])]
        if "gt_refs_b" in df.columns and pd.notna(row.get("gt_refs_b")):
            entry["rb"] = [str(x) for x in json.loads(row["gt_refs_b"])]
        if "has_nucleus" in df.columns:
            entry["n"] = 1 if row.get("has_nucleus") else 0
        contacts_js.append(entry)

    has_mesh_data = "both_meshes" in df.columns
    has_nucleus_data = "has_nucleus" in df.columns
    metrics_js = []
    for cfg in METRIC_CONFIGS:
        if cfg is None:
            metrics_js.append(None)
            continue
        if cfg == "mesh":
            metrics_js.append("mesh" if has_mesh_data else None)
            continue
        col, title, log_x, filt_dir, x_label, param_names = cfg
        if col not in df.columns:
            print(f"  Skipping {col} (not in data)")
            continue
        values = df[col].values.astype(float)
        bins = compute_bins(values, args.nbins, log_x)
        if bins is None:
            print(f"  Skipping {col} (all NaN)")
            continue

        valid = values[~np.isnan(values)]
        thr_min = float(valid.min())
        thr_max = float(valid.max())

        metrics_js.append({
            "col": col, "title": title, "log_x": log_x,
            "filter_dir": filt_dir, "vi": _COL_TO_VI[col],
            "x_label": x_label, "param_names": param_names,
            "thr_min": thr_min, "thr_max": thr_max,
            **bins,
        })
        print(f"  {title}: done")

    resolution = info.get("resolution", [16, 16, 40])
    ng_info = {
        "resolution": resolution,
        "base_url": "https://zetta-portal.vercel.app/?ng=Spelunker",
    }
    for key in ["image_path", "affinity_path", "segmentation_path", "ground_truth_path", "nucleus_path"]:
        val = info.get(key)
        if val:
            ng_info[key] = val

    print("  Serializing JSON...")
    data_json = json.dumps(
        {"contacts": contacts_js, "metrics": metrics_js, "ngInfo": ng_info,
         "nChunks": n_chunks,
         "tableHeaders": [h for _, _, h in TABLE_COLS]},
        separators=(",", ":"),
    )

    # Compress: JSON -> gzip -> base64
    json_bytes = data_json.encode("utf-8")
    print(f"  Compressing {len(json_bytes)/1e6:.1f}MB JSON...")
    compressed = gzip.compress(json_bytes, compresslevel=6)
    data_b64 = base64.b64encode(compressed).decode("ascii")
    ratio = len(json_bytes) / len(data_b64)
    print(f"  Data: {len(json_bytes)/1e6:.1f}MB JSON -> {len(compressed)/1e6:.1f}MB gzip -> {len(data_b64)/1e6:.1f}MB base64 ({ratio:.1f}x)")

    # Build dataset info for header
    res = info.get("resolution", [])
    vox_off = info.get("voxel_offset", [])
    size = info.get("size", [])
    chunk_size = info.get("chunk_size", [])
    bbox_end = [vox_off[i] + size[i] for i in range(3)] if len(vox_off) == 3 and len(size) == 3 else []
    total_chunks = 1
    if len(size) == 3 and len(chunk_size) == 3:
        for i in range(3):
            total_chunks *= math.ceil(size[i] / chunk_size[i])

    dataset_info = {
        "path": args.path,
        "segmentation": info.get("segmentation_path", ""),
        "ground_truth": info.get("ground_truth_path", ""),
        "resolution": res,
        "bbox_start": vox_off,
        "bbox_end": bbox_end,
        "size": size,
        "chunk_size": chunk_size,
        "total_chunks": total_chunks,
        "sampled_chunks": n_chunks,
    }

    print("  Building HTML...")
    html = _build_html(data_b64, len(df), dataset_info)
    print(f"  Writing {len(html)/1e6:.1f}MB to {args.output}...")
    with open(args.output, "w") as f:
        f.write(html)
    print(f"Wrote {args.output}")


def _build_html(data_b64, n_contacts, dataset_info):
    di = dataset_info
    res_str = " x ".join(str(r) for r in di["resolution"]) + " nm" if di["resolution"] else "?"
    bbox_str = (
        f'[{", ".join(str(v) for v in di["bbox_start"])}] to [{", ".join(str(v) for v in di["bbox_end"])}]'
        if di["bbox_start"] and di["bbox_end"] else "?"
    )
    size_str = " x ".join(str(s) for s in di["size"]) + " vx" if di["size"] else "?"
    chunk_str = " x ".join(str(s) for s in di["chunk_size"]) if di["chunk_size"] else "?"

    return _HTML_TEMPLATE.substitute(
        data_b64=data_b64,
        n_contacts=n_contacts,
        n_samples=N_SAMPLES_PER_BIN,
        path=di["path"],
        segmentation=di["segmentation"],
        ground_truth=di["ground_truth"],
        res_str=res_str,
        bbox_str=bbox_str,
        size_str=size_str,
        chunk_str=chunk_str,
        sampled_chunks=di["sampled_chunks"],
        total_chunks=di["total_chunks"],
    )


_HTML_TEMPLATE = Template("""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Filter Stats Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
* { box-sizing: border-box; }
body { font-family: sans-serif; margin: 12px; background: #fafafa; }
h1 { margin: 0 0 4px 0; font-size: 30px; }
.info-bar { font-size: 18px; color: #444; background: #eef2f7; border: 1px solid #ccd;
             border-radius: 4px; padding: 6px 10px; margin-bottom: 8px; line-height: 1.6; }
.info-bar b { color: #222; }
.info-bar .info-path { font-family: monospace; font-size: 16px; color: #555; word-break: break-all; }
.header { display: flex; align-items: center; gap: 20px; margin-bottom: 10px; flex-wrap: wrap; }
.summary { color: #555; font-size: 21px; }
.gt-toggle { display: flex; align-items: center; gap: 8px; font-size: 20px; }
.gt-toggle label { cursor: pointer; }
.legend-info { font-size: 18px; color: #666; }
.legend-info span { font-weight: bold; padding: 1px 6px; border-radius: 3px; margin: 0 2px; }
.legend-info .lg-merge { background: rgba(76,175,80,0.4); }
.legend-info .lg-nomerge { background: rgba(229,57,53,0.4); }
.legend-info .lg-unknown { background: rgba(158,158,158,0.35); }
.legend-info .lg-correct { background: rgba(76,175,80,0.5); }
.legend-info .lg-wrong { background: rgba(229,57,53,0.5); }
.legend-info .lg-unclear { background: rgba(255,193,7,0.45); }
.legend-info .lg-unannotated { background: rgba(158,158,158,0.35); }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.cell { background: white; border: 1px solid #ddd; border-radius: 6px;
         padding: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.ctrl-row { display: flex; align-items: center; gap: 8px;
             padding: 4px 8px; background: #f5f5f5; border-radius: 4px; font-size: 18px;
             margin-top: 2px; }
.ctrl-row:first-of-type { margin-top: 0; }
.ctrl-row input[type=range] { flex: 1; min-width: 0; }
.ctrl-row input[type=number] { width: 95px; padding: 2px 4px; border: 1px solid #ccc;
                                 border-radius: 3px; font-size: 18px; }
.ctrl-row .dir-label { font-size: 16px; color: #555; white-space: nowrap; min-width: 48px; font-family: monospace; }
.link-panel { background: #f0f4f8; border: 1px solid #ccd; border-radius: 4px;
               padding: 6px 8px; margin-top: 4px; display: none;
               font-size: 16px; line-height: 1.5; max-height: 400px; overflow-y: auto; }
.link-panel a { color: #1a0dab; }
.link-panel .lbl-merge { color: #2e7d32; font-weight: bold; }
.link-panel .lbl-no_merge { color: #c62828; font-weight: bold; }
.link-panel .lbl-unknown { color: #888; font-weight: bold; }
.link-panel .panel-header { font-weight: bold; margin-bottom: 2px; display: flex;
             align-items: center; justify-content: space-between; }
.link-panel .panel-close { cursor: pointer; color: #888; font-size: 24px; line-height: 1;
             padding: 0 4px; border-radius: 3px; }
.link-panel .panel-close:hover { background: #ddd; color: #333; }
.link-panel .cat-header { font-weight: bold; margin-top: 6px; padding: 2px 4px; border-radius: 3px; }
.link-panel .cat-merge { background: rgba(76,175,80,0.15); color: #2e7d32; }
.link-panel .cat-nomerge { background: rgba(229,57,53,0.15); color: #c62828; }
.link-panel .cat-unknown { background: rgba(158,158,158,0.15); color: #666; }
.link-panel table { border-collapse: collapse; width: 100%; margin-top: 2px; margin-bottom: 4px; }
.link-panel th { background: #e8ecf0; padding: 2px 4px; text-align: right; font-size: 15px;
                  border: 1px solid #ccd; white-space: nowrap; }
.link-panel td { padding: 2px 4px; text-align: right; border: 1px solid #dde; font-size: 15px;
                  white-space: nowrap; }
.link-panel td:first-child { text-align: left; }
.link-panel th:first-child { text-align: left; }
.link-panel a.ngl-link { cursor: pointer; }
.link-panel tr.ngl-active { background: rgba(74, 144, 217, 0.15); }
.progress-wrap { margin: 40px auto; max-width: 500px; text-align: center; }
.progress-text { font-size: 21px; color: #555; margin-bottom: 8px; }
.progress-bar-bg { height: 4px; background: #e0e0e0; border-radius: 2px; overflow: hidden; }
.progress-bar-inner { height: 100%; width: 30%; background: #4a90d9; border-radius: 2px;
                       animation: progress-slide 1.2s ease-in-out infinite; }
@keyframes progress-slide { 0% { margin-left: -30%; } 100% { margin-left: 100%; } }
.ann-btn { display: inline-block; width: 24px; height: 24px; line-height: 24px; text-align: center;
           border: 1px solid #ccc; border-radius: 3px; cursor: pointer; font-size: 14px;
           background: #f8f8f8; margin: 0 1px; user-select: none; }
.ann-btn:hover { background: #e8e8e8; }
.ann-btn.active-correct { background: #4caf50; color: white; border-color: #388e3c; }
.ann-btn.active-wrong { background: #e53935; color: white; border-color: #c62828; }
.ann-btn.active-unclear { background: #ffc107; color: #333; border-color: #f9a825; }
.ann-note { width: 80px; padding: 1px 3px; border: 1px solid #ccc; border-radius: 3px; font-size: 13px; }
.ann-summary { background: #f0f4f8; padding: 6px 8px; margin-top: 4px; border-radius: 3px;
               font-size: 14px; line-height: 1.6; }
.ann-toolbar { display: flex; align-items: center; gap: 6px; }
.ann-toolbar button { padding: 4px 10px; border: 1px solid #aab; border-radius: 4px;
                      background: #f0f4f8; cursor: pointer; font-size: 16px; }
.ann-toolbar button:hover { background: #dde4ec; }
.page-nav { display: flex; align-items: center; gap: 6px; margin: 4px 0; font-size: 14px; }
.page-nav button { padding: 2px 8px; border: 1px solid #bbb; border-radius: 3px;
                   background: #f5f5f5; cursor: pointer; font-size: 13px; }
.page-nav button:hover { background: #e0e0e0; }
.page-nav button:disabled { opacity: 0.4; cursor: default; }
.page-nav select { padding: 1px 4px; border: 1px solid #bbb; border-radius: 3px; font-size: 13px; }
</style>
</head>
<body>
<div class="info-bar">
    <b>Dataset:</b> <span class="info-path">$path</span><br>
    <b>Segmentation:</b> <span class="info-path">$segmentation</span>
    &nbsp; <b>Ground Truth:</b> <span class="info-path">$ground_truth</span><br>
    <b>Resolution:</b> $res_str
    &nbsp; <b>BBox:</b> $bbox_str ($size_str)
    &nbsp; <b>Chunk:</b> $chunk_str
    &nbsp; <b>Chunks:</b> $sampled_chunks sampled / $total_chunks total
</div>
<div class="header">
    <h1>Filter Stats Dashboard</h1>
    <div class="summary" id="summary">$n_contacts contacts &mdash; loading data...</div>
    <div class="gt-toggle">
        GT fill:
        <label><input type="radio" name="gt_mode" value="unfiltered" checked> Unfiltered</label>
        <label><input type="radio" name="gt_mode" value="filtered"> Filtered</label>
    </div>
    <div class="gt-toggle">
        Y axis:
        <label><input type="radio" name="y_scale" value="log" checked> Log</label>
        <label><input type="radio" name="y_scale" value="linear"> Linear</label>
    </div>
    <div class="legend-info">
        <span class="lg-merge" id="legend0">merge</span>
        <span class="lg-nomerge" id="legend1">no_merge</span>
        <span class="lg-unknown" id="legend2">unknown</span>
        <span class="lg-unknown" id="legend3" style="display:none">unannotated</span>
        | gray outline = all, black outline = filtered
    </div>
    <div class="gt-toggle">
        Bar color:
        <label><input type="radio" name="bar_color" value="gt" checked> GT labels</label>
        <label><input type="radio" name="bar_color" value="annotations"> Annotations</label>
    </div>
    <div class="ann-toolbar">
        <button id="ann_export">Export annotations</button>
        <button id="ann_import">Import annotations</button>
        <button id="ann_clear">Clear annotations</button>
        <input type="file" id="ann_file_input" accept=".json" style="display:none">
    </div>
</div>
<div id="progress-wrap" class="progress-wrap">
    <div class="progress-text" id="progress-text">Loading...</div>
    <div class="progress-bar-bg"><div class="progress-bar-inner"></div></div>
</div>
<div class="grid" id="grid"></div>
<div id="all-annotations" class="cell" style="margin-top:12px; display:none;"></div>

<script>
// --- Log1p transform: log10(x+1) for log-x axes, handles 0 naturally ---
function toLog1p(x) { return Math.log10(x + 1); }
function fromLog1p(t) { return Math.pow(10, t) - 1; }

// --- Globals (populated after decompression) ---
var contacts, metrics, ngInfo, nContacts, tableHeaders;
var N_SAMPLES = $n_samples;
var thresholds = {};
var gtMode = "unfiltered";
var yScale = "log";
var meshFilter = false;
var hasMeshData = false;
var nucleusFilter = false;
var hasNucleusData = false;
var chunkHasNucleus = {}; // chunkId -> true
var currentMask = null;
var highlightedBin = {};
var chunkSortOrder, chunkRankMap, chunkUnfiltGt, chunkUnfiltTotals, chunkUnfiltBases1, chunkUnfiltBases2;

var GT_NAMES = ["merge", "no_merge", "unknown"];
var GT_COLORS = [
    "rgba(76,175,80,0.45)",
    "rgba(229,57,53,0.45)",
    "rgba(158,158,158,0.35)"
];
var ANN_TRACE_COLORS = [
    "rgba(76,175,80,0.55)",
    "rgba(229,57,53,0.55)",
    "rgba(255,193,7,0.45)",
    "rgba(158,158,158,0.35)"
];
var barColorMode = "gt";
var excludedChunkRanks = {};
var datasetPath = "$path";
var annotations = {};
var STORAGE_KEY = "filter_stats_annotations_" + datasetPath;

function contactKey(c) { return c.cc + "|" + c.a + "|" + c.b; }

function saveAnnotations() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations)); } catch(e) {}
    if (typeof refreshAllAnnotations === "function" && allAnnEl) refreshAllAnnotations();
}

function loadAnnotations() {
    try {
        var s = localStorage.getItem(STORAGE_KEY);
        if (s) annotations = JSON.parse(s);
    } catch(e) { annotations = {}; }
}

function exportAnnotations() {
    var blob = new Blob([JSON.stringify(annotations, null, 2)], {type: "application/json"});
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "annotations.json";
    a.click();
    URL.revokeObjectURL(a.href);
}

function importAnnotations(file) {
    var reader = new FileReader();
    reader.onload = function(e) {
        try {
            var imported = JSON.parse(e.target.result);
            for (var k in imported) annotations[k] = imported[k];
            saveAnnotations();
            document.querySelectorAll('.link-panel').forEach(function(p) {
                if (p.style.display !== "none" && p._refresh) p._refresh();
            });
        } catch(err) { alert("Invalid JSON file"); }
    };
    reader.readAsText(file);
}

// --- Decompress gzipped base64 data ---
async function decompressData(b64) {
    var raw = atob(b64);
    var bytes = new Uint8Array(raw.length);
    for (var i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    var blob = new Blob([bytes]);
    var stream = blob.stream().pipeThrough(new DecompressionStream("gzip"));
    var text = await new Response(stream).text();
    return JSON.parse(text);
}

// --- NGL URL builder ---
function buildNgUrl(c) {
    var res = ngInfo.resolution;
    var pos = [c.c[0]/res[0], c.c[1]/res[1], c.c[2]/res[2]];
    var dims = {"x":[res[0]*1e-9,"m"],"y":[res[1]*1e-9,"m"],"z":[res[2]*1e-9,"m"]};
    var layers = [];
    if (ngInfo.image_path) {
        var src = ngInfo.image_path.indexOf("://")>=0 ? ngInfo.image_path : "precomputed://"+ngInfo.image_path;
        layers.push({"type":"image","source":src,"tab":"source","name":"Image"});
    }
    if (ngInfo.affinity_path) {
        var src = ngInfo.affinity_path.startsWith("precomputed://") ? ngInfo.affinity_path : "precomputed://"+ngInfo.affinity_path;
        layers.push({"type":"image","source":src,"tab":"source","channelDimensions":{"c^":[1,""]},"name":"Affinity"});
    }
    if (ngInfo.segmentation_path) {
        var src = ngInfo.segmentation_path.startsWith("precomputed://") ? ngInfo.segmentation_path : "precomputed://"+ngInfo.segmentation_path;
        layers.push({"type":"segmentation","source":src,"tab":"segments","name":"Segmentation",
                     "segments":[""+c.a,""+c.b]});
    }
    if (ngInfo.ground_truth_path) {
        var src = ngInfo.ground_truth_path.startsWith("precomputed://") ? ngInfo.ground_truth_path : "precomputed://"+ngInfo.ground_truth_path;
        var gtSegs = [];
        if (c.ra) for (var gi = 0; gi < c.ra.length; gi++) gtSegs.push(c.ra[gi]);
        if (c.rb) for (var gi = 0; gi < c.rb.length; gi++) gtSegs.push(c.rb[gi]);
        var gtLayer = {"type":"segmentation","source":src,"tab":"source","name":"Ground Truth"};
        if (gtSegs.length > 0) gtLayer.segments = gtSegs;
        layers.push(gtLayer);
    }
    if (ngInfo.nucleus_path) {
        var src = ngInfo.nucleus_path.startsWith("precomputed://") ? ngInfo.nucleus_path : "precomputed://"+ngInfo.nucleus_path;
        layers.push({"type":"segmentation","source":src,"tab":"source","name":"Nuclei","visible":false});
    }
    if (c.f && c.f.length > 0) {
        var anns = [];
        for (var i = 0; i < c.f.length; i++) {
            var f = c.f[i];
            anns.push({"point":[f[0]/res[0],f[1]/res[1],f[2]/res[2]],"type":"point","id":"c_"+i});
        }
        layers.push({
            "type":"annotation",
            "source":{"url":"local://annotations","transform":{"outputDimensions":dims}},
            "tool":"annotatePoint","tab":"annotations","annotations":anns,"name":"Contacts"
        });
    }
    var state = {
        "dimensions":dims,"position":pos,
        "crossSectionScale":0.1,"projectionScale":220,
        "layers":layers,
        "selectedLayer":{"visible":true,"layer":ngInfo.segmentation_path?"Segmentation":"Image"},
        "layout":"xy-3d","showSlices":false
    };
    return ngInfo.base_url + "#!" + encodeURIComponent(JSON.stringify(state));
}

// --- Filter mask (compares against actual per-contact values, not bin centers) ---
function computeFilterMask() {
    var mask = new Uint8Array(nContacts);
    for (var i = 0; i < nContacts; i++) mask[i] = 1;
    metrics.forEach(function(m) {
        if (!m || m === "mesh") return;
        var vi = m.vi;

        if (m.filter_dir === "minmax") {
            var thrLo = thresholds[m.col + "_lo"];
            var thrHi = thresholds[m.col + "_hi"];
            for (var i = 0; i < nContacts; i++) {
                if (!mask[i]) continue;
                var val = contacts[i].v[vi];
                if (val === null || val < thrLo || val > thrHi) mask[i] = 0;
            }
        } else {
            var thr = thresholds[m.col];
            for (var i = 0; i < nContacts; i++) {
                if (!mask[i]) continue;
                var val = contacts[i].v[vi];
                if (val === null) { mask[i] = 0; continue; }
                if (m.filter_dir === "min" ? val < thr : val > thr) mask[i] = 0;
            }
        }
    });
    if (meshFilter && hasMeshData) {
        for (var i = 0; i < nContacts; i++) {
            if (mask[i] && !contacts[i].m) mask[i] = 0;
        }
    }
    if (nucleusFilter && hasNucleusData) {
        for (var i = 0; i < nContacts; i++) {
            if (mask[i] && chunkHasNucleus[contacts[i].k]) mask[i] = 0;
        }
    }
    if (Object.keys(excludedChunkRanks).length > 0) {
        for (var i = 0; i < nContacts; i++) {
            if (mask[i] && excludedChunkRanks[chunkRankMap[contacts[i].k]]) mask[i] = 0;
        }
    }
    return mask;
}

// --- GT bin counts ---
function computeGtBinCounts(mi, useFiltered) {
    var m = metrics[mi];
    var bins = m.bin_indices;
    var nb = m.unfiltered_counts.length;
    var counts = [new Array(nb).fill(0), new Array(nb).fill(0), new Array(nb).fill(0)];
    for (var i = 0; i < nContacts; i++) {
        if (bins[i] < 0) continue;
        if (useFiltered && !currentMask[i]) continue;
        counts[contacts[i].g][bins[i]]++;
    }
    return counts;
}

function computeAnnBinCounts(mi, useFiltered) {
    var m = metrics[mi];
    var bins = m.bin_indices;
    var nb = m.unfiltered_counts.length;
    var correct = new Array(nb).fill(0);
    var wrong = new Array(nb).fill(0);
    var unclear = new Array(nb).fill(0);
    var unannotated = new Array(nb).fill(0);
    for (var i = 0; i < nContacts; i++) {
        if (bins[i] < 0) continue;
        if (useFiltered && !currentMask[i]) continue;
        var ann = annotations[contactKey(contacts[i])];
        var b = bins[i];
        if (ann && ann.label) {
            if (ann.label === "correct") correct[b]++;
            else if (ann.label === "wrong") wrong[b]++;
            else unclear[b]++;
        } else {
            unannotated[b]++;
        }
    }
    return [correct, wrong, unclear, unannotated];
}

// --- Percentage labels for GT bars ---
function gtPctTexts(gt0, gt1, gt2, gt3) {
    var n = gt0.length;
    var t0 = new Array(n), t1 = new Array(n), t2 = new Array(n);
    var t3 = gt3 ? new Array(n) : null;
    for (var i = 0; i < n; i++) {
        var tot = gt0[i] + gt1[i] + gt2[i] + (gt3 ? gt3[i] : 0);
        if (tot === 0) { t0[i] = ""; t1[i] = ""; t2[i] = ""; if (t3) t3[i] = ""; continue; }
        var p0 = 100 * gt0[i] / tot, p1 = 100 * gt1[i] / tot, p2 = 100 * gt2[i] / tot;
        t0[i] = gt0[i] > 0 ? p0.toFixed(1) + "%" : "";
        t1[i] = gt1[i] > 0 ? p1.toFixed(1) + "%" : "";
        t2[i] = gt2[i] > 0 ? p2.toFixed(1) + "%" : "";
        if (t3) { var p3 = 100 * gt3[i] / tot; t3[i] = gt3[i] > 0 ? p3.toFixed(1) + "%" : ""; }
    }
    return t3 ? [t0, t1, t2, t3] : [t0, t1, t2];
}

// --- Y-axis scale update ---
var allChartIds = [];
function updateYScales() {
    var update;
    if (yScale === "log") {
        update = {"yaxis.type":"log", "yaxis.dtick":1, "yaxis.autorange":"max", "yaxis.range":[0]};
    } else {
        update = {"yaxis.type":"linear", "yaxis.dtick":null, "yaxis.autorange":true, "yaxis.range":null};
    }
    allChartIds.forEach(function(id) {
        if (document.getElementById(id)) Plotly.relayout(id, update);
    });
}

// --- Update all ---
function updateAll() {
    currentMask = computeFilterMask();
    var nFiltered = 0;
    for (var i = 0; i < nContacts; i++) nFiltered += currentMask[i];
    document.getElementById("summary").textContent =
        nFiltered + " / " + nContacts + " contacts pass all filters";

    var useFiltered = (gtMode === "filtered");
    var useAnn = (barColorMode === "annotations");

    metrics.forEach(function(m, mi) {
        if (!m || m === "mesh") return;
        var nb = m.unfiltered_counts.length;
        var bins = m.bin_indices;

        var filtCounts = new Array(nb).fill(0);
        for (var i = 0; i < nContacts; i++) {
            if (currentMask[i] && bins[i] >= 0) filtCounts[bins[i]]++;
        }

        var stacked = useAnn ? computeAnnBinCounts(mi, useFiltered) : computeGtBinCounts(mi, useFiltered);
        var bases1 = stacked[0].slice();
        var bases2 = stacked[0].map(function(v,i) { return v + stacked[1][i]; });
        var bases3 = stacked[0].map(function(v,i) { return v + stacked[1][i] + stacked[2][i]; });
        var s3 = useAnn ? stacked[3] : new Array(nb).fill(0);
        var pct = useAnn ? gtPctTexts(stacked[0], stacked[1], stacked[2], stacked[3]) : gtPctTexts(stacked[0], stacked[1], stacked[2]);

        Plotly.restyle("chart_" + mi, {
            y: [stacked[0], stacked[1], stacked[2], s3, m.unfiltered_counts, filtCounts],
            base: [null, bases1, bases2, bases3, null, null],
            text: [pct[0], pct[1], pct[2], pct[3] || null, null, null],
        });

        // Threshold lines
        var layoutUpdate = {};
        if (m.filter_dir === "minmax") {
            var thrLo = m.log_x ? toLog1p(thresholds[m.col+"_lo"]) : thresholds[m.col+"_lo"];
            var thrHi = m.log_x ? toLog1p(thresholds[m.col+"_hi"]) : thresholds[m.col+"_hi"];
            layoutUpdate["shapes[0].x0"] = thrLo;
            layoutUpdate["shapes[0].x1"] = thrLo;
            layoutUpdate["shapes[1].x0"] = thrHi;
            layoutUpdate["shapes[1].x1"] = thrHi;
        } else {
            var thrVal = m.log_x ? toLog1p(thresholds[m.col]) : thresholds[m.col];
            layoutUpdate["shapes[0].x0"] = thrVal;
            layoutUpdate["shapes[0].x1"] = thrVal;
        }

        // Highlight shape
        var hlIdx = m.filter_dir === "minmax" ? 2 : 1;
        if (highlightedBin[mi] !== undefined) {
            var hb = highlightedBin[mi];
            layoutUpdate["shapes["+hlIdx+"].x0"] = m.edges[hb];
            layoutUpdate["shapes["+hlIdx+"].x1"] = m.edges[hb + 1];
            layoutUpdate["shapes["+hlIdx+"].visible"] = true;
        }

        Plotly.relayout("chart_" + mi, layoutUpdate);
    });

    // Update summary chart
    var sAll = [0,0,0,0], sFilt = [0,0,0,0];
    for (var i = 0; i < nContacts; i++) {
        var sc;
        if (useAnn) {
            var ann = annotations[contactKey(contacts[i])];
            sc = (ann && ann.label === "correct") ? 0 : (ann && ann.label === "wrong") ? 1 : (ann && ann.label) ? 2 : 3;
        } else {
            sc = contacts[i].g;
        }
        sAll[sc]++;
        if (currentMask[i]) sFilt[sc]++;
    }
    var sPct = useAnn
        ? gtPctTexts([sAll[0],sFilt[0]], [sAll[1],sFilt[1]], [sAll[2],sFilt[2]], [sAll[3],sFilt[3]])
        : gtPctTexts([sAll[0],sFilt[0]], [sAll[1],sFilt[1]], [sAll[2],sFilt[2]]);
    Plotly.restyle("chart_summary", {
        y: [[sAll[0], sFilt[0]], [sAll[1], sFilt[1]], [sAll[2], sFilt[2]], [sAll[3], sFilt[3]]],
        text: [sPct[0], sPct[1], sPct[2], sPct[3] || null],
    });

    // Update mesh chart
    if (hasMeshData && document.getElementById("chart_mesh")) {
        var useF = (gtMode === "filtered");
        var stackCats = [[0,0,0,0],[0,0,0,0],[0,0,0,0]]; // [mesh_cat][stack_cat]
        var unfiltCats = [0,0,0];
        var filtCats = [0,0,0];
        for (var i = 0; i < nContacts; i++) {
            var meshCat = contacts[i].m ? 1 : 2;
            unfiltCats[0]++;
            unfiltCats[meshCat]++;
            if (currentMask[i]) {
                filtCats[0]++;
                filtCats[meshCat]++;
            }
            if (!useF || currentMask[i]) {
                var msc;
                if (useAnn) {
                    var ann = annotations[contactKey(contacts[i])];
                    msc = (ann && ann.label === "correct") ? 0 : (ann && ann.label === "wrong") ? 1 : (ann && ann.label) ? 2 : 3;
                } else {
                    msc = contacts[i].g;
                }
                stackCats[0][msc]++;
                stackCats[meshCat][msc]++;
            }
        }
        var b1 = [stackCats[0][0], stackCats[1][0], stackCats[2][0]];
        var b2 = [stackCats[0][0]+stackCats[0][1], stackCats[1][0]+stackCats[1][1], stackCats[2][0]+stackCats[2][1]];
        var b3 = [stackCats[0][0]+stackCats[0][1]+stackCats[0][2], stackCats[1][0]+stackCats[1][1]+stackCats[1][2], stackCats[2][0]+stackCats[2][1]+stackCats[2][2]];
        var ms3 = [stackCats[0][3],stackCats[1][3],stackCats[2][3]];
        var mPct = useAnn
            ? gtPctTexts(b1, [stackCats[0][1],stackCats[1][1],stackCats[2][1]], [stackCats[0][2],stackCats[1][2],stackCats[2][2]], ms3)
            : gtPctTexts(b1, [stackCats[0][1],stackCats[1][1],stackCats[2][1]], [stackCats[0][2],stackCats[1][2],stackCats[2][2]]);
        Plotly.restyle("chart_mesh", {
            y: [[stackCats[0][0],stackCats[1][0],stackCats[2][0]],
                [stackCats[0][1],stackCats[1][1],stackCats[2][1]],
                [stackCats[0][2],stackCats[1][2],stackCats[2][2]],
                ms3,
                unfiltCats,
                filtCats],
            base: [null, b1, b2, b3, null, null],
            text: [mPct[0], mPct[1], mPct[2], mPct[3] || null, null, null],
        });
    }

    // Update chunk chart: recompute stacking per chunk in sorted order
    if (typeof chunkSortOrder !== "undefined") {
        var nChk = chunkSortOrder.length;
        var fStack = [new Array(nChk).fill(0), new Array(nChk).fill(0), new Array(nChk).fill(0), new Array(nChk).fill(0)];
        var ftot = new Array(nChk).fill(0);
        var uStack = [new Array(nChk).fill(0), new Array(nChk).fill(0), new Array(nChk).fill(0), new Array(nChk).fill(0)];
        for (var i = 0; i < nContacts; i++) {
            var rank = chunkRankMap[contacts[i].k];
            var csc;
            if (useAnn) {
                var ann = annotations[contactKey(contacts[i])];
                csc = (ann && ann.label === "correct") ? 0 : (ann && ann.label === "wrong") ? 1 : (ann && ann.label) ? 2 : 3;
            } else {
                csc = contacts[i].g;
            }
            uStack[csc][rank]++;
            if (currentMask[i]) {
                fStack[csc][rank]++;
                ftot[rank]++;
            }
        }
        var fb1 = fStack[0].slice();
        var fb2 = fStack[0].map(function(v,j){ return v + fStack[1][j]; });
        var fb3 = fStack[0].map(function(v,j){ return v + fStack[1][j] + fStack[2][j]; });
        var ub1 = uStack[0].slice();
        var ub2 = uStack[0].map(function(v,j){ return v + uStack[1][j]; });
        var ub3 = uStack[0].map(function(v,j){ return v + uStack[1][j] + uStack[2][j]; });
        var useF = (gtMode === "filtered");
        var cg0 = useF ? fStack[0] : (useAnn ? uStack[0] : chunkUnfiltGt[0]);
        var cg1 = useF ? fStack[1] : (useAnn ? uStack[1] : chunkUnfiltGt[1]);
        var cg2 = useF ? fStack[2] : (useAnn ? uStack[2] : chunkUnfiltGt[2]);
        var cg3 = useAnn ? (useF ? fStack[3] : uStack[3]) : new Array(nChk).fill(0);
        var cPct = useAnn ? gtPctTexts(cg0, cg1, cg2, cg3) : gtPctTexts(cg0, cg1, cg2);
        Plotly.restyle("chart_chunks", {
            y: [cg0, cg1, cg2, cg3, chunkUnfiltTotals, ftot],
            base: [null,
                   useF ? fb1 : (useAnn ? ub1 : chunkUnfiltBases1),
                   useF ? fb2 : (useAnn ? ub2 : chunkUnfiltBases2),
                   useF ? fb3 : (useAnn ? ub3 : chunkUnfiltBases2),
                   null, null],
            text: [cPct[0], cPct[1], cPct[2], cPct[3] || null, null, null],
        });
    }

    // Update bar colors based on mode
    var traceColors = useAnn ? ANN_TRACE_COLORS : GT_COLORS;
    var c3 = useAnn ? ANN_TRACE_COLORS[3] : "rgba(0,0,0,0)";
    allChartIds.forEach(function(id) {
        if (!document.getElementById(id)) return;
        Plotly.restyle(id, {'marker.color': [traceColors[0], traceColors[1], traceColors[2], c3]}, [0, 1, 2, 3]);
    });

    // Update chunk bar opacity for excluded chunks
    if (typeof chunkSortOrder !== "undefined") {
        var nChk = chunkSortOrder.length;
        var chunkOpacity = new Array(nChk);
        var dimExcluded = (gtMode === "filtered");
        for (var r = 0; r < nChk; r++) {
            var chId = chunkSortOrder[r];
            chunkOpacity[r] = (dimExcluded && (excludedChunkRanks[r] || (nucleusFilter && chunkHasNucleus[chId]))) ? 0.15 : 1;
        }
        Plotly.restyle("chart_chunks", {'marker.opacity': [chunkOpacity, chunkOpacity, chunkOpacity, chunkOpacity]}, [0, 1, 2, 3]);
    }

    updateYScales();
}

// --- Chunk exclusion helpers ---
function parseChunkExcludeInput(str) {
    var result = {};
    str.split(",").forEach(function(part) {
        part = part.trim();
        if (!part) return;
        var m = part.match(/^(\d+)\s*-\s*(\d+)$$/);
        if (m) { for (var r = parseInt(m[1]); r <= parseInt(m[2]); r++) result[r] = true; }
        else if (/^\d+$$/.test(part)) { result[parseInt(part)] = true; }
    });
    return result;
}

function formatChunkExcludes() {
    var ranks = Object.keys(excludedChunkRanks).map(Number).sort(function(a,b){return a-b;});
    if (ranks.length === 0) return "";
    var parts = [], start = ranks[0], end = ranks[0];
    for (var i = 1; i < ranks.length; i++) {
        if (ranks[i] === end + 1) end = ranks[i];
        else { parts.push(start === end ? ""+start : start+"-"+end); start = end = ranks[i]; }
    }
    parts.push(start === end ? ""+start : start+"-"+end);
    return parts.join(",");
}

// --- Deterministic contact ordering (seeded PRNG, computed once at startup) ---
function mulberry32(seed) {
    return function() {
        seed |= 0; seed = seed + 0x6D2B79F5 | 0;
        var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

// contactOrder[i] = deterministic rank for contact i (lower = earlier in samples)
var contactOrder = null;

function initContactOrder() {
    var rng = mulberry32(42);
    contactOrder = new Float64Array(nContacts);
    for (var i = 0; i < nContacts; i++) contactOrder[i] = rng();
}

// Collect contacts matching testFn, split by GT, in deterministic order.
// Returns groups with .total (full count passing testFn+filter) and
// .samples (first N_SAMPLES indices in deterministic order).
function stratifiedSample(testFn) {
    var useFiltered = (gtMode === "filtered");
    var byGt = [[], [], []];
    for (var i = 0; i < nContacts; i++) {
        if (testFn(i) && (!useFiltered || currentMask[i]))
            byGt[contacts[i].g].push(i);
    }
    var result = [];
    for (var g = 0; g < 3; g++) {
        byGt[g].sort(function(a, b) { return contactOrder[a] - contactOrder[b]; });
        result.push({
            gt: g,
            total: byGt[g].length,
            samples: byGt[g].slice(0, N_SAMPLES),
        });
    }
    return result;
}

function stratifiedSampleFromBin(mi, binIdx) {
    var bins = metrics[mi].bin_indices;
    return stratifiedSample(function(i) { return bins[i] === binIdx; });
}

function renderSamplePanel(panelEl, headerText, groups) {
    var totalInBin = groups[0].total + groups[1].total + groups[2].total;
    var closeBtn = '<span class="panel-close" onclick="this.closest(\\'.link-panel\\').style.display=\\'none\\'">\u2715</span>';
    if (totalInBin === 0) {
        panelEl.innerHTML = '<div class="panel-header"><span>' + headerText +
            ': no contacts</span>' + closeBtn + '</div>';
        panelEl.style.display = "block";
        return;
    }
    if (!panelEl._pageSize) panelEl._pageSize = 5;
    panelEl._page = 0;
    panelEl._groups = groups;
    panelEl._headerText = headerText;
    panelEl._totalInBin = totalInBin;
    renderSamplePanelPage(panelEl);
    panelEl.style.display = "block";
}

function renderContactRow(c, key, ann) {
    var url = buildNgUrl(c);
    var html = '<tr data-key="' + key.replace(/"/g, '&quot;') + '">';
    html += '<td><a href="' + url + '" target="_blank" class="ngl-link" data-label="' + c.a + '/' + c.b + ' [' + GT_NAMES[c.g] + ']">ngl</a></td>';
    html += '<td>' + c.a + '</td><td>' + c.b + '</td>';
    for (var vi = 0; vi < tableHeaders.length; vi++) {
        var val = c.v[vi];
        html += '<td>' + (val === null ? '-' :
            (typeof val === 'number' && val % 1 !== 0 ? val.toFixed(3) : val)) + '</td>';
    }
    html += '<td><span class="ann-btn' + (ann.label === "correct" ? ' active-correct' : '') + '" data-label="correct">&#x2713;</span></td>';
    html += '<td><span class="ann-btn' + (ann.label === "wrong" ? ' active-wrong' : '') + '" data-label="wrong">&#x2717;</span></td>';
    html += '<td><span class="ann-btn' + (ann.label && ann.label !== "correct" && ann.label !== "wrong" ? ' active-unclear' : '') + '" data-label="unclear">?</span></td>';
    html += '<td><input class="ann-note" value="' + (ann.note || '').replace(/"/g, '&quot;') + '" placeholder="note"></td>';
    html += '</tr>';
    return html;
}

function renderSamplePanelPage(panelEl) {
    var groups = panelEl._groups;
    var headerText = panelEl._headerText;
    var totalInBin = panelEl._totalInBin;
    var totalSampled = groups[0].samples.length + groups[1].samples.length + groups[2].samples.length;
    var page = panelEl._page;
    var pageSize = panelEl._pageSize;
    // Pages apply per-GT: each page shows pageSize items from each category
    var maxSamples = Math.max(groups[0].samples.length, groups[1].samples.length, groups[2].samples.length);
    var totalPages = Math.max(1, Math.ceil(maxSamples / pageSize));
    if (page >= totalPages) page = totalPages - 1;
    if (page < 0) page = 0;
    panelEl._page = page;
    var start = page * pageSize;

    var closeBtn = '<span class="panel-close" onclick="this.closest(\\'.link-panel\\').style.display=\\'none\\'">\u2715</span>';
    var html = '<div class="panel-header"><span>' + headerText +
        ' (' + totalInBin + ' contacts, ' + totalSampled + ' sampled)</span>' + closeBtn + '</div>';

    // Page navigation
    html += '<div class="page-nav">';
    html += '<button class="page-prev"' + (page === 0 ? ' disabled' : '') + '>&laquo; Prev</button>';
    html += '<span>Page ' + (page + 1) + ' of ' + totalPages + '</span>';
    html += '<button class="page-next"' + (page >= totalPages - 1 ? ' disabled' : '') + '>Next &raquo;</button>';
    html += ' <select class="page-size">';
    [5, 10, 20].forEach(function(sz) {
        html += '<option value="' + sz + '"' + (sz === pageSize ? ' selected' : '') + '>' + sz + '/page</option>';
    });
    html += '</select></div>';

    // Build table header once
    var thead = '<tr><th>link</th><th>seg_a</th><th>seg_b</th>';
    for (var h = 0; h < tableHeaders.length; h++) thead += '<th>' + tableHeaders[h] + '</th>';
    thead += '<th>&#x2713;</th><th>&#x2717;</th><th>?</th><th>note</th></tr>';

    // Render per-GT tables, each showing pageSize items from its own sample list
    var catCss = ["cat-merge", "cat-nomerge", "cat-unknown"];
    for (var g = 0; g < 3; g++) {
        if (groups[g].total === 0) continue;
        var pct = (100 * groups[g].total / totalInBin).toFixed(1);
        html += '<div class="cat-header ' + catCss[g] + '">' +
            GT_NAMES[g] + ': ' + groups[g].total + ' (' + pct + '%)</div>';
        var gStart = Math.min(start, groups[g].samples.length);
        var gEnd = Math.min(start + pageSize, groups[g].samples.length);
        if (gStart >= gEnd) continue;
        html += '<table>' + thead;
        for (var s = gStart; s < gEnd; s++) {
            var c = contacts[groups[g].samples[s]];
            var key = contactKey(c);
            var ann = annotations[key] || {};
            html += renderContactRow(c, key, ann);
        }
        html += '</table>';
    }

    // Annotation summary
    html += renderAnnotationSummary(groups);

    panelEl.innerHTML = html;

    // Wire up pagination
    var prevBtn = panelEl.querySelector('.page-prev');
    var nextBtn = panelEl.querySelector('.page-next');
    var sizeSel = panelEl.querySelector('.page-size');
    if (prevBtn) prevBtn.addEventListener('click', function() {
        panelEl._page--;
        renderSamplePanelPage(panelEl);
    });
    if (nextBtn) nextBtn.addEventListener('click', function() {
        panelEl._page++;
        renderSamplePanelPage(panelEl);
    });
    if (sizeSel) sizeSel.addEventListener('change', function() {
        panelEl._pageSize = parseInt(this.value);
        panelEl._page = 0;
        renderSamplePanelPage(panelEl);
    });

    // Wire up annotation buttons
    panelEl.querySelectorAll('.ann-btn').forEach(function(btn) {
        btn.addEventListener('click', function() {
            var row = btn.closest('tr');
            var key = row.dataset.key;
            var label = btn.dataset.label;
            if (!annotations[key]) annotations[key] = {};
            if (annotations[key].label === label) {
                annotations[key].label = null;
            } else {
                annotations[key].label = label;
            }
            saveAnnotations();
            row.querySelectorAll('.ann-btn').forEach(function(b) {
                b.classList.remove('active-correct', 'active-wrong', 'active-unclear');
                if (annotations[key].label && b.dataset.label === annotations[key].label) {
                    b.classList.add('active-' + annotations[key].label);
                }
            });
            var summaryEl = panelEl.querySelector('.ann-summary');
            if (summaryEl) summaryEl.outerHTML = renderAnnotationSummary(panelEl._groups);
        });
    });

    // Wire up note inputs
    panelEl.querySelectorAll('.ann-note').forEach(function(input) {
        var save = function() {
            var row = input.closest('tr');
            var key = row.dataset.key;
            if (!annotations[key]) annotations[key] = {};
            annotations[key].note = input.value;
            saveAnnotations();
        };
        input.addEventListener('blur', save);
        input.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') { save(); input.blur(); }
        });
    });
}

function renderAnnotationSummary(groups) {
    var totalAll = groups[0].total + groups[1].total + groups[2].total;
    var html = '<div class="ann-summary">';
    var wCorrect = 0, wWrong = 0, wUnclear = 0, wTotal = 0;
    var hasAny = false;
    for (var g = 0; g < 3; g++) {
        var grp = groups[g];
        if (grp.total === 0) continue;
        var nCorrect = 0, nWrong = 0, nUnclear = 0, nAnnotated = 0;
        for (var s = 0; s < grp.samples.length; s++) {
            var c = contacts[grp.samples[s]];
            var ann = annotations[contactKey(c)];
            if (ann && ann.label) {
                nAnnotated++;
                if (ann.label === "correct") nCorrect++;
                else if (ann.label === "wrong") nWrong++;
                else nUnclear++;
            }
        }
        var pcts = '';
        if (nAnnotated > 0) {
            pcts = ' &#x2192; ' +
                (100 * nCorrect / nAnnotated).toFixed(1) + '% correct, ' +
                (100 * nWrong / nAnnotated).toFixed(1) + '% wrong, ' +
                (100 * nUnclear / nAnnotated).toFixed(1) + '% unclear';
        }
        html += '<div>' + GT_NAMES[g] + ': ' +
            '<span style="color:#2e7d32">' + nCorrect + ' &#x2713;</span>, ' +
            '<span style="color:#c62828">' + nWrong + ' &#x2717;</span>, ' +
            '<span style="color:#f9a825">' + nUnclear + ' ?</span>' +
            ' (of ' + nAnnotated + ' annotated / ' + grp.samples.length + ' sampled / ' + grp.total + ' total)' +
            pcts + '</div>';
        if (nAnnotated > 0) {
            hasAny = true;
            var weight = grp.total / totalAll;
            wCorrect += (nCorrect / nAnnotated) * weight;
            wWrong += (nWrong / nAnnotated) * weight;
            wUnclear += (nUnclear / nAnnotated) * weight;
            wTotal += weight;
        }
    }
    if (hasAny && wTotal > 0) {
        html += '<div style="font-weight:bold;margin-top:2px">Weighted overall: ' +
            (100 * wCorrect / wTotal).toFixed(1) + '% correct, ' +
            (100 * wWrong / wTotal).toFixed(1) + '% wrong, ' +
            (100 * wUnclear / wTotal).toFixed(1) + '% unclear</div>';
    }
    html += '</div>';
    return html;
}

// --- All-annotations table at bottom of page ---
var allAnnEl = null;

function refreshAllAnnotations() {
    if (!allAnnEl) allAnnEl = document.getElementById("all-annotations");
    if (!allAnnEl) return;
    // Collect all annotated contact indices, grouped by GT
    var byGt = [[], [], []];
    for (var i = 0; i < nContacts; i++) {
        var ann = annotations[contactKey(contacts[i])];
        if (ann && ann.label) byGt[contacts[i].g].push(i);
    }
    // Sort each group by deterministic order
    for (var g = 0; g < 3; g++) {
        byGt[g].sort(function(a, b) { return contactOrder[a] - contactOrder[b]; });
    }
    var totalAnn = byGt[0].length + byGt[1].length + byGt[2].length;
    if (totalAnn === 0) {
        allAnnEl.style.display = "none";
        return;
    }
    allAnnEl.style.display = "block";
    if (!allAnnEl._pageSize) allAnnEl._pageSize = 20;
    if (allAnnEl._page === undefined) allAnnEl._page = 0;
    allAnnEl._byGt = byGt;
    allAnnEl._totalAnn = totalAnn;
    renderAllAnnotationsPage();
}

function renderAllAnnotationsPage() {
    var el = allAnnEl;
    var byGt = el._byGt;
    var totalAnn = el._totalAnn;
    var page = el._page;
    var pageSize = el._pageSize;
    var maxPerGt = Math.max(byGt[0].length, byGt[1].length, byGt[2].length);
    var totalPages = Math.max(1, Math.ceil(maxPerGt / pageSize));
    if (page >= totalPages) page = totalPages - 1;
    if (page < 0) page = 0;
    el._page = page;
    var start = page * pageSize;

    var html = '<div class="panel-header" style="font-size:20px;margin-bottom:4px"><span>All Annotations (' + totalAnn + ')</span></div>';

    // Page navigation
    html += '<div class="page-nav">';
    html += '<button class="aa-prev"' + (page === 0 ? ' disabled' : '') + '>&laquo; Prev</button>';
    html += '<span>Page ' + (page + 1) + ' of ' + totalPages + '</span>';
    html += '<button class="aa-next"' + (page >= totalPages - 1 ? ' disabled' : '') + '>Next &raquo;</button>';
    html += ' <select class="aa-size">';
    [10, 20, 50].forEach(function(sz) {
        html += '<option value="' + sz + '"' + (sz === pageSize ? ' selected' : '') + '>' + sz + '/page</option>';
    });
    html += '</select></div>';

    var thead = '<tr><th>link</th><th>seg_a</th><th>seg_b</th>';
    for (var h = 0; h < tableHeaders.length; h++) thead += '<th>' + tableHeaders[h] + '</th>';
    thead += '<th>&#x2713;</th><th>&#x2717;</th><th>?</th><th>note</th></tr>';

    var catCss = ["cat-merge", "cat-nomerge", "cat-unknown"];
    for (var g = 0; g < 3; g++) {
        if (byGt[g].length === 0) continue;
        var nCorrect = 0, nWrong = 0, nUnclear = 0;
        for (var s = 0; s < byGt[g].length; s++) {
            var ann = annotations[contactKey(contacts[byGt[g][s]])];
            if (ann) {
                if (ann.label === "correct") nCorrect++;
                else if (ann.label === "wrong") nWrong++;
                else nUnclear++;
            }
        }
        html += '<div class="cat-header ' + catCss[g] + '">' +
            GT_NAMES[g] + ': ' + byGt[g].length + ' annotated &mdash; ' +
            '<span style="color:#2e7d32">' + nCorrect + ' &#x2713;</span> ' +
            '<span style="color:#c62828">' + nWrong + ' &#x2717;</span> ' +
            '<span style="color:#f9a825">' + nUnclear + ' ?</span></div>';
        var gStart = Math.min(start, byGt[g].length);
        var gEnd = Math.min(start + pageSize, byGt[g].length);
        if (gStart >= gEnd) continue;
        html += '<table>' + thead;
        for (var s = gStart; s < gEnd; s++) {
            var c = contacts[byGt[g][s]];
            var key = contactKey(c);
            var ann = annotations[key] || {};
            html += renderContactRow(c, key, ann);
        }
        html += '</table>';
    }

    el.innerHTML = html;

    // Wire up pagination
    var prevBtn = el.querySelector('.aa-prev');
    var nextBtn = el.querySelector('.aa-next');
    var sizeSel = el.querySelector('.aa-size');
    if (prevBtn) prevBtn.addEventListener('click', function() {
        el._page--;
        renderAllAnnotationsPage();
    });
    if (nextBtn) nextBtn.addEventListener('click', function() {
        el._page++;
        renderAllAnnotationsPage();
    });
    if (sizeSel) sizeSel.addEventListener('change', function() {
        el._pageSize = parseInt(this.value);
        el._page = 0;
        renderAllAnnotationsPage();
    });

    // Wire up annotation buttons (toggle + save + refresh)
    el.querySelectorAll('.ann-btn').forEach(function(btn) {
        btn.addEventListener('click', function() {
            var row = btn.closest('tr');
            var key = row.dataset.key;
            var label = btn.dataset.label;
            if (!annotations[key]) annotations[key] = {};
            if (annotations[key].label === label) {
                annotations[key].label = null;
            } else {
                annotations[key].label = label;
            }
            saveAnnotations();
            if (!annotations[key].label && !annotations[key].note) {
                delete annotations[key];
                saveAnnotations();
                refreshAllAnnotations();
            } else {
                row.querySelectorAll('.ann-btn').forEach(function(b) {
                    b.classList.remove('active-correct', 'active-wrong', 'active-unclear');
                    if (annotations[key].label && b.dataset.label === annotations[key].label) {
                        b.classList.add('active-' + annotations[key].label);
                    }
                });
            }
        });
    });

    // Wire up note inputs
    el.querySelectorAll('.ann-note').forEach(function(input) {
        var save = function() {
            var row = input.closest('tr');
            var key = row.dataset.key;
            if (!annotations[key]) annotations[key] = {};
            annotations[key].note = input.value;
            saveAnnotations();
        };
        input.addEventListener('blur', save);
        input.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') { save(); input.blur(); }
        });
    });
}

function binRangeStr(m, binIdx) {
    var lo = m.edges[binIdx], hi = m.edges[binIdx + 1];
    if (m.log_x)
        return Math.round(fromLog1p(lo)) + " \u2013 " + Math.round(fromLog1p(hi));
    return lo.toFixed(3) + " \u2013 " + hi.toFixed(3);
}

// --- Neuroglancer popup viewer (reuses same window) ---
var ngPopup = null;
var ngPopupCheckTimer = null;
// Remembered position/size (null = use current window's geometry)
var ngPopupGeom = null;

function clearNglHighlight() {
    var prev = document.querySelector("tr.ngl-active");
    if (prev) prev.classList.remove("ngl-active");
}

function savePopupGeom() {
    try {
        if (ngPopup && !ngPopup.closed) {
            ngPopupGeom = {
                w: ngPopup.outerWidth, h: ngPopup.outerHeight,
                x: ngPopup.screenX, y: ngPopup.screenY
            };
        }
    } catch(e) {}
}

function startPopupCloseCheck() {
    if (ngPopupCheckTimer) return;
    ngPopupCheckTimer = setInterval(function() {
        if (!ngPopup || ngPopup.closed) {
            clearNglHighlight();
            clearInterval(ngPopupCheckTimer);
            ngPopupCheckTimer = null;
            ngPopup = null;
        } else {
            savePopupGeom();
        }
    }, 500);
}

document.addEventListener("click", function(e) {
    var link = e.target.closest(".ngl-link");
    if (link) {
        e.preventDefault();
        clearNglHighlight();
        var row = link.closest("tr");
        if (row) row.classList.add("ngl-active");
        // Open/reuse popup
        if (!ngPopup || ngPopup.closed) {
            var g = ngPopupGeom || {
                w: window.outerWidth, h: window.outerHeight,
                x: window.screenX, y: window.screenY
            };
            ngPopup = window.open(link.href, "ng_viewer",
                "width=" + g.w + ",height=" + g.h +
                ",left=" + g.x + ",top=" + g.y +
                ",menubar=no,toolbar=no,location=yes,status=no,resizable=yes,scrollbars=yes");
        } else {
            savePopupGeom();
            ngPopup.location.href = link.href;
            ngPopup.focus();
        }
        startPopupCloseCheck();
    }
});

// --- Compressed data blob ---
var DATA_B64 = "$data_b64";

// --- Build grid (async: waits for data decompression) ---
document.addEventListener("DOMContentLoaded", async function() {
    var summaryEl = document.getElementById("summary");
    var progressWrap = document.getElementById("progress-wrap");
    var progressText = document.getElementById("progress-text");

    progressText.textContent = "Decompressing data...";
    summaryEl.textContent = "Decompressing data...";

    var DATA = await decompressData(DATA_B64);
    DATA_B64 = null; // free the base64 string

    contacts = DATA.contacts;
    metrics = DATA.metrics;
    ngInfo = DATA.ngInfo;
    nContacts = contacts.length;
    tableHeaders = DATA.tableHeaders;
    var nChunksData = DATA.nChunks;
    DATA = null; // free the wrapper object

    hasMeshData = nContacts > 0 && contacts[0].m !== undefined;
    hasNucleusData = nContacts > 0 && contacts[0].n !== undefined;
    if (hasNucleusData) {
        for (var i = 0; i < nContacts; i++) {
            if (contacts[i].n) chunkHasNucleus[contacts[i].k] = true;
        }
    }
    initContactOrder();

    // Initialize thresholds
    metrics.forEach(function(m) {
        if (!m || m === "mesh") return;
        if (m.filter_dir === "minmax") {
            thresholds[m.col + "_lo"] = m.thr_min;
            thresholds[m.col + "_hi"] = m.thr_max;
        } else if (m.filter_dir === "min") {
            thresholds[m.col] = m.thr_min;
        } else {
            thresholds[m.col] = m.thr_max;
        }
    });

    var grid = document.getElementById("grid");
    currentMask = computeFilterMask();

    progressText.textContent = "Building charts...";
    summaryEl.textContent = "Building charts...";

    document.querySelectorAll('input[name="gt_mode"]').forEach(function(r) {
        r.addEventListener("change", function() {
            gtMode = this.value;
            updateAll();
            document.querySelectorAll('.link-panel').forEach(function(p) {
                if (p.style.display !== "none" && p._refresh) p._refresh();
            });
        });
    });

    document.querySelectorAll('input[name="y_scale"]').forEach(function(r) {
        r.addEventListener("change", function() {
            yScale = this.value;
            updateYScales();
        });
    });

    document.querySelectorAll('input[name="bar_color"]').forEach(function(r) {
        r.addEventListener("change", function() {
            barColorMode = this.value;
            var l0 = document.getElementById("legend0");
            var l1 = document.getElementById("legend1");
            var l2 = document.getElementById("legend2");
            var l3 = document.getElementById("legend3");
            if (barColorMode === "annotations") {
                l0.textContent = "correct"; l0.className = "lg-correct";
                l1.textContent = "wrong";   l1.className = "lg-wrong";
                l2.textContent = "unclear"; l2.className = "lg-unclear";
                l3.textContent = "unannotated"; l3.className = "lg-unannotated"; l3.style.display = "";
            } else {
                l0.textContent = "merge";    l0.className = "lg-merge";
                l1.textContent = "no_merge"; l1.className = "lg-nomerge";
                l2.textContent = "unknown";  l2.className = "lg-unknown";
                l3.style.display = "none";
            }
            updateAll();
        });
    });

    document.getElementById("ann_export").addEventListener("click", exportAnnotations);
    document.getElementById("ann_import").addEventListener("click", function() {
        document.getElementById("ann_file_input").click();
    });
    document.getElementById("ann_file_input").addEventListener("change", function() {
        if (this.files.length > 0) {
            importAnnotations(this.files[0]);
            this.value = "";
        }
    });
    document.getElementById("ann_clear").addEventListener("click", function() {
        var n = Object.keys(annotations).length;
        if (n === 0) return;
        if (!confirm("Clear all " + n + " annotations? This cannot be undone.")) return;
        annotations = {};
        saveAnnotations();
        document.querySelectorAll('.link-panel').forEach(function(p) {
            if (p.style.display !== "none" && p._refresh) p._refresh();
        });
    });

    loadAnnotations();

    metrics.forEach(function(m, mi) {
        if (!m) {
            var emptyCell = document.createElement("div");
            emptyCell.className = "cell";
            emptyCell.style.opacity = "0.3";
            grid.appendChild(emptyCell);
            return;
        }
        if (m === "mesh") {
            var meshCell = document.createElement("div");
            meshCell.className = "cell";
            if (!hasMeshData) {
                meshCell.style.opacity = "0.3";
                meshCell.innerHTML = '<div style="padding:20px;color:#888;font-size:20px">Mesh data not available (re-run stats collection)</div>';
                grid.appendChild(meshCell);
                return;
            }
            var meshDivId = "chart_mesh";
            var meshChartDiv = document.createElement("div");
            meshChartDiv.id = meshDivId;
            meshCell.appendChild(meshChartDiv);

            // Mesh filter toggle
            var meshCtrl = document.createElement("div");
            meshCtrl.className = "ctrl-row";
            meshCtrl.innerHTML =
                '<span class="dir-label">require_mesh</span>' +
                '<label style="font-size:18px;cursor:pointer"><input type="checkbox" id="mesh_filter_cb"> Only show contacts with both meshes available</label>';
            meshCell.appendChild(meshCtrl);

            var meshPanel = document.createElement("div");
            meshPanel.id = "panel_mesh";
            meshPanel.className = "link-panel";
            meshCell.appendChild(meshPanel);
            grid.appendChild(meshCell);

            // Count mesh stats per category and GT
            var meshCats = ["All","Mesh Available","Missing Mesh"];
            var gtAll = [0,0,0], gtWith = [0,0,0], gtWithout = [0,0,0];
            for (var i = 0; i < nContacts; i++) {
                var g = contacts[i].g;
                gtAll[g]++;
                if (contacts[i].m) gtWith[g]++; else gtWithout[g]++;
            }
            var meshGt = [gtAll, gtWith, gtWithout]; // [cat][gt]
            var meshUnfiltTots = meshGt.map(function(c){ return c[0]+c[1]+c[2]; });
            var meshFiltTots = meshUnfiltTots.slice(); // initially same
            var mb1 = [meshGt[0][0], meshGt[1][0], meshGt[2][0]];
            var mb2 = [meshGt[0][0]+meshGt[0][1], meshGt[1][0]+meshGt[1][1], meshGt[2][0]+meshGt[2][1]];
            var mPct = gtPctTexts(mb1, [meshGt[0][1],meshGt[1][1],meshGt[2][1]], [meshGt[0][2],meshGt[1][2],meshGt[2][2]]);

            Plotly.newPlot(meshDivId, [
                { x:meshCats, y:[meshGt[0][0],meshGt[1][0],meshGt[2][0]], type:"bar", name:"merge",
                   marker:{color:GT_COLORS[0], line:{width:0}},
                   text:mPct[0], textposition:"inside", textfont:{color:"white", size:13},
                   hoverinfo:"skip", showlegend:false },
                { x:meshCats, y:[meshGt[0][1],meshGt[1][1],meshGt[2][1]], type:"bar", name:"no_merge",
                   base:mb1,
                   marker:{color:GT_COLORS[1], line:{width:0}},
                   text:mPct[1], textposition:"inside", textfont:{color:"white", size:13},
                   hoverinfo:"skip", showlegend:false },
                { x:meshCats, y:[meshGt[0][2],meshGt[1][2],meshGt[2][2]], type:"bar", name:"unknown",
                   base:mb2,
                   marker:{color:GT_COLORS[2], line:{width:0}},
                   textposition:"inside", textfont:{color:"white", size:13},
                   hoverinfo:"skip", showlegend:false },
                { x:meshCats, y:[0,0,0], type:"bar", name:"unannotated",
                   base:mb2,
                   marker:{color:"rgba(0,0,0,0)", line:{width:0}},
                   textposition:"inside", textfont:{color:"white", size:13},
                   hoverinfo:"skip", showlegend:false },
                { x:meshCats, y:meshUnfiltTots, type:"bar", name:"All",
                   marker:{color:"rgba(0,0,0,0)", line:{color:"rgba(160,160,160,0.8)", width:1.5}},
                   hoverinfo:"y", showlegend:false },
                { x:meshCats, y:meshFiltTots, type:"bar", name:"Filtered",
                   marker:{color:"rgba(30,80,220,0.08)", line:{color:"rgba(0,0,0,0.9)", width:2.5}},
                   hoverinfo:"y", showlegend:false },
            ], {
                title: {text:"Mesh Availability", font:{size:22}},
                yaxis: {title:{text:"Count", font:{size:17}}, type:"log", dtick:1, autorange:"max", range:[0], tickfont:{size:15}},
                xaxis: {tickfont:{size:17}},
                barmode: "overlay",
                margin: {t:32, b:36, l:50, r:8},
                height: 280,
                showlegend: false,
            }, {responsive:true});
            allChartIds.push(meshDivId);

            // Click handler for mesh chart
            meshChartDiv.on("plotly_click", function(data) {
                var pt = data.points[0];
                var label = pt.x;
                var testFn;
                if (label === "Mesh Available") {
                    testFn = function(i) { return contacts[i].m === 1; };
                } else if (label === "Missing Mesh") {
                    testFn = function(i) { return !contacts[i].m; };
                } else {
                    testFn = function(i) { return true; };
                }
                meshPanel._refresh = function() {
                    var groups = stratifiedSample(testFn);
                    renderSamplePanel(meshPanel, label, groups);
                };
                meshPanel._refresh();
            });

            // Checkbox handler
            document.getElementById("mesh_filter_cb").addEventListener("change", function() {
                meshFilter = this.checked;
                updateAll();
                document.querySelectorAll('.link-panel').forEach(function(p) {
                    if (p.style.display !== "none" && p._refresh) p._refresh();
                });
            });
            return;
        }
        var divId = "chart_" + mi;
        var panelId = "panel_" + mi;

        var cell = document.createElement("div");
        cell.className = "cell";

        var chartDiv = document.createElement("div");
        chartDiv.id = divId;
        cell.appendChild(chartDiv);

        // --- Controls ---
        function makeSliderRow(label, thrKey, initVal) {
            var row = document.createElement("div");
            row.className = "ctrl-row";
            var sid = "slider_" + mi + "_" + thrKey;
            var nid = "input_" + mi + "_" + thrKey;
            row.innerHTML =
                '<span class="dir-label">' + label + '</span>' +
                '<input type="range" id="' + sid + '" step="any">' +
                '<input type="number" id="' + nid + '" step="any">';
            cell.appendChild(row);

            var slider = row.querySelector('input[type="range"]');
            var numInput = row.querySelector('input[type="number"]');
            if (m.log_x) {
                slider.min = toLog1p(m.thr_min);
                slider.max = toLog1p(m.thr_max);
                slider.step = (parseFloat(slider.max) - parseFloat(slider.min)) / 200;
                slider.value = toLog1p(initVal);
            } else {
                slider.min = m.thr_min;
                slider.max = m.thr_max;
                slider.step = (m.thr_max - m.thr_min) / 200;
                slider.value = initVal;
            }
            numInput.value = m.log_x ? Math.round(initVal) : parseFloat(initVal).toPrecision(4);

            slider.addEventListener("input", function() {
                var val = m.log_x ? fromLog1p(parseFloat(slider.value)) : parseFloat(slider.value);
                thresholds[thrKey] = val;
                numInput.value = m.log_x ? Math.round(val) : val.toPrecision(4);
                updateAll();
            });
            numInput.addEventListener("change", function() {
                var val = parseFloat(numInput.value);
                if (isNaN(val)) return;
                thresholds[thrKey] = val;
                slider.value = m.log_x ? toLog1p(val) : val;
                updateAll();
            });
        }

        if (m.filter_dir === "minmax") {
            makeSliderRow(m.param_names[0], m.col + "_lo", m.thr_min);
            makeSliderRow(m.param_names[1], m.col + "_hi", m.thr_max);
        } else {
            var initVal = m.filter_dir === "min" ? m.thr_min : m.thr_max;
            makeSliderRow(m.param_names[0], m.col, initVal);
        }

        var linkDiv = document.createElement("div");
        linkDiv.id = panelId;
        linkDiv.className = "link-panel";
        cell.appendChild(linkDiv);

        grid.appendChild(cell);

        // --- Initial GT counts ---
        var useFiltered = (gtMode === "filtered");
        var gt = computeGtBinCounts(mi, useFiltered);
        var bases1 = gt[0].slice();
        var bases2 = gt[0].map(function(v,i) { return v + gt[1][i]; });

        // --- Shapes: threshold line(s) + highlight rect ---
        var shapes = [];
        if (m.filter_dir === "minmax") {
            var thrLo = m.log_x ? toLog1p(m.thr_min) : m.thr_min;
            var thrHi = m.log_x ? toLog1p(m.thr_max) : m.thr_max;
            shapes.push({ type:"line", x0:thrLo, x1:thrLo, y0:0, y1:1, yref:"paper",
                          line:{color:"blue", width:2, dash:"dash"} });
            shapes.push({ type:"line", x0:thrHi, x1:thrHi, y0:0, y1:1, yref:"paper",
                          line:{color:"blue", width:2, dash:"dash"} });
        } else {
            var thrVal = m.log_x
                ? toLog1p(m.filter_dir==="min" ? m.thr_min : m.thr_max)
                : (m.filter_dir==="min" ? m.thr_min : m.thr_max);
            shapes.push({ type:"line", x0:thrVal, x1:thrVal, y0:0, y1:1, yref:"paper",
                          line:{color:"red", width:2, dash:"dash"} });
        }
        // Highlight rect (always last shape)
        shapes.push({ type:"rect", x0:0, x1:0, y0:0, y1:1, yref:"paper",
                       fillcolor:"rgba(255,200,0,0.15)",
                       line:{color:"rgba(255,160,0,0.6)", width:2},
                       visible:false });

        var traces = [
            { x:m.x_centers, y:gt[0], type:"bar", name:"merge",
               marker:{color:GT_COLORS[0], line:{width:0}},
               textposition:"inside", textfont:{color:"white", size:11},
               hoverinfo:"skip", showlegend:false },
            { x:m.x_centers, y:gt[1], type:"bar", name:"no_merge",
               base:bases1,
               marker:{color:GT_COLORS[1], line:{width:0}},
               textposition:"inside", textfont:{color:"white", size:11},
               hoverinfo:"skip", showlegend:false },
            { x:m.x_centers, y:gt[2], type:"bar", name:"unknown",
               base:bases2,
               marker:{color:GT_COLORS[2], line:{width:0}},
               textposition:"inside", textfont:{color:"white", size:11},
               hoverinfo:"skip", showlegend:false },
            { x:m.x_centers, y:new Array(m.x_centers.length).fill(0), type:"bar", name:"unannotated",
               base:bases2,
               marker:{color:"rgba(0,0,0,0)", line:{width:0}},
               textposition:"inside", textfont:{color:"white", size:11},
               hoverinfo:"skip", showlegend:false },
            { x:m.x_centers, y:m.unfiltered_counts, type:"bar", name:"All",
               marker:{color:"rgba(0,0,0,0)", line:{color:"rgba(160,160,160,0.8)", width:1.5}},
               hoverinfo:"y", showlegend:false },
            { x:m.x_centers, y:m.unfiltered_counts.slice(), type:"bar", name:"Filtered",
               marker:{color:"rgba(30,80,220,0.08)", line:{color:"rgba(0,0,0,0.9)", width:2.5}},
               hoverinfo:"y", showlegend:false,
               customdata:Array.from({length:m.x_centers.length}, function(_,i){return i;}) },
        ];

        var layout = {
            title: {text:m.title, font:{size:22}},
            xaxis: {title:{text:m.x_label, font:{size:17}}, tickfont:{size:15}},
            yaxis: {title:{text:"Count", font:{size:17}}, type:"log", dtick:1, autorange:"max", range:[0], tickfont:{size:15}},
            barmode: "overlay",
            bargap: 0.05,
            margin: {t:32, b:36, l:50, r:8},
            height: 280,
            showlegend: false,
            shapes: shapes,
        };
        if (m.tick_vals) {
            layout.xaxis.tickvals = m.tick_vals;
            layout.xaxis.ticktext = m.tick_texts;
        }

        Plotly.newPlot(divId, traces, layout, {responsive:true});
        allChartIds.push(divId);

        // Click handler
        chartDiv.on("plotly_click", function(data) {
            var panel = document.getElementById(panelId);
            var pt = data.points[0];
            var binIdx = pt.customdata;
            if (binIdx === undefined) {
                var xVal = pt.x;
                var closest = 0, closestDist = Infinity;
                for (var k = 0; k < m.x_centers.length; k++) {
                    var d = Math.abs(m.x_centers[k] - xVal);
                    if (d < closestDist) { closestDist = d; closest = k; }
                }
                binIdx = closest;
            }

            highlightedBin[mi] = binIdx;
            var hlIdx = m.filter_dir === "minmax" ? 2 : 1;
            var lu = {};
            lu["shapes["+hlIdx+"].x0"] = m.edges[binIdx];
            lu["shapes["+hlIdx+"].x1"] = m.edges[binIdx + 1];
            lu["shapes["+hlIdx+"].visible"] = true;
            Plotly.relayout(divId, lu);

            panel._refresh = function() {
                var groups = stratifiedSampleFromBin(mi, binIdx);
                renderSamplePanel(panel, binRangeStr(m, binIdx), groups);
            };
            panel._refresh();
        });
    });

    // --- Chunk distribution chart (per-chunk bars sorted by unfiltered count) ---
    var nChunks = nChunksData;

    // Compute unfiltered GT per chunk
    var ugt = [new Array(nChunks).fill(0), new Array(nChunks).fill(0), new Array(nChunks).fill(0)];
    for (var i = 0; i < nContacts; i++) ugt[contacts[i].g][contacts[i].k]++;
    var utot = new Array(nChunks).fill(0);
    for (var ch = 0; ch < nChunks; ch++) utot[ch] = ugt[0][ch] + ugt[1][ch] + ugt[2][ch];

    // Sort chunk indices by unfiltered total (descending)
    chunkSortOrder = Array.from({length: nChunks}, function(_,i){return i;});
    chunkSortOrder.sort(function(a,b){ return utot[b] - utot[a]; });
    // Reverse map: original chunk id -> rank
    chunkRankMap = new Array(nChunks);
    for (var r = 0; r < nChunks; r++) chunkRankMap[chunkSortOrder[r]] = r;

    // Reorder into sorted arrays
    var xLabels = [];
    chunkUnfiltGt = [[], [], []];
    chunkUnfiltTotals = [];
    for (var r = 0; r < nChunks; r++) {
        var ch = chunkSortOrder[r];
        xLabels.push(r);
        chunkUnfiltGt[0].push(ugt[0][ch]);
        chunkUnfiltGt[1].push(ugt[1][ch]);
        chunkUnfiltGt[2].push(ugt[2][ch]);
        chunkUnfiltTotals.push(utot[ch]);
    }
    chunkUnfiltBases1 = chunkUnfiltGt[0].slice();
    chunkUnfiltBases2 = chunkUnfiltGt[0].map(function(v,j){ return v + chunkUnfiltGt[1][j]; });

    // Initial filtered counts in sorted order
    var fgt_init = [new Array(nChunks).fill(0), new Array(nChunks).fill(0), new Array(nChunks).fill(0)];
    var ftot_init = new Array(nChunks).fill(0);
    for (var i = 0; i < nContacts; i++) {
        if (currentMask[i]) {
            var rank = chunkRankMap[contacts[i].k];
            fgt_init[contacts[i].g][rank]++;
            ftot_init[rank]++;
        }
    }

    var chunkCell = document.createElement("div");
    chunkCell.className = "cell";
    var chunkDiv = document.createElement("div");
    chunkDiv.id = "chart_chunks";
    chunkCell.appendChild(chunkDiv);

    // Chunk exclusion controls
    var chunkExclCtrl = document.createElement("div");
    chunkExclCtrl.className = "ctrl-row";
    chunkExclCtrl.innerHTML =
        '<span class="dir-label">exclude</span>' +
        '<input type="text" id="chunk_exclude_input" placeholder="e.g. 0,3,7-10" style="flex:1;padding:2px 6px;border:1px solid #ccc;border-radius:3px;font-size:16px;font-family:monospace">' +
        '<button id="chunk_exclude_apply" style="padding:2px 8px;border:1px solid #aab;border-radius:3px;background:#f0f4f8;cursor:pointer;font-size:16px">Apply</button>' +
        '<button id="chunk_exclude_clear" style="padding:2px 8px;border:1px solid #aab;border-radius:3px;background:#f0f4f8;cursor:pointer;font-size:16px">Clear</button>';
    chunkCell.appendChild(chunkExclCtrl);

    if (hasNucleusData) {
        var nucleusCtrl = document.createElement("div");
        nucleusCtrl.className = "ctrl-row";
        nucleusCtrl.innerHTML =
            '<span class="dir-label">nucleus</span>' +
            '<label style="font-size:18px;cursor:pointer"><input type="checkbox" id="nucleus_filter_cb"> Exclude chunks with nuclei</label>';
        chunkCell.appendChild(nucleusCtrl);
        nucleusCtrl.querySelector("input").addEventListener("change", function() {
            nucleusFilter = this.checked;
            updateAll();
            document.querySelectorAll('.link-panel').forEach(function(p) {
                if (p.style.display !== "none" && p._refresh) p._refresh();
            });
        });
    }

    var chunkPanel = document.createElement("div");
    chunkPanel.id = "panel_chunks";
    chunkPanel.className = "link-panel";
    chunkCell.appendChild(chunkPanel);
    grid.appendChild(chunkCell);

    var useF_init = (gtMode === "filtered");
    var cgt = useF_init ? fgt_init : chunkUnfiltGt;
    var cb1 = cgt[0].slice();
    var cb2 = cgt[0].map(function(v,j){ return v + cgt[1][j]; });
    var cPct = gtPctTexts(cgt[0], cgt[1], cgt[2]);

    Plotly.newPlot("chart_chunks", [
        { x:xLabels, y:cgt[0], type:"bar", name:"merge",
           marker:{color:GT_COLORS[0], line:{width:0}},
           text:cPct[0], textposition:"inside", textfont:{color:"white", size:11},
           hoverinfo:"skip", showlegend:false },
        { x:xLabels, y:cgt[1], type:"bar", name:"no_merge", base:cb1,
           marker:{color:GT_COLORS[1], line:{width:0}},
           text:cPct[1], textposition:"inside", textfont:{color:"white", size:11},
           hoverinfo:"skip", showlegend:false },
        { x:xLabels, y:cgt[2], type:"bar", name:"unknown", base:cb2,
           marker:{color:GT_COLORS[2], line:{width:0}},
           textposition:"inside", textfont:{color:"white", size:11},
           hoverinfo:"skip", showlegend:false },
        { x:xLabels, y:new Array(xLabels.length).fill(0), type:"bar", name:"unannotated", base:cb2,
           marker:{color:"rgba(0,0,0,0)", line:{width:0}},
           textposition:"inside", textfont:{color:"white", size:11},
           hoverinfo:"skip", showlegend:false },
        { x:xLabels, y:chunkUnfiltTotals, type:"bar", name:"All",
           marker:{color:"rgba(0,0,0,0)", line:{color:"rgba(160,160,160,0.8)", width:1.5}},
           hoverinfo:"y", showlegend:false },
        { x:xLabels, y:ftot_init, type:"bar", name:"Filtered",
           marker:{color:"rgba(0,0,0,0)", line:{color:"rgba(0,0,0,0.9)", width:1.5}},
           hoverinfo:"y", showlegend:false },
    ], {
        title: {text:"Contacts per Chunk (" + nChunks + " chunks)", font:{size:22}},
        xaxis: {title:{text:"chunk rank", font:{size:17}}, tickfont:{size:15}},
        yaxis: {title:{text:"Contacts", font:{size:17}}, type:"log", dtick:1, autorange:"max", range:[0], tickfont:{size:15}},
        barmode: "overlay",
        bargap: 0.05,
        margin: {t:32, b:36, l:50, r:8},
        height: 280,
        showlegend: false,
        shapes: [{ type:"rect", x0:0, x1:0, y0:0, y1:1, yref:"paper",
                   fillcolor:"rgba(255,200,0,0.15)",
                   line:{color:"rgba(255,160,0,0.6)", width:2},
                   visible:false }],
    }, {responsive:true});
    allChartIds.push("chart_chunks");

    chunkDiv.on("plotly_click", function(data) {
        var pt = data.points[0];
        var rank = typeof pt.x === "number" ? pt.x : parseInt(pt.x);
        if (rank < 0 || rank >= nChunks) return;
        var evt = data.event;
        if (evt && (evt.ctrlKey || evt.metaKey)) {
            if (excludedChunkRanks[rank]) delete excludedChunkRanks[rank];
            else excludedChunkRanks[rank] = true;
            document.getElementById("chunk_exclude_input").value = formatChunkExcludes();
            updateAll();
            document.querySelectorAll('.link-panel').forEach(function(p) {
                if (p.style.display !== "none" && p._refresh) p._refresh();
            });
            return;
        }
        Plotly.relayout("chart_chunks", {
            "shapes[0].x0": rank - 0.5, "shapes[0].x1": rank + 0.5,
            "shapes[0].visible": true
        });
        var chunkId = chunkSortOrder[rank];
        chunkPanel._refresh = function() {
            var groups = stratifiedSample(function(i) { return contacts[i].k === chunkId; });
            renderSamplePanel(chunkPanel, "Chunk rank " + rank, groups);
        };
        chunkPanel._refresh();
    });

    document.getElementById("chunk_exclude_apply").addEventListener("click", function() {
        excludedChunkRanks = parseChunkExcludeInput(document.getElementById("chunk_exclude_input").value);
        updateAll();
        document.querySelectorAll('.link-panel').forEach(function(p) {
            if (p.style.display !== "none" && p._refresh) p._refresh();
        });
    });
    document.getElementById("chunk_exclude_clear").addEventListener("click", function() {
        excludedChunkRanks = {};
        document.getElementById("chunk_exclude_input").value = "";
        updateAll();
        document.querySelectorAll('.link-panel').forEach(function(p) {
            if (p.style.display !== "none" && p._refresh) p._refresh();
        });
    });

    // --- Summary chart (total vs filtered, GT-stacked) ---
    var sumCell = document.createElement("div");
    sumCell.className = "cell";
    var sumDiv = document.createElement("div");
    sumDiv.id = "chart_summary";
    sumCell.appendChild(sumDiv);
    var sumPanel = document.createElement("div");
    sumPanel.id = "panel_summary";
    sumPanel.className = "link-panel";
    sumCell.appendChild(sumPanel);
    grid.appendChild(sumCell);

    function summaryGtCounts(useFiltered) {
        var counts = [0, 0, 0];
        for (var i = 0; i < nContacts; i++) {
            if (!useFiltered || currentMask[i])
                counts[contacts[i].g]++;
        }
        return counts;
    }

    var gtAll = summaryGtCounts(false);
    var gtFilt = summaryGtCounts(true);
    var sPct = gtPctTexts([gtAll[0],gtFilt[0]], [gtAll[1],gtFilt[1]], [gtAll[2],gtFilt[2]]);

    Plotly.newPlot("chart_summary", [
        { x:["Total","Filtered"], y:[gtAll[0], gtFilt[0]], type:"bar", name:"merge",
           marker:{color:GT_COLORS[0]},
           text:sPct[0], textposition:"inside", textfont:{color:"white", size:13},
           showlegend:false },
        { x:["Total","Filtered"], y:[gtAll[1], gtFilt[1]], type:"bar", name:"no_merge",
           marker:{color:GT_COLORS[1]},
           text:sPct[1], textposition:"inside", textfont:{color:"white", size:13},
           showlegend:false },
        { x:["Total","Filtered"], y:[gtAll[2], gtFilt[2]], type:"bar", name:"unknown",
           marker:{color:GT_COLORS[2]},
           textposition:"inside", textfont:{color:"white", size:13},
           showlegend:false },
        { x:["Total","Filtered"], y:[0, 0], type:"bar", name:"unannotated",
           marker:{color:"rgba(0,0,0,0)"},
           textposition:"inside", textfont:{color:"white", size:13},
           showlegend:false },
    ], {
        title: {text:"Filter Summary", font:{size:22}},
        barmode: "stack",
        yaxis: {title:{text:"Count", font:{size:17}}, type:"log", dtick:1, autorange:"max", range:[0], tickfont:{size:15}},
        xaxis: {tickfont:{size:17}},
        margin: {t:32, b:36, l:50, r:8},
        height: 280,
        showlegend: false,
    }, {responsive:true});
    allChartIds.push("chart_summary");

    sumDiv.on("plotly_click", function(data) {
        var pt = data.points[0];
        var isFiltered = (pt.x === "Filtered");
        sumPanel._refresh = function() {
            var groups = stratifiedSample(function(i) {
                return !isFiltered || currentMask[i];
            });
            renderSamplePanel(sumPanel, isFiltered ? "Filtered" : "Total", groups);
        };
        sumPanel._refresh();
    });

    // Show final count, hide progress bar
    progressWrap.style.display = "none";
    var nFiltered = 0;
    for (var i = 0; i < nContacts; i++) nFiltered += currentMask[i];
    summaryEl.textContent = nFiltered + " / " + nContacts + " contacts pass all filters";

    refreshAllAnnotations();
});
</script>
</body>
</html>""")


if __name__ == "__main__":
    main()
