#!/usr/bin/env python3
"""Visualize validation results for contact merge models.

Interactive dashboard: scatter plot (mean affinity vs shape score),
PR/ROC curves, confusion matrix, sample tables with neuroglancer links,
dataset filtering, and annotation support.

Usage:
    python scripts/visualize_validation_results.py \
        --paths gs://martin_exp/.../contacts_x1 gs://martin_exp/.../contacts_x2 \
        --authority model_v7.0_... \
        --gt-authority ground_truth \
        --output validation_results.html
"""

import argparse
import io
import json
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from string import Template

import fsspec
import numpy as np
from packaging.version import Version
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Binary readers (matching compute_contact_merge_metrics.py format)
# ---------------------------------------------------------------------------

def read_info(path):
    """Read info file from seg_contact layer using fsspec."""
    info_path = f"{path}/info"
    fs, fs_path = fsspec.core.url_to_fs(info_path)
    with fs.open(fs_path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def get_chunk_keys_from_info(info, bbox_start, bbox_end):
    """Generate chunk keys from info file grid, filtered by bbox overlap."""
    voxel_offset = info["voxel_offset"]
    size = info["size"]
    chunk_size = info["chunk_size"]
    keys = []
    for x in range(voxel_offset[0], voxel_offset[0] + size[0], chunk_size[0]):
        x_end = x + chunk_size[0]
        if x_end <= bbox_start[0] or x >= bbox_end[0]:
            continue
        for y in range(voxel_offset[1], voxel_offset[1] + size[1], chunk_size[1]):
            y_end = y + chunk_size[1]
            if y_end <= bbox_start[1] or y >= bbox_end[1]:
                continue
            for z in range(voxel_offset[2], voxel_offset[2] + size[2], chunk_size[2]):
                z_end = z + chunk_size[2]
                if z_end <= bbox_start[2] or z >= bbox_end[2]:
                    continue
                keys.append(f"{x}-{x_end}_{y}-{y_end}_{z}-{z_end}")
    return keys


def read_contacts_chunk(fs, path, format_version="1.0"):
    """Read contacts from a chunk, return dict keyed by contact_id."""
    try:
        with fs.open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        return {}

    contacts = {}
    with io.BytesIO(data) as f:
        n_contacts = struct.unpack("<I", f.read(4))[0]
        for _ in range(n_contacts):
            contact_id = struct.unpack("<q", f.read(8))[0]
            seg_a, seg_b = struct.unpack("<qq", f.read(16))
            com = struct.unpack("<fff", f.read(12))
            n_faces = struct.unpack("<I", f.read(4))[0]

            faces = []
            for _ in range(n_faces):
                face = struct.unpack("<ffff", f.read(16))
                faces.append(face)

            metadata_len = struct.unpack("<I", f.read(4))[0]
            if metadata_len > 0:
                f.read(metadata_len)

            if Version(format_version) >= Version("1.1"):
                f.read(24)

            if faces:
                affinities = np.array([face[3] for face in faces], dtype=np.float32)
                coords = np.array(
                    [[face[0], face[1], face[2]] for face in faces], dtype=np.float32
                )
                nonzero_mask = np.any(coords != 0, axis=1)
                nonzero_count = max(nonzero_mask.sum(), 1)
                mean_affinity = float((affinities * nonzero_mask).sum() / nonzero_count)
            else:
                mean_affinity = 0.0

            contacts[contact_id] = {
                "seg_a": seg_a,
                "seg_b": seg_b,
                "com": com,
                "mean_affinity": mean_affinity,
                "n_faces": n_faces,
            }
    return contacts


def read_merge_decisions_chunk(fs, path):
    """Read merge decisions from a chunk, return dict keyed by contact_id."""
    try:
        with fs.open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        return {}

    decisions = {}
    with io.BytesIO(data) as f:
        n_decisions = struct.unpack("<I", f.read(4))[0]
        for _ in range(n_decisions):
            contact_id = struct.unpack("<q", f.read(8))[0]
            should_merge = struct.unpack("<B", f.read(1))[0]
            decisions[contact_id] = bool(should_merge)
    return decisions


def read_merge_probabilities_chunk(fs, path):
    """Read merge probabilities from a chunk, return dict keyed by contact_id."""
    try:
        with fs.open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        return {}

    probs = {}
    with io.BytesIO(data) as f:
        n_entries = struct.unpack("<I", f.read(4))[0]
        for _ in range(n_entries):
            contact_id = struct.unpack("<q", f.read(8))[0]
            prob = struct.unpack("<f", f.read(4))[0]
            probs[contact_id] = prob
    return probs


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_dataset(path, authority, gt_authority, bbox_start, bbox_end, resolution):
    """Collect contacts with GT and predictions for one dataset."""
    fs, base_path = fsspec.core.url_to_fs(path)
    info = read_info(path)
    format_version = info.get("format_version", "1.0")
    chunk_keys = get_chunk_keys_from_info(info, bbox_start, bbox_end)

    bbox_start_nm = [bbox_start[i] * resolution[i] for i in range(3)]
    bbox_end_nm = [bbox_end[i] * resolution[i] for i in range(3)]

    contacts_out = []
    n_no_gt = 0
    n_no_pred = 0

    for chunk_key in tqdm(chunk_keys, desc=f"Reading {path.split('/')[-1]}", leave=False):
        contacts = read_contacts_chunk(
            fs,
            f"{base_path}/contacts/{chunk_key}",
            format_version=format_version,
        )
        gt_decisions = read_merge_decisions_chunk(
            fs, f"{base_path}/merge_decisions/{gt_authority}/{chunk_key}"
        )
        predictions = read_merge_probabilities_chunk(
            fs, f"{base_path}/merge_probabilities/{authority}/{chunk_key}"
        )

        if not contacts:
            continue

        for contact_id, contact in contacts.items():
            com = contact["com"]
            if not (
                bbox_start_nm[0] <= com[0] < bbox_end_nm[0]
                and bbox_start_nm[1] <= com[1] < bbox_end_nm[1]
                and bbox_start_nm[2] <= com[2] < bbox_end_nm[2]
            ):
                continue

            if contact_id not in gt_decisions:
                n_no_gt += 1
                continue
            if contact_id not in predictions:
                n_no_pred += 1
                continue

            shape_score = predictions[contact_id]
            mean_aff = contact["mean_affinity"]
            if not (np.isfinite(shape_score) and np.isfinite(mean_aff)):
                continue

            contacts_out.append({
                "seg_a": contact["seg_a"],
                "seg_b": contact["seg_b"],
                "com": list(contact["com"]),
                "mean_affinity": mean_aff,
                "n_faces": contact["n_faces"],
                "gt_merge": gt_decisions[contact_id],
                "shape_score": shape_score,
            })

    print(
        f"  {path.split('/')[-1]}: {len(contacts_out)} matched, "
        f"{n_no_gt} no GT, {n_no_pred} no pred"
    )
    return contacts_out, info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize validation results")
    parser.add_argument(
        "--paths", nargs="+", required=True, help="Seg_contact layer paths"
    )
    parser.add_argument("--authority", required=True, help="Merge probability authority")
    parser.add_argument(
        "--gt-authority", default="ground_truth", help="GT authority name"
    )
    parser.add_argument(
        "--output", default="validation_results.html", help="Output HTML file"
    )
    args = parser.parse_args()

    all_contacts_js = []
    dataset_names = []
    ng_infos = {}

    def _process_path(path):
        dset_name = path.rstrip("/").split("/")[-1]
        info = read_info(path)
        resolution = info.get("resolution", [16, 16, 40])
        voxel_offset = info.get("voxel_offset", [0, 0, 0])
        size = info.get("size", [0, 0, 0])
        bbox_start = voxel_offset
        bbox_end = [voxel_offset[i] + size[i] for i in range(3)]

        contacts, _ = collect_dataset(
            path, args.authority, args.gt_authority, bbox_start, bbox_end, resolution
        )

        ng_info = {
            "resolution": resolution,
            "base_url": "https://zetta-portal.vercel.app/?ng=Spelunker",
        }
        for key in [
            "image_path",
            "affinity_path",
            "segmentation_path",
            "ground_truth_path",
            "nucleus_path",
        ]:
            val = info.get(key)
            if val:
                ng_info[key] = val

        contacts_js = []
        for c in contacts:
            contacts_js.append({
                "a": str(c["seg_a"]),
                "b": str(c["seg_b"]),
                "c": [round(c["com"][0], 1), round(c["com"][1], 1), round(c["com"][2], 1)],
                "ma": round(c["mean_affinity"], 4),
                "ss": round(c["shape_score"], 4),
                "gt": 1 if c["gt_merge"] else 0,
                "nf": c["n_faces"],
                "d": dset_name,
            })
        return dset_name, ng_info, contacts_js

    with ThreadPoolExecutor(max_workers=len(args.paths)) as executor:
        futures = {executor.submit(_process_path, p): p for p in args.paths}
        for future in as_completed(futures):
            dset_name, ng_info, contacts_js = future.result()
            dataset_names.append(dset_name)
            ng_infos[dset_name] = ng_info
            all_contacts_js.extend(contacts_js)

    # Sort dataset_names to match input order
    path_order = {p.rstrip("/").split("/")[-1]: i for i, p in enumerate(args.paths)}
    dataset_names.sort(key=lambda n: path_order.get(n, 0))

    print(f"\nTotal: {len(all_contacts_js)} contacts across {len(dataset_names)} datasets")

    # Serialize
    data_json = json.dumps(
        {
            "contacts": all_contacts_js,
            "datasets": dataset_names,
            "ngInfos": ng_infos,
            "authority": args.authority,
        },
        separators=(",", ":"),
    )
    print(f"Data: {len(data_json) / 1e6:.1f}MB JSON")

    html = _build_html(data_json, len(all_contacts_js), args.authority)
    print(f"Writing {len(html) / 1e6:.1f}MB to {args.output}...")
    with open(args.output, "w") as f:
        f.write(html)
    print(f"Wrote {args.output}")


def _build_html(data_json, n_contacts, authority):
    return _HTML_TEMPLATE.substitute(
        data_json=data_json,
        n_contacts=n_contacts,
        authority=authority,
    )


_HTML_TEMPLATE = Template(r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Validation Results Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
* { box-sizing: border-box; }
body { font-family: sans-serif; margin: 12px; background: #fafafa; }
h1 { margin: 0 0 4px 0; font-size: 28px; }
.info-bar { font-size: 16px; color: #444; background: #eef2f7; border: 1px solid #ccd;
             border-radius: 4px; padding: 6px 10px; margin-bottom: 8px; line-height: 1.6; }
.info-bar b { color: #222; }
.info-bar .info-path { font-family: monospace; font-size: 14px; color: #555; word-break: break-all; }
.header { display: flex; align-items: center; gap: 16px; margin-bottom: 10px; flex-wrap: wrap; }
.summary { color: #555; font-size: 18px; }
.legend-info { font-size: 16px; color: #666; }
.legend-info span { font-weight: bold; padding: 1px 6px; border-radius: 3px; margin: 0 2px; }
.lg-merge { background: rgba(76,175,80,0.4); }
.lg-nomerge { background: rgba(229,57,53,0.4); }
.main-layout { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.cell { background: white; border: 1px solid #ddd; border-radius: 6px;
         padding: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.cell h3 { margin: 0 0 4px 0; font-size: 16px; color: #333; }
.full-width { grid-column: 1 / -1; }
.filter-row { display: flex; align-items: center; gap: 8px; padding: 4px 8px;
              background: #f5f5f5; border-radius: 4px; font-size: 14px; margin-top: 4px; flex-wrap: wrap; }
.filter-row label { cursor: pointer; white-space: nowrap; }
.filter-row input[type=range] { width: 200px; }
.filter-row .range-val { font-family: monospace; min-width: 50px; }
.cm-grid { display: grid; grid-template-columns: auto 1fr 1fr; gap: 2px; font-size: 15px; max-width: 300px; }
.cm-cell { padding: 8px 12px; text-align: center; border-radius: 3px; cursor: pointer;
           transition: background 0.15s; }
.cm-cell:hover { filter: brightness(0.9); }
.cm-header { font-weight: bold; padding: 6px 12px; text-align: center; color: #555; }
.cm-pair { display: flex; gap: 20px; flex-wrap: wrap; }
.cm-half { flex: 1; min-width: 320px; }
.cm-active { outline: 3px solid #1a73e8; outline-offset: -1px; }
.auc-label { font-size: 14px; color: #555; margin-top: 2px; }
.link-panel { background: #f0f4f8; border: 1px solid #ccd; border-radius: 4px;
               padding: 6px 8px; margin-top: 4px; display: none;
               font-size: 14px; line-height: 1.5; max-height: 500px; overflow-y: auto; }
.link-panel a { color: #1a0dab; }
.link-panel .lbl-merge { color: #2e7d32; font-weight: bold; }
.link-panel .lbl-nomerge { color: #c62828; font-weight: bold; }
.link-panel .panel-header { font-weight: bold; margin-bottom: 2px; display: flex;
             align-items: center; justify-content: space-between; }
.link-panel .panel-close { cursor: pointer; color: #888; font-size: 22px; line-height: 1;
             padding: 0 4px; border-radius: 3px; }
.link-panel .panel-close:hover { background: #ddd; color: #333; }
.link-panel .cat-header { font-weight: bold; margin-top: 6px; padding: 2px 4px; border-radius: 3px; }
.link-panel .cat-merge { background: rgba(76,175,80,0.15); color: #2e7d32; }
.link-panel .cat-nomerge { background: rgba(229,57,53,0.15); color: #c62828; }
.link-panel table { border-collapse: collapse; width: 100%; margin-top: 2px; margin-bottom: 4px; }
.link-panel th { background: #e8ecf0; padding: 2px 4px; text-align: right; font-size: 13px;
                  border: 1px solid #ccd; white-space: nowrap; }
.link-panel td { padding: 2px 4px; text-align: right; border: 1px solid #dde; font-size: 13px;
                  white-space: nowrap; }
.link-panel td:first-child { text-align: left; }
.link-panel th:first-child { text-align: left; }
.link-panel a.ngl-link { cursor: pointer; }
.link-panel tr.ngl-active { background: rgba(74, 144, 217, 0.15); }
.progress-wrap { margin: 40px auto; max-width: 500px; text-align: center; }
.progress-text { font-size: 18px; color: #555; margin-bottom: 8px; }
.progress-bar-bg { height: 4px; background: #e0e0e0; border-radius: 2px; overflow: hidden; }
.progress-bar-inner { height: 100%; width: 30%; background: #4a90d9; border-radius: 2px;
                       animation: progress-slide 1.2s ease-in-out infinite; }
@keyframes progress-slide { 0% { margin-left: -30%; } 100% { margin-left: 100%; } }
.ann-btn { display: inline-block; width: 22px; height: 22px; line-height: 22px; text-align: center;
           border: 1px solid #ccc; border-radius: 3px; cursor: pointer; font-size: 13px;
           background: #f8f8f8; margin: 0 1px; user-select: none; }
.ann-btn:hover { background: #e8e8e8; }
.ann-btn.active-correct { background: #4caf50; color: white; border-color: #388e3c; }
.ann-btn.active-wrong { background: #e53935; color: white; border-color: #c62828; }
.ann-btn.active-unclear { background: #ffc107; color: #333; border-color: #f9a825; }
.ann-note { width: 70px; padding: 1px 3px; border: 1px solid #ccc; border-radius: 3px; font-size: 12px; }
.ann-toolbar { display: flex; align-items: center; gap: 6px; }
.ann-toolbar button { padding: 3px 8px; border: 1px solid #aab; border-radius: 4px;
                      background: #f0f4f8; cursor: pointer; font-size: 14px; }
.ann-toolbar button:hover { background: #dde4ec; }
.page-nav { display: flex; align-items: center; gap: 6px; margin: 4px 0; font-size: 13px; }
.page-nav button { padding: 2px 8px; border: 1px solid #bbb; border-radius: 3px;
                   background: #f5f5f5; cursor: pointer; font-size: 12px; }
.page-nav button:hover { background: #e0e0e0; }
.page-nav button:disabled { opacity: 0.4; cursor: default; }
.page-nav select { padding: 1px 4px; border: 1px solid #bbb; border-radius: 3px; font-size: 12px; }
.scatter-wrap { display: grid; grid-template-columns: 40px 1fr; grid-template-rows: auto auto; gap: 0; }
.scatter-vslider-wrap { grid-column: 1; grid-row: 1; display: flex; flex-direction: column;
                        align-items: center; justify-content: center; gap: 0; }
.scatter-vslider { writing-mode: vertical-lr; direction: rtl; flex: 1; min-height: 0; }
.scatter-plot-area { grid-column: 2; grid-row: 1; }
.scatter-hslider-wrap { grid-column: 2; grid-row: 2; padding: 2px 0; }
.scatter-vslider-label { grid-column: 1; grid-row: 2; }
.dual-range { position: relative; height: 24px; }
.dual-range input[type=range] { position: absolute; left: 0; right: 0; top: 0; width: 100%;
    pointer-events: none; -webkit-appearance: none; appearance: none; background: transparent; height: 24px; }
.dual-range input[type=range]::-webkit-slider-thumb { pointer-events: auto; -webkit-appearance: none;
    width: 14px; height: 14px; border-radius: 50%; background: #4a90d9; border: 2px solid white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3); cursor: pointer; position: relative; z-index: 2; }
.dual-range input[type=range]::-moz-range-thumb { pointer-events: auto; width: 14px; height: 14px;
    border-radius: 50%; background: #4a90d9; border: 2px solid white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3); cursor: pointer; }
.dual-range input[type=range]::-webkit-slider-runnable-track { height: 4px; background: #ddd; border-radius: 2px; }
.dual-range input[type=range]::-moz-range-track { height: 4px; background: #ddd; border-radius: 2px; }
.dual-range-labels { display: flex; justify-content: space-between; font-family: monospace; font-size: 12px; }
.dual-range-v { position: relative; width: 24px; flex: 1; min-height: 0; }
.dual-range-v input[type=range] { position: absolute; top: 0; bottom: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; -webkit-appearance: none; appearance: none; background: transparent;
    writing-mode: vertical-lr; direction: rtl; }
.dual-range-v input[type=range]::-webkit-slider-thumb { pointer-events: auto; -webkit-appearance: none;
    width: 14px; height: 14px; border-radius: 50%; background: #4a90d9; border: 2px solid white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3); cursor: pointer; }
.dual-range-v input[type=range]::-moz-range-thumb { pointer-events: auto; width: 14px; height: 14px;
    border-radius: 50%; background: #4a90d9; border: 2px solid white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3); cursor: pointer; }
.slider-with-input { display: flex; align-items: center; gap: 6px; }
.slider-with-input input[type=range] { flex: 1; }
.slider-with-input input[type=number] { width: 60px; padding: 1px 3px; border: 1px solid #ccc;
    border-radius: 3px; font-size: 12px; font-family: monospace; }
.range-num { width: 48px; padding: 1px 3px; border: 1px solid #ccc; border-radius: 3px;
    font-size: 11px; font-family: monospace; text-align: center; }
.dset-filter-list { display: flex; flex-direction: column; gap: 1px; font-size: 14px; max-height: 300px; overflow-y: auto; }
.dset-filter-list label { cursor: pointer; white-space: nowrap; padding: 1px 4px; }
.dset-filter-list label:hover { background: #e8ecf0; }
</style>
</head>
<body>
<div class="info-bar">
    <b>Authority:</b> <span class="info-path">$authority</span>
    &nbsp; <b>Total contacts:</b> $n_contacts
</div>
<div class="header">
    <h1>Validation Results Dashboard</h1>
    <div class="summary" id="summary">Loading...</div>
    <div class="legend-info">
        <span class="lg-merge">merge (GT)</span>
        <span class="lg-nomerge">no_merge (GT)</span>
    </div>
    <div style="font-size:14px;">
        Curves/CM:
        <label><input type="radio" name="filter_mode" value="filtered" checked> Filtered</label>
        <label><input type="radio" name="filter_mode" value="unfiltered"> Unfiltered</label>
    </div>
    <div class="ann-toolbar">
        <button id="ann_export">Export annotations</button>
        <button id="ann_import">Import annotations</button>
        <input type="file" id="ann_file_input" accept=".json" style="display:none">
    </div>
</div>
<div id="progress-wrap" class="progress-wrap">
    <div class="progress-text" id="progress-text">Loading data...</div>
    <div class="progress-bar-bg"><div class="progress-bar-inner"></div></div>
</div>

<div id="main-content" style="display:none">

<!-- Dataset filter -->
<div class="cell full-width" id="filter-panel">
    <div id="dataset-filters"></div>
</div>

<div class="main-layout">
    <!-- Scatter plot with integrated sliders -->
    <div class="cell">
        <h3>Mean Affinity vs Shape Score</h3>
        <div class="scatter-wrap">
            <div class="scatter-vslider-wrap">
                <input type="number" class="range-num" id="ss-hi-val" value="1.00" min="0" max="1" step="0.01">
                <div class="dual-range-v">
                    <input type="range" id="ss-hi" min="0" max="1" step="0.01" value="1">
                    <input type="range" id="ss-lo" min="0" max="1" step="0.01" value="0">
                </div>
                <input type="number" class="range-num" id="ss-lo-val" value="0.00" min="0" max="1" step="0.01">
            </div>
            <div class="scatter-plot-area" id="scatter-plot" style="width:100%;height:450px;"></div>
            <div class="scatter-vslider-label"></div>
            <div class="scatter-hslider-wrap">
                <div class="dual-range-labels">
                    <input type="number" class="range-num" id="ma-lo-val" value="0.00" min="0" max="1" step="0.01">
                </div>
                <div class="dual-range">
                    <input type="range" id="ma-lo" min="0" max="1" step="0.01" value="0">
                    <input type="range" id="ma-hi" min="0" max="1" step="0.01" value="1">
                </div>
                <div class="dual-range-labels">
                    <input type="number" class="range-num" id="ma-hi-val" value="1.00" min="0" max="1" step="0.01">
                </div>
            </div>
        </div>
        <div class="link-panel" id="scatter-sample-panel"></div>
    </div>

    <!-- Both confusion matrices side by side -->
    <div class="cell">
        <h3>Confusion Matrices</h3>
        <div class="cm-pair">
            <div class="cm-half">
                <div class="filter-row slider-with-input">
                    <b>Shape Thr:</b>
                    <input type="range" id="cm-threshold" min="0" max="1" step="0.01" value="0.5">
                    <input type="number" id="cm-thr-val" value="0.50" min="0" max="1" step="0.01">
                </div>
                <div class="cm-grid" id="cm-grid">
                    <div class="cm-header"></div>
                    <div class="cm-header">Pred Merge</div>
                    <div class="cm-header">Pred No-Merge</div>
                    <div class="cm-header">GT Merge</div>
                    <div class="cm-cell" id="cm-tp" data-cat="tp">-</div>
                    <div class="cm-cell" id="cm-fn" data-cat="fn">-</div>
                    <div class="cm-header">GT No-Merge</div>
                    <div class="cm-cell" id="cm-fp" data-cat="fp">-</div>
                    <div class="cm-cell" id="cm-tn" data-cat="tn">-</div>
                </div>
                <div id="cm-stats" style="font-size:13px;color:#555;margin-top:4px;"></div>
            </div>
            <div class="cm-half">
                <div class="filter-row slider-with-input">
                    <b>Affinity Thr:</b>
                    <input type="range" id="ma-cm-threshold" min="0" max="1" step="0.01" value="0.5">
                    <input type="number" id="ma-cm-thr-val" value="0.50" min="0" max="1" step="0.01">
                </div>
                <div class="cm-grid" id="ma-cm-grid">
                    <div class="cm-header"></div>
                    <div class="cm-header">Pred Merge</div>
                    <div class="cm-header">Pred No-Merge</div>
                    <div class="cm-header">GT Merge</div>
                    <div class="cm-cell" id="ma-cm-tp" data-cat="ma-tp">-</div>
                    <div class="cm-cell" id="ma-cm-fn" data-cat="ma-fn">-</div>
                    <div class="cm-header">GT No-Merge</div>
                    <div class="cm-cell" id="ma-cm-fp" data-cat="ma-fp">-</div>
                    <div class="cm-cell" id="ma-cm-tn" data-cat="ma-tn">-</div>
                </div>
                <div id="ma-cm-stats" style="font-size:13px;color:#555;margin-top:4px;"></div>
            </div>
        </div>
        <div class="link-panel" id="cm-sample-panel"></div>
        <div class="link-panel" id="ma-cm-sample-panel"></div>
    </div>

    <!-- Histograms -->
    <div class="cell">
        <h3>Shape Score Distribution</h3>
        <div id="hist-ss" style="width:100%;height:300px;"></div>
        <div class="link-panel" id="hist-ss-panel"></div>
    </div>
    <div class="cell">
        <h3>Mean Affinity Distribution</h3>
        <div id="hist-ma" style="width:100%;height:300px;"></div>
        <div class="link-panel" id="hist-ma-panel"></div>
    </div>

    <!-- PR curve -->
    <div class="cell">
        <h3>Precision-Recall Curve</h3>
        <div id="pr-plot" style="width:100%;height:350px;"></div>
        <div class="auc-label" id="pr-auc"></div>
    </div>

    <!-- ROC curve -->
    <div class="cell">
        <h3>ROC Curve</h3>
        <div id="roc-plot" style="width:100%;height:350px;"></div>
        <div class="auc-label" id="roc-auc"></div>
    </div>

</div>
</div>

<script>
// --- Globals ---
var contacts, datasets, ngInfos, authority;
var nContacts;
var currentMask = null;
var activeCmCat = null;
var activeMaCmCat = null;
var cmThreshold = 0.5;
var maThreshold = 0.5;
var datasetFilters = {};
var maLo = 0, maHi = 1, ssLo = 0, ssHi = 1;
var annotations = {};
var STORAGE_KEY = "validation_results_annotations";
var filterMode = "filtered";
var curveMask = null;
var hoverDataset = null;

var GT_NAMES = ["no_merge", "merge"];
var GT_COLORS_SCATTER = ["rgb(229,57,53)", "rgb(76,175,80)"];

function contactKey(c) { return c.d + "|" + c.a + "|" + c.b; }

// --- Annotations ---
function saveAnnotations() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations)); } catch(e) {}
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
    a.download = "validation_annotations.json";
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
        } catch(err) { alert("Invalid JSON file"); }
    };
    reader.readAsText(file);
}

// --- Data is embedded inline as JSON ---

// --- NG URL builder (per-dataset ngInfo) ---
function buildNgUrl(c) {
    var info = ngInfos[c.d];
    if (!info) return "#";
    var res = info.resolution;
    var pos = [c.c[0]/res[0], c.c[1]/res[1], c.c[2]/res[2]];
    var dims = {"x":[res[0]*1e-9,"m"],"y":[res[1]*1e-9,"m"],"z":[res[2]*1e-9,"m"]};
    var layers = [];
    if (info.image_path) {
        var src = info.image_path.indexOf("://")>=0 ? info.image_path : "precomputed://"+info.image_path;
        layers.push({"type":"image","source":src,"tab":"source","name":"Image"});
    }
    if (info.affinity_path) {
        var src = info.affinity_path.startsWith("precomputed://") ? info.affinity_path : "precomputed://"+info.affinity_path;
        layers.push({"type":"image","source":src,"tab":"source","channelDimensions":{"c^":[1,""]},"name":"Affinity"});
    }
    if (info.segmentation_path) {
        var src = info.segmentation_path.startsWith("precomputed://") ? info.segmentation_path : "precomputed://"+info.segmentation_path;
        layers.push({"type":"segmentation","source":src,"tab":"segments","name":"Segmentation",
                     "segments":[""+c.a,""+c.b]});
    }
    if (info.ground_truth_path) {
        var src = info.ground_truth_path.startsWith("precomputed://") ? info.ground_truth_path : "precomputed://"+info.ground_truth_path;
        layers.push({"type":"segmentation","source":src,"tab":"source","name":"Ground Truth"});
    }
    var state = {
        "dimensions":dims,"position":pos,
        "crossSectionScale":0.1,"projectionScale":220,
        "layers":layers,
        "selectedLayer":{"visible":true,"layer":info.segmentation_path?"Segmentation":"Image"},
        "layout":"xy-3d","showSlices":false
    };
    return info.base_url + "#!" + encodeURIComponent(JSON.stringify(state));
}

// --- Filter mask ---
function computeFilterMask() {
    var mask = new Uint8Array(nContacts);
    for (var i = 0; i < nContacts; i++) {
        var c = contacts[i];
        if (hoverDataset !== null) {
            if (hoverDataset === "__all__") { /* pass all datasets */ }
            else if (c.d !== hoverDataset) continue;
        } else {
            if (!datasetFilters[c.d]) continue;
        }
        if (c.ma < maLo || c.ma > maHi) continue;
        if (c.ss < ssLo || c.ss > ssHi) continue;
        mask[i] = 1;
    }
    return mask;
}

// --- Deterministic ordering ---
function mulberry32(seed) {
    return function() {
        seed |= 0; seed = seed + 0x6D2B79F5 | 0;
        var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}
var contactOrder = null;
function initContactOrder() {
    var rng = mulberry32(42);
    contactOrder = new Float64Array(nContacts);
    for (var i = 0; i < nContacts; i++) contactOrder[i] = rng();
}

// --- PR/ROC computation in JS ---
function computePrCurve(gt, scores, mask) {
    var pairs = [];
    for (var i = 0; i < gt.length; i++) {
        if (!mask[i]) continue;
        pairs.push([scores[i], gt[i]]);
    }
    if (pairs.length === 0) return {precision: [], recall: [], thresholds: [], auc: 0, bestF1: null};
    pairs.sort(function(a, b) { return b[0] - a[0]; });
    var tp = 0, fp = 0;
    var totalPos = 0;
    for (var i = 0; i < pairs.length; i++) totalPos += pairs[i][1];
    if (totalPos === 0) return {precision: [1], recall: [0], thresholds: [1], auc: 0, bestF1: null};

    var precisions = [1], recalls = [0], thresholds = [pairs[0][0] + 0.001];
    var prevRecall = 0, aucSum = 0;
    var bestF1 = 0, bestF1Idx = -1;
    for (var i = 0; i < pairs.length; i++) {
        if (pairs[i][1]) tp++; else fp++;
        var prec = tp / (tp + fp);
        var rec = tp / totalPos;
        precisions.push(prec);
        recalls.push(rec);
        thresholds.push(pairs[i][0]);
        var f1 = 2 * prec * rec / (prec + rec + 1e-8);
        if (f1 > bestF1) { bestF1 = f1; bestF1Idx = precisions.length - 1; }
        aucSum += prec * (rec - prevRecall);
        prevRecall = rec;
    }
    return {
        precision: precisions, recall: recalls, thresholds: thresholds, auc: aucSum,
        bestF1: bestF1Idx >= 0 ? {f1: bestF1, precision: precisions[bestF1Idx], recall: recalls[bestF1Idx], threshold: thresholds[bestF1Idx]} : null
    };
}

function findOperatingPoint(prCurve, threshold) {
    for (var i = 1; i < prCurve.thresholds.length; i++) {
        if (prCurve.thresholds[i] <= threshold) {
            return {recall: prCurve.recall[i], precision: prCurve.precision[i]};
        }
    }
    var last = prCurve.recall.length - 1;
    return {recall: prCurve.recall[last], precision: prCurve.precision[last]};
}

function computeRocCurve(gt, scores, mask) {
    var pairs = [];
    for (var i = 0; i < gt.length; i++) {
        if (!mask[i]) continue;
        pairs.push([scores[i], gt[i]]);
    }
    if (pairs.length === 0) return {fpr: [], tpr: [], thresholds: [], auc: 0};
    pairs.sort(function(a, b) { return b[0] - a[0]; });
    var totalPos = 0, totalNeg = 0;
    for (var i = 0; i < pairs.length; i++) {
        if (pairs[i][1]) totalPos++; else totalNeg++;
    }
    if (totalPos === 0 || totalNeg === 0) return {fpr: [0,1], tpr: [0,1], thresholds: [1,0], auc: 0.5};

    var tp = 0, fp = 0;
    var fprs = [0], tprs = [0], thresholds = [pairs[0][0] + 0.001];
    var prevFpr = 0, aucSum = 0;
    for (var i = 0; i < pairs.length; i++) {
        if (pairs[i][1]) tp++; else fp++;
        var fpr = fp / totalNeg;
        var tpr = tp / totalPos;
        fprs.push(fpr);
        tprs.push(tpr);
        thresholds.push(pairs[i][0]);
        aucSum += tpr * (fpr - prevFpr);
        prevFpr = fpr;
    }
    return {fpr: fprs, tpr: tprs, thresholds: thresholds, auc: aucSum};
}

function findRocOperatingPoint(rocCurve, threshold) {
    for (var i = 1; i < rocCurve.thresholds.length; i++) {
        if (rocCurve.thresholds[i] <= threshold) {
            return {fpr: rocCurve.fpr[i], tpr: rocCurve.tpr[i]};
        }
    }
    var last = rocCurve.fpr.length - 1;
    return {fpr: rocCurve.fpr[last], tpr: rocCurve.tpr[last]};
}

// --- Confusion matrix ---
function computeConfusionMatrix(mask, threshold) {
    var tp = 0, fp = 0, fn = 0, tn = 0;
    for (var i = 0; i < nContacts; i++) {
        if (!mask[i]) continue;
        var pred = contacts[i].ss >= threshold ? 1 : 0;
        var gt = contacts[i].gt;
        if (pred && gt) tp++;
        else if (pred && !gt) fp++;
        else if (!pred && gt) fn++;
        else tn++;
    }
    return {tp: tp, fp: fp, fn: fn, tn: tn};
}

// --- Sample table rendering ---
function renderContactRow(c, ann) {
    var key = contactKey(c);
    var url = buildNgUrl(c);
    var gtLabel = c.gt ? "merge" : "no_merge";
    var html = '<tr data-key="' + key.replace(/"/g, '&quot;') + '">';
    html += '<td><a href="' + url + '" target="_blank" class="ngl-link">ngl</a></td>';
    html += '<td>' + c.a + '</td><td>' + c.b + '</td>';
    html += '<td>' + c.ma.toFixed(3) + '</td>';
    html += '<td>' + c.ss.toFixed(3) + '</td>';
    html += '<td class="lbl-' + (c.gt ? 'merge' : 'nomerge') + '">' + gtLabel + '</td>';
    html += '<td>' + c.nf + '</td>';
    html += '<td>' + c.d + '</td>';
    html += '<td><span class="ann-btn' + (ann.label === "correct" ? ' active-correct' : '') + '" data-label="correct">&#x2713;</span></td>';
    html += '<td><span class="ann-btn' + (ann.label === "wrong" ? ' active-wrong' : '') + '" data-label="wrong">&#x2717;</span></td>';
    html += '<td><span class="ann-btn' + (ann.label && ann.label !== "correct" && ann.label !== "wrong" ? ' active-unclear' : '') + '" data-label="unclear">?</span></td>';
    html += '<td><input class="ann-note" value="' + (ann.note || '').replace(/"/g, '&quot;') + '" placeholder="note"></td>';
    html += '</tr>';
    return html;
}

function wireAnnotationHandlers(panelEl) {
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
        });
    });
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

var N_SAMPLES = 20;

function stratifiedSample(indices) {
    var byGt = {0: [], 1: []};
    for (var i = 0; i < indices.length; i++) {
        byGt[contacts[indices[i]].gt].push(indices[i]);
    }
    var groups = [];
    for (var g = 1; g >= 0; g--) {
        byGt[g].sort(function(a, b) { return contactOrder[a] - contactOrder[b]; });
        groups.push({
            gt: g,
            total: byGt[g].length,
            samples: byGt[g].slice(0, N_SAMPLES),
        });
    }
    return groups;
}

function renderSamplePanel(panelEl, headerText, indices) {
    var groups = stratifiedSample(indices);
    var totalInBin = groups[0].total + groups[1].total;
    if (totalInBin === 0) {
        var closeBtn = '<span class="panel-close" onclick="this.closest(\'.link-panel\').style.display=\'none\'">\u2715</span>';
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

function renderAnnotationSummary(groups) {
    var totalAll = groups[0].total + groups[1].total;
    var catNames = {1: "merge", 0: "no_merge"};
    var html = '<div style="font-size:13px;color:#555;margin-top:4px;border-top:1px solid #ddd;padding-top:4px;">';
    var wCorrect = 0, wWrong = 0, wUnclear = 0, wTotal = 0;
    var hasAny = false;
    for (var gi = 0; gi < groups.length; gi++) {
        var grp = groups[gi];
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
        html += '<div>' + catNames[grp.gt] + ': ' +
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

function renderSamplePanelPage(panelEl) {
    var groups = panelEl._groups;
    var headerText = panelEl._headerText;
    var totalInBin = panelEl._totalInBin;
    var totalSampled = groups[0].samples.length + groups[1].samples.length;
    var page = panelEl._page;
    var pageSize = panelEl._pageSize;
    var maxSamples = Math.max(groups[0].samples.length, groups[1].samples.length);
    var totalPages = Math.max(1, Math.ceil(maxSamples / pageSize));
    if (page >= totalPages) page = totalPages - 1;
    if (page < 0) page = 0;
    panelEl._page = page;
    var start = page * pageSize;

    var closeBtn = '<span class="panel-close" onclick="this.closest(\'.link-panel\').style.display=\'none\'">\u2715</span>';
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

    var thead = '<tr><th>link</th><th>seg_a</th><th>seg_b</th><th>mean_aff</th><th>shape_sc</th><th>GT</th><th>faces</th><th>dataset</th><th>&#x2713;</th><th>&#x2717;</th><th>?</th><th>note</th></tr>';
    var catNames = {1: "merge", 0: "no_merge"};
    var catCss = {1: "cat-merge", 0: "cat-nomerge"};

    for (var gi = 0; gi < groups.length; gi++) {
        var grp = groups[gi];
        if (grp.total === 0) continue;
        var pct = totalInBin > 0 ? (100 * grp.total / totalInBin).toFixed(1) : '0.0';
        html += '<div class="cat-header ' + catCss[grp.gt] + '">' + catNames[grp.gt] + ': ' + grp.total + ' (' + pct + '%)</div>';
        var gStart = Math.min(start, grp.samples.length);
        var gEnd = Math.min(start + pageSize, grp.samples.length);
        if (gStart >= gEnd) continue;
        html += '<table>' + thead;
        for (var s = gStart; s < gEnd; s++) {
            var c = contacts[grp.samples[s]];
            var ann = annotations[contactKey(c)] || {};
            html += renderContactRow(c, ann);
        }
        html += '</table>';
    }

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

    wireAnnotationHandlers(panelEl);
}

// --- Scatter plot ---
function buildScatter() {
    var mergeX = [], mergeY = [], mergeIdx = [];
    var noMergeX = [], noMergeY = [], noMergeIdx = [];
    for (var i = 0; i < nContacts; i++) {
        if (!curveMask[i]) continue;
        var c = contacts[i];
        if (c.gt) {
            mergeX.push(c.ma); mergeY.push(c.ss); mergeIdx.push(i);
        } else {
            noMergeX.push(c.ma); noMergeY.push(c.ss); noMergeIdx.push(i);
        }
    }

    var traces = [
        {
            x: noMergeX, y: noMergeY, customdata: noMergeIdx,
            mode: 'markers', type: 'scattergl', name: 'no_merge',
            marker: {color: 'rgb(229,57,53)', size: 5, symbol: 'cross-thin', line: {width: 1, color: 'rgb(229,57,53)'}},
        },
        {
            x: mergeX, y: mergeY, customdata: mergeIdx,
            mode: 'markers', type: 'scattergl', name: 'merge',
            marker: {color: 'rgb(76,175,80)', size: 5, symbol: 'cross-thin', line: {width: 1, color: 'rgb(76,175,80)'}},
        },
        {
            x: [0, 1], y: [cmThreshold, cmThreshold],
            mode: 'lines', name: 'Shape thr',
            line: {color: 'rgba(26,115,232,0.5)', dash: 'dash', width: 1.5},
            showlegend: true,
        },
        {
            x: [maThreshold, maThreshold], y: [0, 1],
            mode: 'lines', name: 'Aff thr',
            line: {color: 'rgba(255,152,0,0.5)', dash: 'dash', width: 1.5},
            showlegend: true,
        }
    ];

    var layout = {
        xaxis: {title: 'Mean Affinity', range: [0, 1]},
        yaxis: {title: 'Shape Score', range: [0, 1]},
        margin: {l: 50, r: 20, t: 10, b: 40},
        legend: {x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.7)'},
        hovermode: 'closest',
    };

    Plotly.newPlot('scatter-plot', traces, layout, {responsive: true});

    document.getElementById('scatter-plot').on('plotly_click', function(data) {
        if (!data.points || data.points.length === 0) return;
        var pt = data.points[0];
        var idx = pt.customdata;
        if (idx === undefined) return;
        var c = contacts[idx];
        var url = buildNgUrl(c);
        openNgPopup(url);
        renderSamplePanel(
            document.getElementById('scatter-sample-panel'),
            'Selected contact: ' + c.a + '/' + c.b,
            [idx]
        );
    });
}

function updateScatter() {
    var mergeX = [], mergeY = [], mergeIdx = [];
    var noMergeX = [], noMergeY = [], noMergeIdx = [];
    for (var i = 0; i < nContacts; i++) {
        if (!curveMask[i]) continue;
        var c = contacts[i];
        if (c.gt) {
            mergeX.push(c.ma); mergeY.push(c.ss); mergeIdx.push(i);
        } else {
            noMergeX.push(c.ma); noMergeY.push(c.ss); noMergeIdx.push(i);
        }
    }
    Plotly.restyle('scatter-plot', {
        x: [noMergeX, mergeX],
        y: [noMergeY, mergeY],
        customdata: [noMergeIdx, mergeIdx],
    }, [0, 1]);
    Plotly.restyle('scatter-plot', {y: [[cmThreshold, cmThreshold]]}, [2]);
    Plotly.restyle('scatter-plot', {x: [[maThreshold, maThreshold]]}, [3]);
}

// --- PR/ROC plots ---
function updateCurves() {
    var gt = [], ssScores = [], maScores = [];
    for (var i = 0; i < nContacts; i++) {
        gt.push(contacts[i].gt);
        ssScores.push(contacts[i].ss);
        maScores.push(contacts[i].ma);
    }

    // PR curves
    var prModel = computePrCurve(gt, ssScores, curveMask);
    var prMA = computePrCurve(gt, maScores, curveMask);
    // Random baseline
    var nFiltered = 0, nPos = 0;
    for (var i = 0; i < nContacts; i++) {
        if (curveMask[i]) { nFiltered++; nPos += contacts[i].gt; }
    }
    var baseRate = nFiltered > 0 ? nPos / nFiltered : 0.5;

    // F1 optimal points
    var prTraces = [
        {x: prModel.recall, y: prModel.precision, mode: 'lines', name: 'Model (AUC=' + prModel.auc.toFixed(3) + ')',
         line: {color: '#1a73e8', width: 2}},
        {x: prMA.recall, y: prMA.precision, mode: 'lines', name: 'Mean Aff (AUC=' + prMA.auc.toFixed(3) + ')',
         line: {color: '#ff9800', width: 2, dash: 'dash'}},
        {x: [0, 1], y: [baseRate, baseRate], mode: 'lines', name: 'Random',
         line: {color: '#999', width: 1, dash: 'dot'}},
    ];
    // F1 optimal markers
    if (prModel.bestF1) {
        prTraces.push({x: [prModel.bestF1.recall], y: [prModel.bestF1.precision], mode: 'markers+text',
            name: 'Model F1*=' + prModel.bestF1.f1.toFixed(3) + ' @' + prModel.bestF1.threshold.toFixed(3),
            marker: {color: '#1a73e8', size: 10, symbol: 'star'}, textposition: 'top right',
            text: ['F1=' + prModel.bestF1.f1.toFixed(3)], textfont: {size: 11, color: '#1a73e8'}});
    }
    if (prMA.bestF1) {
        prTraces.push({x: [prMA.bestF1.recall], y: [prMA.bestF1.precision], mode: 'markers+text',
            name: 'MA F1*=' + prMA.bestF1.f1.toFixed(3) + ' @' + prMA.bestF1.threshold.toFixed(3),
            marker: {color: '#ff9800', size: 10, symbol: 'star'}, textposition: 'top right',
            text: ['F1=' + prMA.bestF1.f1.toFixed(3)], textfont: {size: 11, color: '#ff9800'}});
    }
    // Current threshold operating points
    var ssOp = findOperatingPoint(prModel, cmThreshold);
    prTraces.push({x: [ssOp.recall], y: [ssOp.precision], mode: 'markers',
        name: 'Shape thr=' + cmThreshold.toFixed(2),
        marker: {color: '#1a73e8', size: 12, symbol: 'diamond', line: {width: 2, color: '#0d47a1'}}});
    var maOp = findOperatingPoint(prMA, maThreshold);
    prTraces.push({x: [maOp.recall], y: [maOp.precision], mode: 'markers',
        name: 'Aff thr=' + maThreshold.toFixed(2),
        marker: {color: '#ff9800', size: 12, symbol: 'diamond', line: {width: 2, color: '#e65100'}}});

    Plotly.react('pr-plot', prTraces, {
        xaxis: {title: 'Recall', range: [0, 1]},
        yaxis: {title: 'Precision', range: [0, 1.05]},
        margin: {l: 50, r: 20, t: 10, b: 40},
        legend: {x: 0.01, y: 0.01, bgcolor: 'rgba(255,255,255,0.7)'},
    }, {responsive: true});
    document.getElementById('pr-auc').innerHTML =
        '<b>AUC-PR:</b> Model=' + prModel.auc.toFixed(4) +
        ', MeanAff=' + prMA.auc.toFixed(4);

    // ROC curves
    var rocModel = computeRocCurve(gt, ssScores, curveMask);
    var rocMA = computeRocCurve(gt, maScores, curveMask);

    // ROC operating points
    var ssRocOp = findRocOperatingPoint(rocModel, cmThreshold);
    var maRocOp = findRocOperatingPoint(rocMA, maThreshold);

    Plotly.react('roc-plot', [
        {x: rocModel.fpr, y: rocModel.tpr, mode: 'lines', name: 'Model (AUC=' + rocModel.auc.toFixed(3) + ')',
         line: {color: '#1a73e8', width: 2}},
        {x: rocMA.fpr, y: rocMA.tpr, mode: 'lines', name: 'Mean Aff (AUC=' + rocMA.auc.toFixed(3) + ')',
         line: {color: '#ff9800', width: 2, dash: 'dash'}},
        {x: [0, 1], y: [0, 1], mode: 'lines', name: 'Random',
         line: {color: '#999', width: 1, dash: 'dot'}},
        {x: [ssRocOp.fpr], y: [ssRocOp.tpr], mode: 'markers',
         name: 'Shape thr=' + cmThreshold.toFixed(2),
         marker: {color: '#1a73e8', size: 12, symbol: 'diamond', line: {width: 2, color: '#0d47a1'}}},
        {x: [maRocOp.fpr], y: [maRocOp.tpr], mode: 'markers',
         name: 'Aff thr=' + maThreshold.toFixed(2),
         marker: {color: '#ff9800', size: 12, symbol: 'diamond', line: {width: 2, color: '#e65100'}}},
    ], {
        xaxis: {title: 'False Positive Rate', range: [0, 1]},
        yaxis: {title: 'True Positive Rate', range: [0, 1.05]},
        margin: {l: 50, r: 20, t: 10, b: 40},
        legend: {x: 0.6, y: 0.01, bgcolor: 'rgba(255,255,255,0.7)'},
    }, {responsive: true});
    document.getElementById('roc-auc').innerHTML =
        '<b>AUC-ROC:</b> Model=' + rocModel.auc.toFixed(4) +
        ', MeanAff=' + rocMA.auc.toFixed(4);
}

// --- Confusion matrix helpers ---
function computeConfusionMatrixMA(mask, threshold) {
    var tp = 0, fp = 0, fn = 0, tn = 0;
    for (var i = 0; i < nContacts; i++) {
        if (!mask[i]) continue;
        var pred = contacts[i].ma >= threshold ? 1 : 0;
        var gt = contacts[i].gt;
        if (pred && gt) tp++;
        else if (pred && !gt) fp++;
        else if (!pred && gt) fn++;
        else tn++;
    }
    return {tp: tp, fp: fp, fn: fn, tn: tn};
}

function renderCmStats(cm) {
    var total = cm.tp + cm.fp + cm.fn + cm.tn;
    var acc = total > 0 ? ((cm.tp + cm.tn) / total * 100).toFixed(1) : '0';
    var prec = (cm.tp + cm.fp) > 0 ? (cm.tp / (cm.tp + cm.fp) * 100).toFixed(1) : '0';
    var rec = (cm.tp + cm.fn) > 0 ? (cm.tp / (cm.tp + cm.fn) * 100).toFixed(1) : '0';
    var f1n = 2 * cm.tp / (2 * cm.tp + cm.fp + cm.fn + 1e-8);
    return '<b>Accuracy:</b> ' + acc + '% &nbsp; ' +
        '<b>Precision:</b> ' + prec + '% &nbsp; ' +
        '<b>Recall:</b> ' + rec + '% &nbsp; ' +
        '<b>F1:</b> ' + (f1n * 100).toFixed(1) + '%';
}

function cmCellHtml(count, total) {
    var pct = total > 0 ? (count / total * 100) : 0;
    return pct.toFixed(1) + '% (' + count + ')';
}

function cmCellColor(count, total, isCorrect) {
    var pct = total > 0 ? count / total : 0;
    var alpha = 0.08 + pct * 0.72;
    if (isCorrect) return 'rgba(76,175,80,' + alpha.toFixed(2) + ')';
    return 'rgba(229,57,53,' + alpha.toFixed(2) + ')';
}

function applyCmCell(id, count, total, isCorrect) {
    var el = document.getElementById(id);
    el.innerHTML = cmCellHtml(count, total);
    el.style.background = cmCellColor(count, total, isCorrect);
}

function updateConfusionMatrix() {
    var cm = computeConfusionMatrix(curveMask, cmThreshold);
    var cmTotal = cm.tp + cm.fp + cm.fn + cm.tn;
    applyCmCell('cm-tp', cm.tp, cmTotal, true);
    applyCmCell('cm-fn', cm.fn, cmTotal, false);
    applyCmCell('cm-fp', cm.fp, cmTotal, false);
    applyCmCell('cm-tn', cm.tn, cmTotal, true);
    document.getElementById('cm-stats').innerHTML = renderCmStats(cm);

    // Affinity CM
    var maCm = computeConfusionMatrixMA(curveMask, maThreshold);
    var maTotal = maCm.tp + maCm.fp + maCm.fn + maCm.tn;
    applyCmCell('ma-cm-tp', maCm.tp, maTotal, true);
    applyCmCell('ma-cm-fn', maCm.fn, maTotal, false);
    applyCmCell('ma-cm-fp', maCm.fp, maTotal, false);
    applyCmCell('ma-cm-tn', maCm.tn, maTotal, true);
    document.getElementById('ma-cm-stats').innerHTML = renderCmStats(maCm);

    // Update active highlights
    document.querySelectorAll('#cm-grid .cm-cell').forEach(function(el) {
        el.classList.toggle('cm-active', el.dataset.cat === activeCmCat);
    });
    document.querySelectorAll('#ma-cm-grid .cm-cell').forEach(function(el) {
        el.classList.toggle('cm-active', el.dataset.cat === activeMaCmCat);
    });
}

function showCmSamples(cat) {
    if (activeCmCat === cat) {
        activeCmCat = null;
        document.getElementById('cm-sample-panel').style.display = 'none';
        document.querySelectorAll('.cm-cell').forEach(function(el) {
            el.classList.remove('cm-active');
        });
        return;
    }
    activeCmCat = cat;

    var indices = [];
    for (var i = 0; i < nContacts; i++) {
        if (!curveMask[i]) continue;
        var pred = contacts[i].ss >= cmThreshold ? 1 : 0;
        var gt = contacts[i].gt;
        var match = false;
        if (cat === 'tp' && pred && gt) match = true;
        if (cat === 'fp' && pred && !gt) match = true;
        if (cat === 'fn' && !pred && gt) match = true;
        if (cat === 'tn' && !pred && !gt) match = true;
        if (match) indices.push(i);
    }

    var names = {tp: 'True Positives', fp: 'False Positives', fn: 'False Negatives', tn: 'True Negatives'};
    renderSamplePanel(
        document.getElementById('cm-sample-panel'),
        names[cat],
        indices
    );
    updateConfusionMatrix();
}

function showMaCmSamples(cat) {
    var realCat = cat.replace('ma-', '');
    if (activeMaCmCat === cat) {
        activeMaCmCat = null;
        document.getElementById('ma-cm-sample-panel').style.display = 'none';
        document.querySelectorAll('#ma-cm-grid .cm-cell').forEach(function(el) {
            el.classList.remove('cm-active');
        });
        return;
    }
    activeMaCmCat = cat;

    var indices = [];
    for (var i = 0; i < nContacts; i++) {
        if (!curveMask[i]) continue;
        var pred = contacts[i].ma >= maThreshold ? 1 : 0;
        var gt = contacts[i].gt;
        var match = false;
        if (realCat === 'tp' && pred && gt) match = true;
        if (realCat === 'fp' && pred && !gt) match = true;
        if (realCat === 'fn' && !pred && gt) match = true;
        if (realCat === 'tn' && !pred && !gt) match = true;
        if (match) indices.push(i);
    }

    var names = {tp: 'True Positives (Aff)', fp: 'False Positives (Aff)', fn: 'False Negatives (Aff)', tn: 'True Negatives (Aff)'};
    renderSamplePanel(
        document.getElementById('ma-cm-sample-panel'),
        names[realCat],
        indices
    );
    updateConfusionMatrix();
}

// --- Histograms ---
var HIST_NBINS = 50;
var histHighlightedSS = -1;
var histHighlightedMA = -1;

function computeHistogram(field, mask, nBins) {
    var edges = [];
    for (var i = 0; i <= nBins; i++) edges.push(i / nBins);
    var mergeCounts = new Array(nBins).fill(0);
    var noMergeCounts = new Array(nBins).fill(0);
    var binIndices = [];
    for (var b = 0; b < nBins; b++) binIndices.push([]);

    for (var i = 0; i < nContacts; i++) {
        if (!mask[i]) continue;
        var val = contacts[i][field];
        var bin = Math.floor(val * nBins);
        if (bin >= nBins) bin = nBins - 1;
        if (bin < 0) bin = 0;
        if (contacts[i].gt) mergeCounts[bin]++;
        else noMergeCounts[bin]++;
        binIndices[bin].push(i);
    }

    var centers = [];
    for (var i = 0; i < nBins; i++) centers.push((edges[i] + edges[i+1]) / 2);
    return {edges: edges, centers: centers, merge: mergeCounts, noMerge: noMergeCounts, binIndices: binIndices};
}

var histYMax = {};

function buildHistogram(divId, panelId, field, title, highlightedRef) {
    // Compute on ALL data to fix y-axis range
    var hAll = computeHistogram(field, computeAllMask(), HIST_NBINS);
    var yMax = 0;
    for (var i = 0; i < hAll.merge.length; i++) {
        yMax = Math.max(yMax, hAll.merge[i] + hAll.noMerge[i]);
    }
    histYMax[divId] = yMax;

    var h = computeHistogram(field, curveMask, HIST_NBINS);

    var traces = [
        {x: h.centers, y: h.merge, type: 'bar', name: 'merge',
         marker: {color: 'rgba(76,175,80,0.8)', line: {width: 0}},
         hoverinfo: 'y', showlegend: false},
        {x: h.centers, y: h.noMerge, type: 'bar', name: 'no_merge',
         marker: {color: 'rgba(229,57,53,0.8)', line: {width: 0}},
         hoverinfo: 'y', showlegend: false,
         customdata: Array.from({length: h.centers.length}, function(_, i) { return i; })},
    ];

    var shapes = [
        {type: 'rect', x0: 0, x1: 0, y0: 0, y1: 1, yref: 'paper',
         fillcolor: 'rgba(255,200,0,0.15)', line: {color: 'rgba(255,160,0,0.6)', width: 2},
         visible: false}
    ];

    // Use log scale with fixed range from unfiltered data
    var logMax = yMax > 0 ? Math.ceil(Math.log10(yMax * 1.2)) : 1;

    var layout = {
        xaxis: {title: title, range: [0, 1]},
        yaxis: {title: 'Count', type: 'log', dtick: 1, range: [0, logMax]},
        barmode: 'stack',
        bargap: 0.05,
        margin: {l: 50, r: 20, t: 10, b: 40},
        showlegend: false,
        shapes: shapes,
    };

    Plotly.newPlot(divId, traces, layout, {responsive: true});

    document.getElementById(divId).on('plotly_click', function(data) {
        var pt = data.points[0];
        var binIdx = pt.customdata;
        if (binIdx === undefined) {
            var xVal = pt.x;
            var closest = 0, closestDist = Infinity;
            for (var k = 0; k < h.centers.length; k++) {
                var d = Math.abs(h.centers[k] - xVal);
                if (d < closestDist) { closestDist = d; closest = k; }
            }
            binIdx = closest;
        }

        // Highlight bin
        Plotly.relayout(divId, {
            'shapes[0].x0': h.edges[binIdx],
            'shapes[0].x1': h.edges[binIdx + 1],
            'shapes[0].visible': true
        });

        var lo = h.edges[binIdx].toFixed(2);
        var hi = h.edges[binIdx + 1].toFixed(2);
        renderSamplePanel(
            document.getElementById(panelId),
            title + ' [' + lo + ', ' + hi + ')',
            h.binIndices[binIdx]
        );
    });
}

function updateHistograms() {
    var hSS = computeHistogram('ss', curveMask, HIST_NBINS);
    var hMA = computeHistogram('ma', curveMask, HIST_NBINS);

    Plotly.restyle('hist-ss', {y: [hSS.merge, hSS.noMerge]}, [0, 1]);
    Plotly.restyle('hist-ma', {y: [hMA.merge, hMA.noMerge]}, [0, 1]);
}

function computeAllMask() {
    var mask = new Uint8Array(nContacts);
    for (var i = 0; i < nContacts; i++) mask[i] = 1;
    return mask;
}

// --- Main update ---
function updateAll() {
    currentMask = computeFilterMask();
    curveMask = (filterMode === "unfiltered") ? computeAllMask() : currentMask;
    var nFiltered = 0, nShown = 0;
    for (var i = 0; i < nContacts; i++) { nFiltered += currentMask[i]; nShown += curveMask[i]; }
    var summaryText = nShown + ' / ' + nContacts + ' contacts shown';
    if (filterMode === "unfiltered" && nFiltered !== nContacts) {
        summaryText += ' (filters: ' + nFiltered + ')';
    }
    document.getElementById('summary').textContent = summaryText;

    updateScatter();
    updateHistograms();
    updateCurves();
    updateConfusionMatrix();
}

// --- Initialization ---
function init() {
    document.getElementById('progress-text').textContent = 'Initializing...';
    var data = DATA_JSON;

    contacts = data.contacts;
    datasets = data.datasets;
    ngInfos = data.ngInfos;
    authority = data.authority;
    nContacts = contacts.length;

    loadAnnotations();
    STORAGE_KEY = "validation_results_annotations_" + authority;
    loadAnnotations();

    initContactOrder();

    // Build dataset filter table with common prefix detection
    var commonPrefix = datasets[0] || '';
    for (var i = 1; i < datasets.length; i++) {
        while (commonPrefix && datasets[i].indexOf(commonPrefix) !== 0) {
            commonPrefix = commonPrefix.slice(0, -1);
        }
    }
    // Also find common suffix
    var rev = function(s) { return s.split('').reverse().join(''); };
    var commonSuffix = rev(datasets[0] || '');
    for (var i = 1; i < datasets.length; i++) {
        var r = rev(datasets[i]);
        while (commonSuffix && r.indexOf(commonSuffix) !== 0) {
            commonSuffix = commonSuffix.slice(0, -1);
        }
    }
    commonSuffix = rev(commonSuffix);
    var prefLen = commonPrefix.length;
    var sufLen = commonSuffix.length;

    var filterHtml = '<div class="dset-filter-list">';
    filterHtml += '<label><input type="checkbox" id="dset-all" checked> <b>All datasets</b></label>';
    for (var i = 0; i < datasets.length; i++) {
        datasetFilters[datasets[i]] = true;
        var name = datasets[i];
        var varying = name.slice(prefLen, sufLen ? name.length - sufLen : name.length);
        var displayName = '<span style="color:#999">' + commonPrefix + '</span><b>' + varying + '</b><span style="color:#999">' + commonSuffix + '</span>';
        filterHtml += '<label><input type="checkbox" class="dset-cb" data-dset="' +
            name + '" checked> ' + displayName + '</label>';
    }
    filterHtml += '</div>';
    document.getElementById('dataset-filters').innerHTML = filterHtml;

    // Wire dataset checkboxes
    document.getElementById('dset-all').addEventListener('change', function() {
        var checked = this.checked;
        document.querySelectorAll('.dset-cb').forEach(function(cb) {
            cb.checked = checked;
            datasetFilters[cb.dataset.dset] = checked;
        });
        updateAll();
    });
    document.querySelectorAll('.dset-cb').forEach(function(cb) {
        cb.addEventListener('change', function() {
            datasetFilters[this.dataset.dset] = this.checked;
            var allChecked = true;
            document.querySelectorAll('.dset-cb').forEach(function(c) {
                if (!c.checked) allChecked = false;
            });
            document.getElementById('dset-all').checked = allChecked;
            updateAll();
        });
        // Hover preview: use hoverDataset override (doesn't touch datasetFilters)
        var label = cb.closest('label');
        label.addEventListener('mouseenter', function() {
            hoverDataset = cb.dataset.dset;
            updateAll();
        });
        label.addEventListener('mouseleave', function() {
            hoverDataset = null;
            updateAll();
        });
    });
    // Hover on "All datasets" label
    var allLabel = document.getElementById('dset-all').closest('label');
    allLabel.addEventListener('mouseenter', function() {
        hoverDataset = "__all__";
        updateAll();
    });
    allLabel.addEventListener('mouseleave', function() {
        hoverDataset = null;
        updateAll();
    });

    // Wire slider+number input pairs (bidirectional sync)
    function wireSliderNum(sliderId, numId, setter) {
        var slider = document.getElementById(sliderId);
        var numIn = document.getElementById(numId);
        slider.addEventListener('input', function() {
            var v = parseFloat(this.value);
            numIn.value = v.toFixed(2);
            setter(v);
            updateAll();
        });
        numIn.addEventListener('change', function() {
            var v = parseFloat(this.value);
            if (isNaN(v)) return;
            v = Math.max(parseFloat(slider.min), Math.min(parseFloat(slider.max), v));
            this.value = v.toFixed(2);
            slider.value = v;
            setter(v);
            updateAll();
        });
    }
    wireSliderNum('ma-lo', 'ma-lo-val', function(v) { maLo = v; });
    wireSliderNum('ma-hi', 'ma-hi-val', function(v) { maHi = v; });
    wireSliderNum('ss-lo', 'ss-lo-val', function(v) { ssLo = v; });
    wireSliderNum('ss-hi', 'ss-hi-val', function(v) { ssHi = v; });

    // Wire CM threshold (slider + number)
    wireSliderNum('cm-threshold', 'cm-thr-val', function(v) { cmThreshold = v; });

    // Wire CM cell clicks (shape)
    document.querySelectorAll('#cm-grid .cm-cell').forEach(function(el) {
        el.addEventListener('click', function() {
            showCmSamples(this.dataset.cat);
        });
    });

    // Wire affinity CM threshold (slider + number)
    wireSliderNum('ma-cm-threshold', 'ma-cm-thr-val', function(v) { maThreshold = v; });

    // Wire affinity CM cell clicks
    document.querySelectorAll('#ma-cm-grid .cm-cell').forEach(function(el) {
        el.addEventListener('click', function() {
            showMaCmSamples(this.dataset.cat);
        });
    });

    // Wire filter mode radio
    document.querySelectorAll('input[name="filter_mode"]').forEach(function(radio) {
        radio.addEventListener('change', function() {
            filterMode = this.value;
            updateAll();
        });
    });

    // Wire annotation toolbar
    document.getElementById('ann_export').addEventListener('click', exportAnnotations);
    document.getElementById('ann_import').addEventListener('click', function() {
        document.getElementById('ann_file_input').click();
    });
    document.getElementById('ann_file_input').addEventListener('change', function() {
        if (this.files.length > 0) importAnnotations(this.files[0]);
        this.value = '';
    });

    // Hide progress, show content
    document.getElementById('progress-wrap').style.display = 'none';
    document.getElementById('main-content').style.display = '';

    // Initial render
    currentMask = computeFilterMask();
    curveMask = (filterMode === "unfiltered") ? computeAllMask() : currentMask;
    buildScatter();
    buildHistogram('hist-ss', 'hist-ss-panel', 'ss', 'Shape Score');
    buildHistogram('hist-ma', 'hist-ma-panel', 'ma', 'Mean Affinity');
    updateAll();
}

// --- Neuroglancer popup viewer (reuses same window) ---
var ngPopup = null;
var ngPopupCheckTimer = null;
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

function openNgPopup(url) {
    if (!ngPopup || ngPopup.closed) {
        var g = ngPopupGeom || {
            w: window.outerWidth, h: window.outerHeight,
            x: window.screenX, y: window.screenY
        };
        ngPopup = window.open(url, "ng_viewer",
            "width=" + g.w + ",height=" + g.h +
            ",left=" + g.x + ",top=" + g.y +
            ",menubar=no,toolbar=no,location=yes,status=no,resizable=yes,scrollbars=yes");
    } else {
        savePopupGeom();
        ngPopup.location.href = url;
        ngPopup.focus();
    }
    startPopupCloseCheck();
}

document.addEventListener("click", function(e) {
    var link = e.target.closest(".ngl-link");
    if (link) {
        e.preventDefault();
        clearNglHighlight();
        var row = link.closest("tr");
        if (row) row.classList.add("ngl-active");
        openNgPopup(link.href);
    }
});

var DATA_JSON = $data_json;
init();
</script>
</body>
</html>
""")

if __name__ == "__main__":
    main()
