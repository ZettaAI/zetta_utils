"""
Compare two ribbon-assignment-eval runs (e.g. network vs nearest-neighbour
baseline) on the same ribbon set, and produce a single NG state with the
TP/FP/FN annotations split by which approach got/missed each assignment:

    TP (both)   — assignments both approaches got right
    TP (Net)    — only the network got it
    TP (NN)     — only the nearest-neighbour baseline got it
    FP (both)   — both approaches made the same wrong assignment
    FP (Net)    — only the network made this wrong assignment
    FP (NN)     — only NN made this wrong assignment
    FN (both)   — neither approach predicted this GT partner
    FN (Net)    — only the network missed it (NN got it)
    FN (NN)     — only NN missed it (the network got it)

Inputs: two `eval_multi_partner.py` output dirs containing tp.json /
fp.json / fn.json each. Annotation IDs of the form `<kind>_r<syn_id>_p<post_id>`
provide the (ribbon, partner) keys used to compute set differences.
"""
from __future__ import annotations

import argparse
import json
import os
import re

from eval_synapses import _local_annotation_layer
from zetta_utils.geometry import Vec3D


_KEY_RE = re.compile(r"r(?P<syn>\d+)_p(?P<post>\d+)$")


def _extract_key(ann: dict) -> tuple[int, int] | None:
    m = _KEY_RE.search(ann.get("id", ""))
    if not m:
        return None
    return (int(m["syn"]), int(m["post"]))


def _load(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _by_key(anns: list[dict]) -> dict[tuple[int, int], dict]:
    out: dict[tuple[int, int], dict] = {}
    for a in anns:
        k = _extract_key(a)
        if k is not None:
            out[k] = a
    return out


def _re_id(ann: dict, prefix: str) -> dict:
    """Return a copy of `ann` with its `id` reprefixed (so the 9 layers don't
    collide on duplicate IDs)."""
    new_id = re.sub(r"^[a-zA-Z_]+_r", f"{prefix}_r", ann["id"], count=1)
    return {**ann, "id": new_id}


def build_split_layers(
    net_dir: str,
    nn_dir: str,
) -> dict[str, list[dict]]:
    net_tp = _by_key(_load(os.path.join(net_dir, "tp.json")))
    net_fp = _by_key(_load(os.path.join(net_dir, "fp.json")))
    net_fn = _by_key(_load(os.path.join(net_dir, "fn.json")))
    nn_tp = _by_key(_load(os.path.join(nn_dir, "tp.json")))
    nn_fp = _by_key(_load(os.path.join(nn_dir, "fp.json")))
    nn_fn = _by_key(_load(os.path.join(nn_dir, "fn.json")))

    def split(net: dict, nn: dict, prefix: str):
        net_keys, nn_keys = set(net), set(nn)
        both = net_keys & nn_keys
        only_net = net_keys - nn_keys
        only_nn = nn_keys - net_keys
        return (
            [_re_id(net[k], f"{prefix}both") for k in sorted(both)],
            [_re_id(net[k], f"{prefix}net") for k in sorted(only_net)],
            [_re_id(nn[k], f"{prefix}nn") for k in sorted(only_nn)],
        )

    tp_both, tp_net, tp_nn = split(net_tp, nn_tp, "tp_")
    fp_both, fp_net, fp_nn = split(net_fp, nn_fp, "fp_")
    fn_both, fn_net, fn_nn = split(net_fn, nn_fn, "fn_")

    return {
        "tp_both": tp_both,
        "tp_net": tp_net,
        "tp_nn": tp_nn,
        "fp_both": fp_both,
        "fp_net": fp_net,
        "fp_nn": fp_nn,
        "fn_both": fn_both,
        "fn_net": fn_net,
        "fn_nn": fn_nn,
    }


# Layer styling: same kind shares a hue, "both" is brightest.
_LAYER_STYLE: list[tuple[str, str, str]] = [
    ("TP (both)", "tp_both", "#00ff00"),
    ("TP (Net)", "tp_net", "#ffff00"),
    ("TP (NN)", "tp_nn", "#80ff00"),
    ("FP (both)", "fp_both", "#ff0000"),
    ("FP (Net)", "fp_net", "#ff00ff"),
    ("FP (NN)", "fp_nn", "#ff8800"),
    ("FN (both)", "fn_both", "#0000ff"),
    ("FN (Net)", "fn_net", "#00ffff"),
    ("FN (NN)", "fn_nn", "#8800ff"),
]


def build_ng_state(
    resolution: Vec3D,
    position: list[float],
    image_path: str,
    pcg_source: str,
    ribbon_seg_path: str,
    split: dict[str, list[dict]],
) -> dict:
    dims = {
        "x": [resolution[0] * 1e-9, "m"],
        "y": [resolution[1] * 1e-9, "m"],
        "z": [resolution[2] * 1e-9, "m"],
    }
    layers: list[dict] = [
        {"type": "image", "source": image_path + "/|neuroglancer-precomputed:", "name": "EM"},
        {
            "type": "segmentation",
            "source": pcg_source,
            "selectedAlpha": 0.15,
            "segments": [],
            "name": "neuron segmentation",
        },
        {
            "type": "segmentation",
            "source": ribbon_seg_path + "/|neuroglancer-precomputed:",
            "segments": [],
            "name": "ribbon segmentation",
        },
    ]
    for name, key, color in _LAYER_STYLE:
        anns = split.get(key, [])
        layers.append(_local_annotation_layer(name, anns, resolution, color=color))

    return {
        "dimensions": dims,
        "position": position,
        "crossSectionScale": 0.18,
        "projectionScale": 220,
        "layers": layers,
        "showSlices": False,
        "layout": "4panel-alt",
    }


def parse_args():
    p = argparse.ArgumentParser(description="Compare two ribbon-assignment evaluation runs.")
    p.add_argument("--net-dir", required=True, help="Output dir from the network run.")
    p.add_argument(
        "--nn-dir", required=True, help="Output dir from the nearest-neighbour baseline run."
    )
    p.add_argument(
        "--bbox-start",
        type=int,
        nargs=3,
        required=True,
        help="Bbox start in voxels (used to center the NG state).",
    )
    p.add_argument("--bbox-end", type=int, nargs=3, required=True)
    p.add_argument("--resolution", type=float, nargs=3, default=[16, 16, 40])
    p.add_argument("--output-dir", default=".")
    p.add_argument("--upload-state", action="store_true")
    p.add_argument("--image-path", default="gs://stroeh_sem_mouse_retina/image/v2")
    p.add_argument(
        "--pcg-source",
        default="graphene://middleauth+https://minnie.microns-daf.com/segmentation/table/stroeh_mouse_retina",
    )
    p.add_argument(
        "--ribbon-seg-path", required=True, help="Precomputed ribbon-segmentation layer path."
    )
    p.add_argument("--cave-server", default="https://global.daf-apis.com")
    p.add_argument("--cave-token", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    resolution: Vec3D = Vec3D(*args.resolution)

    print(f"Loading net JSONs from {args.net_dir}")
    print(f"Loading NN  JSONs from {args.nn_dir}")
    split = build_split_layers(args.net_dir, args.nn_dir)

    print("\nSplit counts:")
    print(f"{'layer':<12} {'count':>6}")
    print("-" * 22)
    total = 0
    for name, key, _ in _LAYER_STYLE:
        n = len(split[key])
        total += n
        print(f"{name:<12} {n:>6}")
    print(f"{'total':<12} {total:>6}")

    # Quick derived metrics
    net_tp = len(split["tp_both"]) + len(split["tp_net"])
    net_fp = len(split["fp_both"]) + len(split["fp_net"])
    net_fn = len(split["fn_both"]) + len(split["fn_net"])
    nn_tp = len(split["tp_both"]) + len(split["tp_nn"])
    nn_fp = len(split["fp_both"]) + len(split["fp_nn"])
    nn_fn = len(split["fn_both"]) + len(split["fn_nn"])

    def f1(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    p_n, r_n, f_n = f1(net_tp, net_fp, net_fn)
    p_b, r_b, f_b = f1(nn_tp, nn_fp, nn_fn)
    print(f"\nNet:   TP={net_tp} FP={net_fp} FN={net_fn}  P={p_n:.3f} R={r_n:.3f} F1={f_n:.3f}")
    print(f"NN:    TP={nn_tp} FP={nn_fp} FN={nn_fn}  P={p_b:.3f} R={r_b:.3f} F1={f_b:.3f}")
    print(
        f"\nNet's marginal lift over NN (TP only Net - FP only Net): "
        f"{len(split['tp_net'])} - {len(split['fp_net'])} = "
        f"{len(split['tp_net']) - len(split['fp_net'])}"
    )
    print(
        f"NN's  marginal lift over Net (TP only NN  - FP only NN ): "
        f"{len(split['tp_nn'])} - {len(split['fp_nn'])} = "
        f"{len(split['tp_nn']) - len(split['fp_nn'])}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    for _, key, _ in _LAYER_STYLE:
        path = os.path.join(args.output_dir, f"{key}.json")
        with open(path, "w") as f:
            json.dump(split[key], f, indent=2)
    print(f"\nWrote split JSONs to {args.output_dir}")

    if args.upload_state:
        from caveclient import CAVEclient

        cx = (args.bbox_start[0] + args.bbox_end[0]) / 2
        cy = (args.bbox_start[1] + args.bbox_end[1]) / 2
        cz = (args.bbox_start[2] + args.bbox_end[2]) / 2
        position = [cx, cy, cz + 0.5]
        ng_state = build_ng_state(
            resolution=resolution,
            position=position,
            image_path=args.image_path,
            pcg_source=args.pcg_source,
            ribbon_seg_path=args.ribbon_seg_path,
            split=split,
        )
        state_path = os.path.join(args.output_dir, "ng_state.json")
        with open(state_path, "w") as f:
            json.dump(ng_state, f, indent=2)
        print(f"  Saved NG state to {state_path}")

        token = args.cave_token or os.environ.get("CAVE_TOKEN")
        cave_kwargs = {"server_address": args.cave_server}
        if token:
            cave_kwargs["auth_token"] = token
        client = CAVEclient(**cave_kwargs)
        state_id = client.state.upload_state_json(ng_state)
        print(
            f"  NG link: https://spelunker.cave-explorer.org/#!middleauth+"
            f"{args.cave_server}/nglstate/api/v1/{state_id}"
        )


if __name__ == "__main__":
    main()
