from __future__ import annotations

import html
import random
from collections.abc import Sequence
from datetime import datetime

import attrs
import fsspec
import numpy as np

from zetta_utils import builder, log, mazepa
from zetta_utils.internal.regimes.contact_merge import _build_ng_url
from zetta_utils.layer.volumetric.seg_contact.backend import SegContactLayerBackend
from zetta_utils.layer.volumetric.seg_contact.contact import SegContact
from zetta_utils.mazepa import taskable_operation_cls

logger = log.get_logger("zetta_utils")


def _iter_all_chunk_indices(backend: SegContactLayerBackend):
    """Iterate over all chunk grid indices for a backend."""
    import itertools

    nx = int(np.ceil(backend.size[0] / backend.chunk_size[0]))
    ny = int(np.ceil(backend.size[1] / backend.chunk_size[1]))
    nz = int(np.ceil(backend.size[2] / backend.chunk_size[2]))
    return itertools.product(range(nx), range(ny), range(nz))


def _contact_key(seg_a: int, seg_b: int) -> tuple[int, int]:
    return (min(seg_a, seg_b), max(seg_a, seg_b))


@builder.register("SegContactCompareChunkOp")
@taskable_operation_cls
@attrs.frozen
class SegContactCompareChunkOp:
    """Compare contacts from two backends for a single chunk.

    Returns a dict with per-chunk stats and contact keys for discrepancy detection.
    """

    merge_authority: str | None = "ground_truth"

    def __call__(
        self,
        path_a: str,
        path_b: str,
        chunk_idx: tuple[int, int, int],
    ) -> dict:
        backend_a = SegContactLayerBackend.from_path(path_a)
        backend_b = SegContactLayerBackend.from_path(path_b)

        contacts_a = backend_a._read_contacts_chunk(chunk_idx)
        contacts_b = backend_b._read_contacts_chunk(chunk_idx)

        def _summarize(contacts: list[dict], backend: SegContactLayerBackend) -> dict:
            n_contacts = 0
            face_sum = 0
            face_sum_sq = 0
            face_min = float("inf")
            face_max = 0
            seg_ids = set()
            keys = set()
            for c in contacts:
                faces = c["contact_faces"]
                n = faces.shape[0] if faces is not None else 0
                n_contacts += 1
                face_sum += n
                face_sum_sq += n * n
                if n < face_min:
                    face_min = n
                if n > face_max:
                    face_max = n
                seg_ids.add(int(c["seg_a"]))
                seg_ids.add(int(c["seg_b"]))
                keys.add(_contact_key(int(c["seg_a"]), int(c["seg_b"])))

            decisions_merge = 0
            decisions_split = 0
            if self.merge_authority is not None:
                info = backend.read_info()
                if self.merge_authority in info.get("merge_decisions", []):
                    for _cid, should_merge in backend._read_merge_decision_chunk(
                        chunk_idx, self.merge_authority
                    ):
                        if should_merge:
                            decisions_merge += 1
                        else:
                            decisions_split += 1

            return {
                "n_contacts": n_contacts,
                "face_sum": face_sum,
                "face_sum_sq": face_sum_sq,
                "face_min": face_min if n_contacts > 0 else 0,
                "face_max": face_max,
                "seg_ids": list(seg_ids),
                "keys": list(keys),
                "gt_merge": decisions_merge,
                "gt_split": decisions_split,
            }

        return {
            "chunk_idx": list(chunk_idx),
            "a": _summarize(contacts_a, backend_a),
            "b": _summarize(contacts_b, backend_b),
        }


@attrs.mutable
class _RunningStats:
    """Accumulates stats incrementally without storing per-contact values."""

    n_contacts: int = 0
    face_sum: int = 0
    face_sum_sq: int = 0
    face_min: int = 0
    face_max: int = 0
    n_seg_ids: int = 0
    gt_merge: int = 0
    gt_split: int = 0

    def to_dict(self) -> dict:
        n = self.n_contacts
        mean = self.face_sum / n if n > 0 else 0
        var = (self.face_sum_sq / n - mean * mean) if n > 0 else 0
        std = var**0.5 if var > 0 else 0
        n_merge = self.gt_merge
        n_split = self.gt_split
        return {
            "total_contacts": n,
            "unique_segments": self.n_seg_ids,
            "faces_per_contact_mean": mean,
            "faces_per_contact_std": std,
            "faces_per_contact_min": self.face_min,
            "faces_per_contact_max": self.face_max,
            "gt_merge": n_merge,
            "gt_split": n_split,
            "gt_merge_fraction": (
                n_merge / (n_merge + n_split) if (n_merge + n_split) > 0 else 0
            ),
        }


def _aggregate_chunk_results(
    tasks: list,
) -> tuple[dict, dict, dict, dict, set, set, set]:
    """Aggregate task outcomes incrementally, freeing each as it's processed.

    Returns (stats_a, stats_b, keys_a_dict, keys_b_dict, only_a, only_b, shared).
    keys_a_dict/keys_b_dict map (seg_a, seg_b) -> chunk_idx (first occurrence only).
    """
    rs_a = _RunningStats()
    rs_b = _RunningStats()
    seg_ids_a: set[int] = set()
    seg_ids_b: set[int] = set()
    # Only store first chunk_idx per key (enough for re-reading examples)
    keys_a: dict[tuple[int, int], list[int]] = {}
    keys_b: dict[tuple[int, int], list[int]] = {}

    for task in tasks:
        r = task.outcome.return_value
        chunk_idx = r["chunk_idx"]
        a, b = r["a"], r["b"]

        for side, rs, seg_ids, keys_dict in [
            (a, rs_a, seg_ids_a, keys_a),
            (b, rs_b, seg_ids_b, keys_b),
        ]:
            rs.n_contacts += side["n_contacts"]
            rs.face_sum += side["face_sum"]
            rs.face_sum_sq += side["face_sum_sq"]
            if side["n_contacts"] > 0:
                if rs.n_contacts == side["n_contacts"]:
                    rs.face_min = side["face_min"]
                else:
                    rs.face_min = min(rs.face_min, side["face_min"])
                rs.face_max = max(rs.face_max, side["face_max"])
            seg_ids.update(side["seg_ids"])
            rs.gt_merge += side["gt_merge"]
            rs.gt_split += side["gt_split"]

            for k in side["keys"]:
                k_tuple = tuple(k)
                if k_tuple not in keys_dict:
                    keys_dict[k_tuple] = chunk_idx

        # Free task outcome to release memory
        task.outcome = None

    rs_a.n_seg_ids = len(seg_ids_a)
    rs_b.n_seg_ids = len(seg_ids_b)
    del seg_ids_a, seg_ids_b

    set_a = set(keys_a.keys())
    set_b = set(keys_b.keys())
    only_a = set_a - set_b
    only_b = set_b - set_a
    shared = set_a & set_b

    return rs_a.to_dict(), rs_b.to_dict(), keys_a, keys_b, only_a, only_b, shared


def _print_stats(name: str, stats: dict):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Total contacts:          {stats['total_contacts']}")
    print(f"  Unique segments:         {stats['unique_segments']}")
    print(
        f"  Faces/contact:           {stats['faces_per_contact_mean']:.1f} "
        f"+/- {stats['faces_per_contact_std']:.1f}  "
        f"(min={stats['faces_per_contact_min']}, "
        f"max={stats['faces_per_contact_max']})"
    )
    if stats["gt_merge"] + stats["gt_split"] > 0:
        print(
            f"  GT merge/split:          {stats['gt_merge']} / {stats['gt_split']}  "
            f"({stats['gt_merge_fraction']:.1%} merge)"
        )


def _print_comparison(name_a: str, stats_a: dict, name_b: str, stats_b: dict):
    all_stats = [(name_a, stats_a), (name_b, stats_b)]
    names = [name for name, _ in all_stats]
    max_name_len = max(len(n) for n in names)

    print(f"\n{'=' * 60}")
    print("  COMPARISON")
    print(f"{'=' * 60}")

    has_gt = any(s["gt_merge"] + s["gt_split"] > 0 for _, s in all_stats)
    compare_keys = [
        ("total_contacts", "Total contacts"),
        ("faces_per_contact_mean", "Faces/contact (mean)"),
        ("unique_segments", "Unique segments"),
    ]
    if has_gt:
        compare_keys += [
            ("gt_merge", "GT merge"),
            ("gt_split", "GT split"),
            ("gt_merge_fraction", "GT merge fraction"),
        ]

    header = f"{'Metric':<30}"
    for name in names:
        header += f"  {name:>{max_name_len}}"
    print(header)
    print("-" * len(header))

    for key, label in compare_keys:
        row = f"{label:<30}"
        for _, stats in all_stats:
            val = stats.get(key, 0)
            if isinstance(val, float):
                row += f"  {val:>{max_name_len}.2f}"
            else:
                row += f"  {val:>{max_name_len}}"
        print(row)

    print(f"\nDiff ({name_b} vs {name_a}):")
    for key, label in compare_keys:
        bv = stats_a.get(key, 0)
        ov = stats_b.get(key, 0)
        if isinstance(bv, (int, float)) and bv != 0:
            diff = ov - bv
            pct = (ov / bv - 1) * 100
            print(f"  {label:<30}  {diff:+.2f}  ({pct:+.1f}%)")


def _load_example_contacts(
    path: str,
    keys_with_chunks: dict[tuple[int, int], list[int]],
    sampled_keys: list[tuple[int, int]],
) -> dict[tuple[int, int], SegContact]:
    """Re-read full contacts for a small set of sampled keys (for NG links)."""
    backend = SegContactLayerBackend.from_path(path)
    result: dict[tuple[int, int], SegContact] = {}

    chunks_needed: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    for key in sampled_keys:
        chunk_idx = tuple(keys_with_chunks[key])
        chunks_needed.setdefault(chunk_idx, []).append(key)

    for chunk_idx, needed_keys in chunks_needed.items():
        needed_set = set(needed_keys)
        for c in backend._read_contacts_chunk(chunk_idx):
            k = _contact_key(int(c["seg_a"]), int(c["seg_b"]))
            if k in needed_set and k not in result:
                result[k] = SegContact(
                    id=c["id"],
                    seg_a=c["seg_a"],
                    seg_b=c["seg_b"],
                    com=c["com"],
                    contact_faces=c["contact_faces"],
                    partner_metadata=c["partner_metadata"],
                    representative_points=c["representative_points"],
                )
    return result


def _build_html_report(
    name_a: str,
    name_b: str,
    stats_a: dict,
    stats_b: dict,
    only_a: set[tuple[int, int]],
    only_b: set[tuple[int, int]],
    shared: set[tuple[int, int]],
    keys_a: dict[tuple[int, int], list[int]],
    keys_b: dict[tuple[int, int], list[int]],
    examples_a: dict[tuple[int, int], SegContact],
    examples_b: dict[tuple[int, int], SegContact],
    info_path_a: str,
    info_path_b: str,
) -> str:
    """Build an HTML report with stats, discrepancies, and NG links."""

    def _stats_table(sa: dict, sb: dict) -> str:
        rows = ""
        keys = [
            ("total_contacts", "Total contacts", "d"),
            ("unique_segments", "Unique segments", "d"),
            ("faces_per_contact_mean", "Faces/contact (mean)", ".1f"),
            ("faces_per_contact_std", "Faces/contact (std)", ".1f"),
            ("gt_merge", "GT merge", "d"),
            ("gt_split", "GT split", "d"),
            ("gt_merge_fraction", "GT merge fraction", ".1%"),
        ]
        for key, label, fmt in keys:
            va = sa.get(key, 0)
            vb = sb.get(key, 0)
            diff = vb - va
            pct = (vb / va - 1) * 100 if va != 0 else 0
            rows += (
                f"<tr><td>{html.escape(label)}</td>"
                f"<td>{va:{fmt}}</td>"
                f"<td>{vb:{fmt}}</td>"
                f"<td>{diff:+{fmt}}</td>"
                f"<td>{pct:+.1f}%</td></tr>\n"
            )
        return rows

    def _contact_rows(
        examples: dict[tuple[int, int], SegContact],
        info_path: str,
    ) -> str:
        if not examples:
            return "<tr><td colspan='5'>None</td></tr>"
        rows = ""
        for key, contact in examples.items():
            com = (contact.com[0], contact.com[1], contact.com[2])
            n_faces = (
                contact.contact_faces.shape[0] if contact.contact_faces is not None else 0
            )
            url = _build_ng_url(
                info_path,
                com,
                contact.seg_a,
                contact.seg_b,
                contact.contact_faces,
                n_faces if contact.contact_faces is not None else None,
            )
            rows += (
                f"<tr>"
                f"<td>{contact.seg_a}</td>"
                f"<td>{contact.seg_b}</td>"
                f"<td>{n_faces}</td>"
                f"<td>({com[0]:.0f}, {com[1]:.0f}, {com[2]:.0f})</td>"
                f"<td><a href='{html.escape(url)}' target='_blank'>Open in NG</a></td>"
                f"</tr>\n"
            )
        return rows

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_examples_a = len(examples_a)
    n_examples_b = len(examples_b)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Seg Contact Comparison: {html.escape(name_a)} vs {html.escape(name_b)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; }}
  h1 {{ font-size: 1.4em; }}
  h2 {{ font-size: 1.2em; margin-top: 2em; }}
  table {{ border-collapse: collapse; margin: 10px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 12px; text-align: right; }}
  th {{ background: #f5f5f5; }}
  td:first-child, th:first-child {{ text-align: left; }}
  .summary {{ background: #f0f8ff; padding: 10px; border-radius: 5px; margin: 10px 0; }}
  a {{ color: #0066cc; }}
</style>
</head>
<body>
<h1>Seg Contact Comparison</h1>
<p>Generated: {timestamp}</p>
<div class="summary">
  <b>A:</b> {html.escape(name_a)}<br>
  <b>B:</b> {html.escape(name_b)}
</div>

<h2>Summary Statistics</h2>
<table>
<tr><th>Metric</th><th>{html.escape(name_a)}</th><th>{html.escape(name_b)}</th><th>Diff</th><th>%</th></tr>
{_stats_table(stats_a, stats_b)}
</table>

<h2>Discrepancies (by segment pair)</h2>
<div class="summary">
  Unique pairs in A: {len(keys_a)} | Unique pairs in B: {len(keys_b)}<br>
  Only in A: <b>{len(only_a)}</b> | Only in B: <b>{len(only_b)}</b> | Shared: {len(shared)}
</div>

<h2>Only in {html.escape(name_a)} ({len(only_a)} total, showing {n_examples_a})</h2>
<table>
<tr><th>seg_a</th><th>seg_b</th><th>n_faces</th><th>COM (nm)</th><th>Link</th></tr>
{_contact_rows(examples_a, info_path_a)}
</table>

<h2>Only in {html.escape(name_b)} ({len(only_b)} total, showing {n_examples_b})</h2>
<table>
<tr><th>seg_a</th><th>seg_b</th><th>n_faces</th><th>COM (nm)</th><th>Link</th></tr>
{_contact_rows(examples_b, info_path_b)}
</table>

</body>
</html>"""


@builder.register("build_seg_contact_compare_flow")
def build_seg_contact_compare_flow(
    paths: Sequence[str],
    output_path: str,
    op: SegContactCompareChunkOp | None = None,
    show_discrepancies: bool = True,
    n_discrepancy_examples: int = 50,
) -> mazepa.Flow:
    schema = SegContactCompareFlowSchema(
        op=op or SegContactCompareChunkOp(),
        show_discrepancies=show_discrepancies,
        n_discrepancy_examples=n_discrepancy_examples,
    )
    return schema(paths=paths, output_path=output_path)


@mazepa.flow_schema_cls
@attrs.mutable
class SegContactCompareFlowSchema:
    """Flow that compares two seg_contact datasets chunk-by-chunk in parallel."""

    op: SegContactCompareChunkOp = attrs.Factory(SegContactCompareChunkOp)
    show_discrepancies: bool = True
    n_discrepancy_examples: int = 50

    def flow(
        self,
        paths: Sequence[str],
        output_path: str,
    ) -> mazepa.FlowFnReturnType:
        assert len(paths) == 2, "Exactly 2 paths required for comparison"

        path_a, path_b = paths[0], paths[1]
        backend_a = SegContactLayerBackend.from_path(path_a)
        backend_b = SegContactLayerBackend.from_path(path_b)

        assert backend_a.chunk_size == backend_b.chunk_size, (
            f"Chunk sizes must match: {backend_a.chunk_size} != {backend_b.chunk_size}"
        )
        assert backend_a.size == backend_b.size, (
            f"Sizes must match: {backend_a.size} != {backend_b.size}"
        )

        chunk_indices = list(_iter_all_chunk_indices(backend_a))
        logger.info(f"Comparing {len(chunk_indices)} chunks between datasets")

        # Phase 1: process all chunks in parallel
        tasks = []
        for chunk_idx in chunk_indices:
            task = self.op.make_task(
                path_a=path_a,
                path_b=path_b,
                chunk_idx=chunk_idx,
            )
            tasks.append(task)

        yield tasks
        yield mazepa.Dependency()

        # Phase 2: aggregate results (processes and frees outcomes incrementally)
        for task in tasks:
            if task.outcome is None:
                return
            if task.outcome.exception is not None:
                raise task.outcome.exception

        stats_a, stats_b, keys_a, keys_b, only_a, only_b, shared = (
            _aggregate_chunk_results(tasks)
        )

        name_a = path_a.rstrip("/").split("/")[-1]
        name_b = path_b.rstrip("/").split("/")[-1]

        _print_stats(name_a, stats_a)
        _print_stats(name_b, stats_b)
        _print_comparison(name_a, stats_a, name_b, stats_b)

        if self.show_discrepancies:
            logger.info(
                f"Discrepancies: {len(only_a)} only in A, {len(only_b)} only in B, "
                f"{len(shared)} shared"
            )

            # Sample and re-read full contacts for examples
            sampled_a = random.sample(
                sorted(only_a), min(self.n_discrepancy_examples, len(only_a))
            )
            sampled_b = random.sample(
                sorted(only_b), min(self.n_discrepancy_examples, len(only_b))
            )

            examples_a = _load_example_contacts(path_a, keys_a, sampled_a)
            examples_b = _load_example_contacts(path_b, keys_b, sampled_b)

            report_html = _build_html_report(
                name_a,
                name_b,
                stats_a,
                stats_b,
                only_a,
                only_b,
                shared,
                keys_a,
                keys_b,
                examples_a,
                examples_b,
                f"{path_a}/info",
                f"{path_b}/info",
            )

            fs, fs_path = fsspec.core.url_to_fs(output_path)
            fs.makedirs(fs.sep.join(fs_path.split(fs.sep)[:-1]), exist_ok=True)
            with fs.open(fs_path, "w") as f:
                f.write(report_html)
            logger.info(f"Report written to: {output_path}")
