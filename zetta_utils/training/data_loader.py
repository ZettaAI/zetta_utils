import queue
import threading
from collections.abc import Iterable, Iterator

import torch

from zetta_utils import builder

builder.register("TorchDataLoader")(torch.utils.data.DataLoader)


@builder.register("RebatchingDataLoader")
class RebatchingDataLoader:
    """Wraps a DataLoader yielding variable-size batches and redistributes into
    fixed-size batches with optional shuffle buffering for cross-batch mixing.

    Pipeline: unbatch (flatten) → shuffle buffer (optional) → rebatch (fixed size).

    The inner DataLoader is expected to yield dicts of tensors/lists where the
    first dimension (after an optional DataLoader batch_size=1 wrapper) is the
    variable item count. All items must have the same tensor shapes beyond the
    first dimension.

    For training: use shuffle_buffer_size > 0 for cross-chunk mixing.
    For validation: use shuffle_buffer_size = 0 for deterministic, exhaustive iteration.
    """

    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader,
        batch_size: int,
        shuffle_buffer_size: int = 0,
        prefetch: int = 4,
    ):
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.prefetch = prefetch

    @property
    def worker_init_fn(self):
        return self.dataloader.worker_init_fn

    @worker_init_fn.setter
    def worker_init_fn(self, fn):
        self.dataloader.worker_init_fn = fn

    @property
    def dataset(self):
        return self.dataloader.dataset

    def __iter__(self) -> Iterator[dict]:
        chunks = _squeeze_chunks(self.dataloader)
        batches = _tensor_rebatch(chunks, self.batch_size, self.shuffle_buffer_size)
        if self.prefetch > 0:
            yield from _prefetch(batches, self.prefetch)
        else:
            yield from batches


def _prefetch(batches: Iterator[dict], n: int) -> Iterator[dict]:
    """Yield batches with background-thread prefetching."""
    q: queue.Queue = queue.Queue(maxsize=n)
    sentinel = object()

    def _fill():
        try:
            for batch in batches:
                q.put(batch)
        except Exception as e:
            q.put(e)
        q.put(sentinel)

    t = threading.Thread(target=_fill, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        if isinstance(item, Exception):
            raise item
        yield item
    t.join()


def _squeeze_chunks(dataloader: Iterable[dict]) -> Iterator[dict]:
    """Yield squeezed chunks from the inner DataLoader, skipping empties."""
    for chunk in dataloader:
        squeezed = {}
        n = 0
        for key, val in chunk.items():
            if isinstance(val, torch.Tensor):
                if val.dim() >= 2 and val.shape[0] == 1:
                    val = val.squeeze(0)
                if n == 0:
                    n = val.shape[0]
            elif isinstance(val, (list, tuple)):
                if len(val) == 1 and isinstance(val[0], (list, tuple)):
                    val = val[0]
                if n == 0:
                    n = len(val)
            squeezed[key] = val
        if n > 0:
            yield squeezed


def _cat_chunks(buf_parts: list[dict]) -> dict:
    """Concatenate a list of chunk dicts along the batch dimension."""
    result = {}
    for key in buf_parts[0]:
        vals = [p[key] for p in buf_parts]
        if isinstance(vals[0], torch.Tensor):
            result[key] = torch.cat(vals, dim=0)
        else:
            merged: list = []
            for v in vals:
                if isinstance(v, (list, tuple)):
                    merged.extend(v)
                else:
                    merged.append(v)
            result[key] = merged
    return result


def _buf_len(buf: dict) -> int:
    """Get the number of items in a concatenated buffer dict."""
    for val in buf.values():
        if isinstance(val, torch.Tensor):
            return val.shape[0]
        if isinstance(val, (list, tuple)):
            return len(val)
    return 0


def _buf_index(buf: dict, idx: torch.Tensor) -> dict:
    """Index a buffer dict with a tensor of indices."""
    result = {}
    for key, val in buf.items():
        if isinstance(val, torch.Tensor):
            result[key] = val[idx]
        else:
            result[key] = [val[i] for i in idx.tolist()]
    return result


def _buf_slice(buf: dict, start: int, end: int) -> dict:
    """Slice a buffer dict from start to end."""
    result = {}
    for key, val in buf.items():
        if isinstance(val, torch.Tensor):
            result[key] = val[start:end]
        else:
            result[key] = val[start:end]
    return result


def _tensor_rebatch(
    chunks: Iterator[dict],
    batch_size: int,
    shuffle_buffer_size: int,
) -> Iterator[dict]:
    """Accumulate chunks into a buffer, shuffle, and yield fixed-size batches.

    Operates at the tensor level: torch.cat to accumulate, torch.randperm to
    shuffle, and slice to emit batches. Avoids per-item Python dict overhead.
    """
    import logging
    import time

    logger = logging.getLogger(__name__)

    buf_parts: list[dict] = []
    buf_total = 0
    threshold = max(shuffle_buffer_size, batch_size)
    t_wait_total = 0.0
    t_rebatch_total = 0.0
    n_cycles = 0
    n_batches = 0

    chunks_iter = iter(chunks)
    while True:
        t0 = time.monotonic()
        try:
            chunk = next(chunks_iter)
        except StopIteration:
            break
        t_wait_total += time.monotonic() - t0
        buf_parts.append(chunk)
        n = _buf_len(chunk)
        buf_total += n

        if buf_total >= threshold:
            t1 = time.monotonic()
            buf = _cat_chunks(buf_parts)
            buf_parts = []
            buf_total = 0

            if shuffle_buffer_size > 0:
                perm = torch.randperm(n := _buf_len(buf))
                buf = _buf_index(buf, perm)
            else:
                n = _buf_len(buf)
            t_rebatch_total += time.monotonic() - t1

            pos = 0
            while pos + batch_size <= n:
                yield _buf_slice(buf, pos, pos + batch_size)
                n_batches += 1
                pos += batch_size

            if pos < n:
                remainder = _buf_slice(buf, pos, n)
                buf_parts = [remainder]
                buf_total = n - pos

            n_cycles += 1
            if n_cycles % 1 == 0:
                logger.warning(
                    "[rebatcher] %d cycles, %d batches | "
                    "chunk_wait: %.1fs, rebatch: %.1fs",
                    n_cycles, n_batches, t_wait_total, t_rebatch_total,
                )

    # Drain remainder
    if buf_parts:
        buf = _cat_chunks(buf_parts)
        n = _buf_len(buf)
        if shuffle_buffer_size > 0 and n > 1:
            perm = torch.randperm(n)
            buf = _buf_index(buf, perm)
        pos = 0
        while pos < n:
            end = min(pos + batch_size, n)
            yield _buf_slice(buf, pos, end)
            pos = end

    logger.warning(
        "[rebatcher] DONE %d cycles, %d batches | "
        "chunk_wait: %.1fs, rebatch: %.1fs",
        n_cycles, n_batches, t_wait_total, t_rebatch_total,
    )
