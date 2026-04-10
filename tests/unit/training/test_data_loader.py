"""Tests for RebatchingDataLoader and its helper functions."""

import torch

from zetta_utils.training.data_loader import (
    RebatchingDataLoader,
    _buf_index,
    _buf_len,
    _buf_slice,
    _cat_chunks,
    _pin_batches,
    _prefetch,
    _squeeze_chunks,
    _tensor_rebatch,
)


# --- _buf_len ---


def test_buf_len_tensor():
    buf = {"x": torch.zeros(5, 3), "y": torch.zeros(5)}
    assert _buf_len(buf) == 5


def test_buf_len_list():
    buf = {"ids": [1, 2, 3]}
    assert _buf_len(buf) == 3


def test_buf_len_empty():
    assert _buf_len({}) == 0


# --- _buf_index ---


def test_buf_index_tensor():
    buf = {"x": torch.tensor([[1, 2], [3, 4], [5, 6]])}
    idx = torch.tensor([2, 0])
    result = _buf_index(buf, idx)
    torch.testing.assert_close(result["x"], torch.tensor([[5, 6], [1, 2]]))


def test_buf_index_list():
    buf = {"ids": ["a", "b", "c"]}
    idx = torch.tensor([1, 2])
    result = _buf_index(buf, idx)
    assert result["ids"] == ["b", "c"]


# --- _buf_slice ---


def test_buf_slice_tensor():
    buf = {"x": torch.arange(10)}
    result = _buf_slice(buf, 2, 5)
    torch.testing.assert_close(result["x"], torch.tensor([2, 3, 4]))


def test_buf_slice_list():
    buf = {"ids": list(range(10))}
    result = _buf_slice(buf, 3, 6)
    assert result["ids"] == [3, 4, 5]


# --- _cat_chunks ---


def test_cat_chunks_tensors():
    parts = [
        {"x": torch.tensor([[1, 2]]), "y": torch.tensor([0.5])},
        {"x": torch.tensor([[3, 4], [5, 6]]), "y": torch.tensor([0.6, 0.7])},
    ]
    result = _cat_chunks(parts)
    assert result["x"].shape == (3, 2)
    assert result["y"].shape == (3,)
    torch.testing.assert_close(result["x"], torch.tensor([[1, 2], [3, 4], [5, 6]]))


def test_cat_chunks_lists():
    parts = [
        {"ids": ["a", "b"]},
        {"ids": ["c"]},
    ]
    result = _cat_chunks(parts)
    assert result["ids"] == ["a", "b", "c"]


def test_cat_chunks_mixed():
    parts = [
        {"x": torch.tensor([1]), "ids": ["a"]},
        {"x": torch.tensor([2, 3]), "ids": ["b", "c"]},
    ]
    result = _cat_chunks(parts)
    torch.testing.assert_close(result["x"], torch.tensor([1, 2, 3]))
    assert result["ids"] == ["a", "b", "c"]


# --- _squeeze_chunks ---


def test_squeeze_chunks_removes_batch_dim():
    chunks = [
        {"x": torch.zeros(1, 5, 3), "y": torch.zeros(1, 5)},
    ]
    result = list(_squeeze_chunks(chunks))
    assert len(result) == 1
    assert result[0]["x"].shape == (5, 3)
    assert result[0]["y"].shape == (5,)


def test_squeeze_chunks_skips_empty():
    chunks = [
        {"x": torch.zeros(1, 0, 3)},
        {"x": torch.zeros(1, 5, 3)},
    ]
    result = list(_squeeze_chunks(chunks))
    assert len(result) == 1
    assert result[0]["x"].shape == (5, 3)


def test_squeeze_chunks_nested_list():
    chunks = [
        {"ids": [["a", "b"]], "x": torch.zeros(1, 2, 3)},
    ]
    result = list(_squeeze_chunks(chunks))
    assert result[0]["ids"] == ["a", "b"]


def test_squeeze_chunks_no_squeeze_needed():
    chunks = [
        {"x": torch.zeros(5, 3)},
    ]
    result = list(_squeeze_chunks(chunks))
    assert result[0]["x"].shape == (5, 3)


# --- _tensor_rebatch ---


def test_tensor_rebatch_basic():
    """Chunks of variable size → fixed-size batches."""
    chunks = iter([
        {"x": torch.arange(3), "y": torch.arange(3) * 10},
        {"x": torch.arange(3, 8), "y": torch.arange(3, 8) * 10},
        {"x": torch.arange(8, 10), "y": torch.arange(8, 10) * 10},
    ])
    batches = list(_tensor_rebatch(chunks, batch_size=4, shuffle_buffer_size=0))

    # 10 items total, batch_size=4 → 2 full + 1 remainder
    assert len(batches) == 3
    assert batches[0]["x"].shape[0] == 4
    assert batches[1]["x"].shape[0] == 4
    assert batches[2]["x"].shape[0] == 2


def test_tensor_rebatch_with_shuffle():
    """With shuffle buffer, output should still have correct total count."""
    chunks = iter([
        {"x": torch.ones(10)},
        {"x": torch.ones(10) * 2},
    ])
    batches = list(_tensor_rebatch(chunks, batch_size=5, shuffle_buffer_size=10))

    total = sum(b["x"].shape[0] for b in batches)
    assert total == 20


def test_tensor_rebatch_with_watermark():
    """Low watermark keeps residual in buffer for cross-chunk mixing."""
    chunks = iter([
        {"x": torch.arange(20)},
        {"x": torch.arange(20, 40)},
    ])
    batches = list(_tensor_rebatch(
        chunks, batch_size=5, shuffle_buffer_size=15, shuffle_buffer_low_watermark=5
    ))

    total = sum(b["x"].shape[0] for b in batches)
    assert total == 40


def test_tensor_rebatch_empty_input():
    batches = list(_tensor_rebatch(iter([]), batch_size=4, shuffle_buffer_size=0))
    assert batches == []


def test_tensor_rebatch_single_small_chunk():
    chunks = iter([{"x": torch.tensor([1, 2])}])
    batches = list(_tensor_rebatch(chunks, batch_size=4, shuffle_buffer_size=0))
    assert len(batches) == 1
    assert batches[0]["x"].shape[0] == 2


# --- _pin_batches ---


def test_pin_batches():
    batches = [{"x": torch.zeros(3), "ids": ["a", "b", "c"]}]
    result = list(_pin_batches(iter(batches)))
    assert len(result) == 1
    assert result[0]["x"].shape == (3,)
    assert result[0]["ids"] == ["a", "b", "c"]


# --- _prefetch ---


def test_prefetch_yields_all():
    def gen():
        for i in range(5):
            yield {"x": torch.tensor([i])}

    result = list(_prefetch(gen(), n=2))
    assert len(result) == 5
    for i, batch in enumerate(result):
        assert batch["x"].item() == i


def test_prefetch_propagates_error():
    def gen():
        yield {"x": torch.tensor([0])}
        raise ValueError("test error")

    batches = _prefetch(gen(), n=2)
    first = next(batches)
    assert first["x"].item() == 0
    try:
        next(batches)
        assert False, "Should have raised"
    except ValueError as e:
        assert "test error" in str(e)


def test_prefetch_empty():
    result = list(_prefetch(iter([]), n=2))
    assert result == []


# --- RebatchingDataLoader integration ---


class _FakeDataset(torch.utils.data.Dataset):
    """Returns pre-batched dicts (simulating SegContactDataset chunks)."""

    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


def test_rebatching_dataloader_basic():
    """End-to-end: variable chunks → fixed batches through the full pipeline."""
    # Each chunk is a dict with tensors that have a leading batch_size=1 dim
    # (as DataLoader with batch_size=1 wraps them)
    chunks = [
        {"x": torch.ones(10, 3)},
        {"x": torch.ones(10, 3) * 2},
    ]
    ds = _FakeDataset(chunks)
    inner = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    loader = RebatchingDataLoader(
        dataloader=inner, batch_size=5, shuffle_buffer_size=0, prefetch=0
    )

    batches = list(loader)
    total = sum(b["x"].shape[0] for b in batches)
    assert total == 20
    for b in batches:
        assert b["x"].shape[0] <= 5


def test_rebatching_dataloader_skips_empty():
    """Empty chunks should be skipped."""
    chunks = [
        {"x": torch.ones(0, 3)},  # empty
        {"x": torch.ones(5, 3)},  # 5 items
    ]
    ds = _FakeDataset(chunks)
    inner = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    loader = RebatchingDataLoader(
        dataloader=inner, batch_size=3, shuffle_buffer_size=0, prefetch=0
    )

    batches = list(loader)
    total = sum(b["x"].shape[0] for b in batches)
    assert total == 5


def test_rebatching_dataloader_with_pin_memory_and_prefetch():
    """Test pin_memory and prefetch delegation paths in __iter__."""
    chunks = [
        {"x": torch.ones(5, 3)},
        {"x": torch.ones(5, 3) * 2},
    ]
    ds = _FakeDataset(chunks)
    inner = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    loader = RebatchingDataLoader(
        dataloader=inner, batch_size=4, shuffle_buffer_size=0,
        pin_memory=True, prefetch=1,
    )

    batches = list(loader)
    total = sum(b["x"].shape[0] for b in batches)
    assert total == 10


def test_cat_chunks_scalar_values():
    """Test _cat_chunks with bare scalar (non-list, non-tuple) values."""
    parts = [
        {"x": torch.tensor([1]), "tag": "a"},
        {"x": torch.tensor([2]), "tag": "b"},
    ]
    result = _cat_chunks(parts)
    torch.testing.assert_close(result["x"], torch.tensor([1, 2]))
    assert result["tag"] == ["a", "b"]


def test_rebatching_dataloader_properties():
    """Test dataset and worker_init_fn properties."""
    ds = _FakeDataset([])
    inner = torch.utils.data.DataLoader(ds, batch_size=1)
    loader = RebatchingDataLoader(dataloader=inner, batch_size=5)

    assert loader.dataset is ds

    def my_init(worker_id):
        pass

    loader.worker_init_fn = my_init
    assert loader.worker_init_fn is my_init
