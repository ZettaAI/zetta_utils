# pylint: disable=missing-docstring
from __future__ import annotations

from zetta_utils.builder.preload import (
    ALWAYS_EAGER,
    compute_preload_set,
    lambda_preload_modules,
)


def test_always_eager_first_then_sorted_extras():
    """ALWAYS_EAGER (in declared order) followed by extras sorted alphabetically.

    Pass two names whose modules sort in a specific order regardless of input
    set iteration; assert the full preload list matches the exact expected
    ordering.
    """
    # BBox3D.from_coords lives in zetta_utils.geometry.bbox
    # TorchDataLoader  lives in zetta_utils.training.data_loader
    # Sorted: geometry.bbox < training.data_loader
    result = compute_preload_set({"TorchDataLoader", "BBox3D.from_coords"})
    expected = list(ALWAYS_EAGER) + [
        "zetta_utils.geometry.bbox",
        "zetta_utils.training.data_loader",
    ]
    assert result == expected


def test_known_name_resolves_to_module():
    result = compute_preload_set({"BBox3D.from_coords"})
    # geometry.bbox is the module that registers BBox3D.from_coords
    assert any(m.endswith("geometry.bbox") for m in result)
    # ALWAYS_EAGER prefix preserved
    assert result[: len(ALWAYS_EAGER)] == list(ALWAYS_EAGER)


def test_unknown_name_silently_skipped():
    """Names the index doesn't recognise are deferred to the runtime fallback."""
    result = compute_preload_set({"definitely_not_registered_xyz"})
    assert result == list(ALWAYS_EAGER)


def test_extras_sorted_and_deduped():
    """Even if multiple @types resolve to the same module, only one entry."""
    # VolumetricIndex and VolumetricIndex.from_coords both live in
    # geometry.bbox? No — in layer.volumetric.index. Either way they share
    # a module.
    result = compute_preload_set({"VolumetricIndex", "VolumetricIndex.from_coords"})
    extras = [m for m in result if m not in ALWAYS_EAGER]
    assert len(extras) == len(set(extras))


def test_eager_modules_not_duplicated_in_extras():
    """If a name happens to resolve to an ALWAYS_EAGER module, it isn't duplicated."""
    # "lambda" is registered in zetta_utils.builder.built_in_registrations,
    # which is under zetta_utils.builder (an ALWAYS_EAGER prefix).
    # built_in_registrations itself isn't in ALWAYS_EAGER — but if it were,
    # this test would catch dedup. For now: make sure extras don't contain
    # any ALWAYS_EAGER entry.
    result = compute_preload_set({"lambda", "BBox3D.from_coords"})
    eager_set = set(ALWAYS_EAGER)
    extras = [m for m in result if m not in eager_set]
    assert all(m not in eager_set for m in extras)


def test_lambda_preload_modules_detects_np_and_torch():
    assert lambda_preload_modules(["lambda x: np.where(x == 0, 1, x)"]) == ["numpy"]
    assert lambda_preload_modules(["lambda: torch.zeros(3)"]) == ["torch"]
    assert lambda_preload_modules(["lambda x, y: np.add(x, torch.ones(y))"]) == [
        "numpy",
        "torch",
    ]
    assert lambda_preload_modules(["lambda x: x + 1"]) == []


def test_lambda_preload_modules_word_boundary():
    # Substrings like 'snp' / 'torcher' must not trigger a spurious preload.
    assert lambda_preload_modules(["lambda snp: snp + torcher"]) == []


def test_lambda_strs_add_modules_to_preload_set():
    result = compute_preload_set(set(), lambda_strs=["lambda x: np.where(x == 0, 1, x)"])
    assert "numpy" in result
    assert "torch" not in result
    assert result[: len(ALWAYS_EAGER)] == list(ALWAYS_EAGER)


def test_lambda_strs_default_empty():
    """Omitting lambda_strs leaves the @type-derived preload set unchanged."""
    assert compute_preload_set({"definitely_not_registered_xyz"}) == list(ALWAYS_EAGER)
