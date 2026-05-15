# pylint: disable=missing-docstring,protected-access
"""Coverage for the np.* and torch.* dynamic resolvers."""
from __future__ import annotations

import pytest

from zetta_utils.builder import built_in_registrations as bir
from zetta_utils.builder.building import BuilderPartial

# ---------- lambda + invoke_lambda_str (covers lines 13-26 in built_in_registrations) ----------


def test_efficient_parse_lambda_str_returns_partial():
    out = bir.efficient_parse_lambda_str("lambda x: x + 1")
    assert isinstance(out, BuilderPartial)
    assert out.spec["@type"] == "invoke_lambda_str"


def test_efficient_parse_lambda_str_rejects_non_string():
    with pytest.raises(TypeError, match="must be a string"):
        bir.efficient_parse_lambda_str(123)  # type: ignore[arg-type]


def test_efficient_parse_lambda_str_rejects_non_lambda():
    with pytest.raises(ValueError, match="must start with 'lambda'"):
        bir.efficient_parse_lambda_str("def f(): pass")


def test_invoke_lambda_str_runs_lambda():
    # Pre-existing signature annotates *args as `list`; ints unpack into the
    # lambda fine at runtime.
    args: tuple = (2, 3)
    assert bir.invoke_lambda_str(*args, lambda_str="lambda a, b: a * b") == 6


# ---------- _resolve_numpy ----------


def test_resolve_numpy_routine():
    entry = bir._resolve_numpy("np.allclose")
    assert entry is not None
    assert callable(entry.fn)


def test_resolve_numpy_constant():
    entry = bir._resolve_numpy("np.inf")
    assert entry is not None
    # Constant is wrapped in a thunk that returns the value when called.
    assert entry.fn() == float("inf")


def test_resolve_numpy_unknown_attr():
    assert bir._resolve_numpy("np.totally_made_up_xyz") is None


def test_resolve_numpy_underscore_rejected():
    assert bir._resolve_numpy("np._private") is None


def test_resolve_numpy_empty_suffix():
    assert bir._resolve_numpy("np.") is None


def test_resolve_numpy_non_routine_non_constant():
    # numpy module itself (np.ndarray is a class) — not a routine, not a float/None
    # numpy attribute that's a type: dtype is a class, isroutine returns False,
    # not isinstance(., (float, NoneType)) → resolver returns None.
    assert bir._resolve_numpy("np.dtype") is None


# ---------- _resolve_torch ----------


def test_resolve_torch_nested_class():
    entry = bir._resolve_torch("torch.nn.Linear")
    assert entry is not None
    assert callable(entry.fn)


def test_resolve_torch_unknown_path():
    assert bir._resolve_torch("torch.nn.NotARealClass") is None


def test_resolve_torch_empty_suffix():
    assert bir._resolve_torch("torch.") is None


def test_resolve_torch_underscore_rejected():
    assert bir._resolve_torch("torch._private") is None


def test_resolve_torch_non_callable_returns_none():
    # torch.AVG is an AggregationType enum value: public name, not callable.
    assert bir._resolve_torch("torch.AVG") is None
