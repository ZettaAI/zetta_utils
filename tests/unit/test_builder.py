# pylint: disable=missing-docstring,protected-access,unused-argument,redefined-outer-name,invalid-name, line-too-long
from dataclasses import dataclass
from typing import Any

import pytest

from zetta_utils import builder
from zetta_utils.builder import SPECIAL_KEYS


@dataclass
class DummyA:
    a: Any


@dataclass
class DummyB:
    b: Any


@dataclass
class DummyC:
    c: Any
    cc: Any


@pytest.fixture
def register_dummy_a():
    builder.register("dummy_a")(DummyA)
    yield
    del builder.REGISTRY["dummy_a"]


@pytest.fixture
def register_dummy_b():
    builder.register("dummy_b")(DummyB)
    yield
    del builder.REGISTRY["dummy_b"]


@pytest.fixture
def register_dummy_c():
    builder.register("dummy_c")(DummyC)
    yield
    del builder.REGISTRY["dummy_c"]


def test_build_from_path(mocker):
    spec = {"k": "v"}
    mocker.patch("zetta_utils.parsing.cue.load", return_value=spec)
    result = builder.build(path="dummy_path")
    assert result == spec


@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        "abc",
        (1, "abc"),
        {"int": 1, "str": "abc", "tuple": (1, 2), "dict": {"yes": "sir"}},
    ],
)
def test_identity_builds(value):
    spec = {"k": value}
    result = builder.build(spec)
    assert result == spec


@pytest.mark.parametrize(
    "value, expected_exc",
    [
        [None, ValueError],
        [1, Exception],
        ["yo", Exception],
        [{"@type": "something_not_registered"}, ValueError],
        [{"@type": "dummy_a", "a": 1, "@mode": "unsupported_mode_5566"}, ValueError],
    ],
)
def test_parse_exc(value, expected_exc, register_dummy_a):
    with pytest.raises(expected_exc):
        builder.build(value)


def test_register(register_dummy_a):
    assert builder.REGISTRY["dummy_a"].fn == DummyA


@pytest.mark.parametrize(
    "spec, expected",
    [
        [{"a": "b"}, {"a": "b"}],
        [{"a": ValueError}, {"a": ValueError}],
        [{SPECIAL_KEYS["type"]: "dummy_a", "a": 2}, DummyA(a=2)],
        [{SPECIAL_KEYS["type"]: "dummy_b", "b": 2}, DummyB(b=2)],
        [
            {SPECIAL_KEYS["type"]: "dummy_a", "a": [{SPECIAL_KEYS["type"]: "dummy_b", "b": 3}]},
            DummyA(a=[DummyB(b=3)]),
        ],
        [
            {
                SPECIAL_KEYS["type"]: "dummy_a",
                "a": ({SPECIAL_KEYS["type"]: "dummy_b", "b": 3},),
            },
            DummyA(a=(DummyB(b=3),)),
        ],
    ],
)
def test_build(spec: dict, expected: Any, register_dummy_a, register_dummy_b):
    result = builder.build(spec)
    assert result == expected
    if hasattr(result, "__dict__"):
        assert result.__built_with_spec == spec


def test_build_partial(register_dummy_a, register_dummy_c):
    partial = builder.build(
        {"@type": "dummy_c", "@mode": "partial", "c": {"@type": "dummy_a", "a": "yolo"}}
    )
    result = partial(cc="mate")
    expected = DummyC(c=DummyA(a="yolo"), cc="mate")
    assert result == expected


@pytest.mark.parametrize(
    "spec, arg, expected",
    [
        [{"@type": "lambda", "lambda_str": "lambda a:a"}, "foo", "foo"],
        [{"@type": "lambda", "lambda_str": "lambda a:a+1"}, 5, 6],
    ],
)
def test_lambda(spec: dict, arg: Any, expected: Any):
    assert builder.build(spec)(arg) == expected


@pytest.mark.parametrize(
    "value, expected_exc",
    [
        [{"@type": "lambda", "lambda_str": 3}, TypeError],
        [
            {"@type": "lambda", "@mode": "partial", "lambda_str": "lambda badmode: badmode"},
            ValueError,
        ],
        [{"@type": "lambda", "lambda_str": "notalambdastring"}, ValueError],
        [
            {
                "@type": "lambda",
                "lambda_str": "lambda really_long_lambda_str_that_can_contain_arbitrary_code_to_execute_like_bitcoin_mining: None",
            },
            ValueError,
        ],
    ],
)
def test_lambda_exc(value, expected_exc):
    with pytest.raises(expected_exc):
        builder.build(value)


def test_double_register_exc(register_dummy_a):
    with pytest.raises(RuntimeError):
        builder.register("dummy_a")(DummyA)
