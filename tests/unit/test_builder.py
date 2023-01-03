# pylint: disable=missing-docstring,protected-access,unused-argument,redefined-outer-name,invalid-name
from dataclasses import dataclass
from typing import Any

import pytest

from zetta_utils import builder
from zetta_utils.common.partial import ComparablePartial
from zetta_utils.typing import IntVec3D, Vec3D

PARSE_KEY = "@type"
RECURSIVE_KEY = "@recursive_parse"
MODE_KEY = "@mode"


@dataclass
class DummyA:
    a: Any


@dataclass
class DummyB:
    b: Any


@dataclass
class DummyC:
    vec: Any
    intvec: Any


@pytest.fixture
def register_dummy_a():
    builder.parser.register("dummy_a")(DummyA)
    yield
    del builder.parser.REGISTRY["dummy_a"]


@pytest.fixture
def register_dummy_b():
    builder.parser.register("dummy_b")(DummyB)
    yield
    del builder.parser.REGISTRY["dummy_b"]


@pytest.fixture
def register_dummy_c():
    builder.parser.register("dummy_c", cast_to_vec3d=["vec"], cast_to_intvec3d=["intvec"])(DummyC)
    yield
    del builder.parser.REGISTRY["dummy_c"]


def test_build_from_path(mocker):
    spec = {"k": "v"}
    mocker.patch("zetta_utils.parsing.cue.load", return_value=spec)
    result = builder.parser.build(path="dummy_path", must_build=False)
    assert result == spec


@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        "abc",
        (1, "abc"),
        {"int": 1, "str": "abc", "tuple": (1, 2, 3), "dict": {"yes": "sir"}},
    ],
)
def test_identity_builds(value):
    spec = {"k": value}
    result = builder.parser.build(spec, must_build=False)
    assert result == spec


@pytest.mark.parametrize(
    "value, expected_exc",
    [
        [None, ValueError],
        [1, TypeError],
        ["yo", TypeError],
        [{}, ValueError],
        [{"a": "b"}, ValueError],
        [{"@type": "something_not_registered"}, KeyError],
        [{"@type": "dummy_a", "a": 1, "@mode": "unsupported_mode_5566"}, ValueError],
        [{"@type": "dummy_a", "a": TypeError}, ValueError],
    ],
)
def test_parse_exc(value, expected_exc, register_dummy_a):
    with pytest.raises(expected_exc):
        builder.build(value, must_build=True)


def test_register(register_dummy_a):
    assert builder.parser.REGISTRY["dummy_a"]["class"] == DummyA


def test_register_casting(register_dummy_c):
    assert builder.parser.REGISTRY["dummy_c"]["class"] == DummyC
    assert builder.parser.REGISTRY["dummy_c"]["cast_to_vec3d"] == ["vec"]
    assert builder.parser.REGISTRY["dummy_c"]["cast_to_intvec3d"] == ["intvec"]


@pytest.mark.parametrize(
    "spec, expected",
    [
        [{"a": "b"}, {"a": "b"}],
        [{PARSE_KEY: "dummy_a", "a": 2}, DummyA(a=2)],
        [{PARSE_KEY: "dummy_b", "b": 2}, DummyB(b=2)],
        [
            {PARSE_KEY: "dummy_a", "a": [{PARSE_KEY: "dummy_b", "b": 3}]},
            DummyA(a=[DummyB(b=3)]),
        ],
        [
            {
                PARSE_KEY: "dummy_a",
                RECURSIVE_KEY: True,
                "a": ({PARSE_KEY: "dummy_b", "b": 3},),
            },
            DummyA(a=(DummyB(b=3),)),
        ],
        [
            {
                PARSE_KEY: "dummy_a",
                RECURSIVE_KEY: False,
                "a": ({PARSE_KEY: "dummy_b", "b": 3},),
            },
            DummyA(a=({PARSE_KEY: "dummy_b", "b": 3},)),
        ],
        [
            {PARSE_KEY: "dummy_a", MODE_KEY: "partial", "a": [{PARSE_KEY: "dummy_b", "b": 3}]},
            ComparablePartial(DummyA, a=[DummyB(b=3)]),
        ],
        [
            {PARSE_KEY: "dummy_a", MODE_KEY: "lazy", "a": [{PARSE_KEY: "dummy_b", "b": 3}]},
            ComparablePartial(
                builder.build, spec={PARSE_KEY: "dummy_a", "a": [{PARSE_KEY: "dummy_b", "b": 3}]}
            ),
        ],
    ],
)
def test_build(spec: dict, expected: Any, register_dummy_a, register_dummy_b):
    result = builder.build(spec, must_build=False)
    assert result == expected
    if hasattr(result, "__dict__"):
        assert result.__built_with_spec == spec


@pytest.mark.parametrize(
    "spec, expected",
    [
        [
            {"@type": "dummy_c", "vec": (1.5, 2.5, 3.5), "intvec": (1, 2, 3)},
            DummyC(vec=Vec3D(1.5, 2.5, 3.5), intvec=IntVec3D(1, 2, 3)),
        ],
    ],
)
def test_cast(spec: dict, expected: Any, register_dummy_c):
    result = builder.build(spec, must_build=False)
    assert result == expected
