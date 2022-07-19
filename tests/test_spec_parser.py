# pylint: disable=missing-docstring,protected-access,unused-argument,redefined-outer-name
from dataclasses import dataclass
from typing import Any
import pytest

import zetta_utils as zu


@dataclass
class DummyA:
    a: Any


@dataclass
class DummyB:
    b: Any


@pytest.fixture
def register_dummy_a():
    zu.spec_parser.parser.register("dummy_a")(DummyA)
    yield
    del zu.spec_parser.parser.REGISTRY["dummy_a"]


@pytest.fixture
def register_dummy_b():
    zu.spec_parser.parser.register("dummy_b")(DummyB)
    yield
    del zu.spec_parser.parser.REGISTRY["dummy_b"]


@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        "abc",
        [1, "abc"],
        {"int": 1, "str": "abc", "list": [1, 2, 3], "dict": {"yes": "sir"}},
    ],
)
def test_identity_builds(value):
    result = zu.spec_parser.parser._build(value)
    assert result == value


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"a": "b"},
    ],
)
def test_must_build_exc(value):
    with pytest.raises(Exception):
        zu.spec_parser.build(value)


@pytest.mark.parametrize("value", [1, ["yo"]])
def test_nondict_exc(value):
    with pytest.raises(TypeError):
        zu.spec_parser.build(value)


def test_register(register_dummy_a):
    assert zu.spec_parser.parser.REGISTRY["dummy_a"] == DummyA
    assert DummyA._spec_name == "dummy_a"  # pylint: disable=no-member


@pytest.mark.parametrize(
    "spec, expected",
    [
        [{"a": "b"}, {"a": "b"}],
        [{"_parse_as": "dummy_a", "a": 2}, DummyA(a=2)],
        [{"_parse_as": "dummy_b", "b": 2}, DummyB(b=2)],
        [
            {"_parse_as": "dummy_a", "a": [{"_parse_as": "dummy_b", "b": 3}]},
            DummyA(a=[DummyB(b=3)]),
        ],
        [
            {
                "_parse_as": "dummy_a",
                "_recursive_parse": True,
                "a": [{"_parse_as": "dummy_b", "b": 3}],
            },
            DummyA(a=[DummyB(b=3)]),
        ],
        [
            {
                "_parse_as": "dummy_a",
                "_recursive_parse": False,
                "a": [{"_parse_as": "dummy_b", "b": 3}],
            },
            DummyA(a=[{"_parse_as": "dummy_b", "b": 3}]),
        ],
    ],
)
def test_build(spec: dict, expected: Any, register_dummy_a, register_dummy_b):
    result = zu.spec_parser.build(spec, must_build=False)
    assert result == expected
