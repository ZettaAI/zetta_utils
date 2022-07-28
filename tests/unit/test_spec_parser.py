# pylint: disable=missing-docstring,protected-access,unused-argument,redefined-outer-name
from dataclasses import dataclass
from typing import Any
import pytest

from zetta_utils import spec_parser


PARSE_KEY = "<type>"
RECURSIVE_KEY = "<recursive_parse>"


@dataclass
class DummyA:
    a: Any


@dataclass
class DummyB:
    b: Any


@pytest.fixture
def register_dummy_a():
    spec_parser.parser.register("dummy_a")(DummyA)
    yield
    del spec_parser.parser.REGISTRY["dummy_a"]


@pytest.fixture
def register_dummy_b():
    spec_parser.parser.register("dummy_b")(DummyB)
    yield
    del spec_parser.parser.REGISTRY["dummy_b"]


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
    result = spec_parser.parser._build(value)
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
        spec_parser.build(value)


@pytest.mark.parametrize("value", [1, ["yo"]])
def test_nondict_exc(value):
    with pytest.raises(TypeError):
        spec_parser.build(value)


def test_register(register_dummy_a):
    assert spec_parser.parser.REGISTRY["dummy_a"] == DummyA


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
                "a": [{PARSE_KEY: "dummy_b", "b": 3}],
            },
            DummyA(a=[DummyB(b=3)]),
        ],
        [
            {
                PARSE_KEY: "dummy_a",
                RECURSIVE_KEY: False,
                "a": [{PARSE_KEY: "dummy_b", "b": 3}],
            },
            DummyA(a=[{PARSE_KEY: "dummy_b", "b": 3}]),
        ],
    ],
)
def test_build(spec: dict, expected: Any, register_dummy_a, register_dummy_b):
    result = spec_parser.build(spec, must_build=False)
    assert result == expected
