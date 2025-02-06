# pylint: disable=missing-docstring,protected-access,unused-argument,redefined-outer-name,invalid-name, line-too-long
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import dill
import pytest

from zetta_utils import builder
from zetta_utils.builder import SPECIAL_KEYS, BuilderPartial


@dataclass
class DummyA:
    a: Any


@dataclass
class DummyAV2:
    a: Any


@dataclass
class DummyB:
    b: Any


@dataclass
class DummyBV2:
    b: Any


@dataclass
class DummyC:
    c: Any
    cc: Any


@dataclass
class DummyNonparallel:
    ...


@pytest.fixture
def register_dummy_a():
    builder.register("dummy_a", versions=">=0.0.0")(DummyA)
    yield
    builder.unregister(name="dummy_a", fn=DummyA)


@pytest.fixture
def register_dummy_b():
    builder.register("dummy_b", versions=">=0.0.0")(DummyB)
    yield
    builder.unregister(name="dummy_b", fn=DummyB)


@pytest.fixture
def register_dummy_c():
    builder.register("dummy_c", versions=">=0.0.0")(DummyC)
    yield
    builder.unregister(name="dummy_c", fn=DummyC)


@pytest.fixture
def register_dummy_a_v0():
    builder.register("dummy_a", allow_parallel=False, versions="==0.0.0")(DummyA)
    yield
    builder.unregister(name="dummy_a", fn=DummyA, allow_parallel=False, versions="==0.0.0")


@pytest.fixture
def register_dummy_nonparallel():
    builder.register("dummy_nonparallel", allow_parallel=False, versions="==0.0.0")(
        DummyNonparallel
    )
    yield
    builder.unregister(
        name="dummy_nonparallel", fn=DummyNonparallel, allow_parallel=False, versions="==0.0.0"
    )


@pytest.fixture
def register_dummy_a_v2():
    builder.register("dummy_a", versions="==0.0.2")(DummyAV2)
    yield
    builder.unregister(name="dummy_a", fn=DummyAV2, versions="==0.0.2")


@pytest.fixture
def register_dummy_b_v2():
    builder.register("dummy_b", versions="==0.0.2")(DummyBV2)
    yield
    builder.unregister(name="dummy_b", fn=DummyBV2)


def test_build_from_path(mocker):
    spec = {"k": "v"}
    mocker.patch("zetta_utils.parsing.cue.load", return_value=spec)
    result = builder.build(path="dummy_path")
    assert result == spec


def sleeper_function(sleep_time: float, **kwargs):
    for _, v in kwargs.items():
        assert v is True
    time.sleep(sleep_time)
    return True


@pytest.fixture
def register_sleeper_func():
    builder.register("sleeper_func", versions=">=0.0.0")(sleeper_function)
    yield
    builder.unregister(name="sleeper_func", fn=sleeper_function)


# Has to be pretty long to manifest parallelization difference
# when pytest-cov is enabled
SLEEP_TIME = 2

SLEEPER_SPEC = {
    "@type": "sleeper_func",
    "sleep_time": SLEEP_TIME,
    "arg1": {"@type": "sleeper_func", "sleep_time": SLEEP_TIME},
    "arg2": {"@type": "sleeper_func", "sleep_time": SLEEP_TIME},
}


def test_sleeper_serial(register_sleeper_func):
    s = time.time()
    result = builder.build(spec=SLEEPER_SPEC, parallel=False)
    e = time.time()
    assert result is True
    assert e - s > SLEEP_TIME * 3


def test_sleeper_parallel(register_sleeper_func):
    s = time.time()
    result = builder.build(spec=SLEEPER_SPEC, parallel=True)
    e = time.time()
    assert result is True
    time_ellapsed = e - s
    assert time_ellapsed < SLEEP_TIME * 3


@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        "abc",
        {"k": "v"},
        ["a", 1, None],
        ("a", 1, None),
        {"int": 1, "str": "abc", "dict": {"yes": "sir"}},
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
        [{"a": ValueError}, ValueError],
        [{"@type": "something_not_registered"}, RuntimeError],
        [{"@type": "dummy_a", "a": 1, "@mode": "unsupported_mode_5566"}, ValueError],
    ],
)
def test_parse_exc(value, expected_exc, register_dummy_a):
    with pytest.raises(expected_exc):
        builder.build(value)


def test_register(register_dummy_a):
    assert builder.get_matching_entry("dummy_a").fn == DummyA


@pytest.mark.parametrize(
    "spec, expected",
    [
        [{"a": "b"}, {"a": "b"}],
        [{"@type": "dummy_a", "a": 2}, DummyA(a=2)],
        [{"k": {"@type": "dummy_a", "a": 2}}, {"k": DummyA(a=2)}],
        [{"k": {"kk": {"@type": "dummy_a", "a": 2}}}, {"k": {"kk": DummyA(a=2)}}],
        [[[{"@type": "dummy_a", "a": 2}]], [[DummyA(a=2)]]],
        [
            [[{"@type": "dummy_a", "a": 2, "@mode": "partial"}]],
            [[BuilderPartial(spec={"@type": "dummy_a", "a": 2})]],
        ],
        [{"@type": "dummy_b", "b": 2}, DummyB(b=2)],
        [{"@type": "dummy_b", "b": {"@type": "dummy_nonparallel"}}, DummyB(b=DummyNonparallel())],
        [{"k": {"@type": "dummy_nonparallel"}}, {"k": DummyNonparallel()}],
        [[{"@type": "dummy_nonparallel"}], [DummyNonparallel()]],
        [{"@type": "dummy_b", "b": [{"@type": "dummy_a", "a": 2}]}, DummyB(b=[DummyA(a=2)])],
        [
            {"@type": "dummy_a", "a": [{"@type": "dummy_b", "b": 3}]},
            DummyA(a=[DummyB(b=3)]),
        ],
        [
            {
                "@type": "dummy_a",
                "a": {"@type": "dummy_b", "b": 3},
            },
            DummyA(a=DummyB(b=3)),
        ],
        [
            {
                "key": [
                    *({"@type": "dummy_a", "a": i} for i in range(11)),
                    {"@type": "dummy_a", "@mode": "partial"},
                ]
            },
            {
                "key": [
                    *(DummyA(a=i) for i in range(11)),
                    BuilderPartial(spec={"@type": "dummy_a"}),
                ]
            },
        ],
    ],
)
def test_build_unversioned(
    spec: dict, expected: Any, register_dummy_a, register_dummy_b, register_dummy_nonparallel
):
    result_parallel = builder.build(spec, parallel=True)
    result_serial = builder.build(spec, parallel=False)
    assert result_parallel == expected
    assert result_serial == expected
    assert builder.get_initial_builder_spec(result_parallel) == spec
    assert builder.get_initial_builder_spec(result_serial) == spec


@pytest.mark.parametrize(
    "spec, expected",
    [
        [{"@type": "dummy_a", SPECIAL_KEYS["version"]: "0.0.0", "a": 2}, DummyA(a=2)],
        [
            {"@type": "dummy_a", SPECIAL_KEYS["version"]: "0.0.2", "a": 2},
            DummyAV2(a=2),
        ],
        [
            {
                "@type": "dummy_a",
                SPECIAL_KEYS["version"]: "0.0.2",
                "a": {"@type": "dummy_a", "a": 2},
            },
            DummyAV2(a=DummyAV2(a=2)),
        ],
    ],
)
def test_build_versioned(spec: dict, expected: Any, register_dummy_a_v0, register_dummy_a_v2):
    result_parallel = builder.build(spec, parallel=True)
    result_serial = builder.build(spec, parallel=False)
    assert result_parallel == expected
    assert result_serial == expected
    assert builder.get_initial_builder_spec(result_parallel) == spec
    assert builder.get_initial_builder_spec(result_serial) == spec


def test_build_partial_invocation(register_dummy_a, register_dummy_c):
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


def test_double_different_register_exc(register_dummy_a, register_dummy_a_v0):
    with pytest.raises(RuntimeError):
        builder.get_matching_entry("dummy_a", version="0.0.0")


def test_unpickleable_dict():
    obj = builder.UnpicklableDict()
    obj["bad"] = ProcessPoolExecutor()  # not pickleable
    recons = dill.loads(
        dill.dumps(
            obj, protocol=dill.DEFAULT_PROTOCOL, byref=False, recurse=True, fmode=dill.FILE_FMODE
        )
    )
    assert len(recons) == 0


def assert_file_exists(path):
    assert os.path.exists(path)


def write_file(path):
    with open(path, "w", encoding="utf8") as f:
        f.write("hello world")


def dummy_f(arg):
    ...


@pytest.fixture
def file_io_functions():
    builder.register("assert_file_exists", versions=">=0.0.0")(assert_file_exists)
    builder.register("write_file", versions=">=0.0.0")(write_file)
    builder.register("dummy_f", versions=">=0.0.0")(dummy_f)
    yield
    builder.unregister(name="dummy_f", fn=dummy_f)
    builder.unregister(name="write_file", fn=write_file)
    builder.unregister(name="assert_file_exists", fn=assert_file_exists)


def test_sequential_side_effects_x0(file_io_functions):
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "file.txt")
        builder.build(
            {
                "parts": [
                    {"@type": "write_file", "path": file_path},
                    {"@type": "assert_file_exists", "path": file_path},
                    [{"@type": "assert_file_exists", "path": file_path}],
                    [[{"@type": "assert_file_exists", "path": file_path}]],
                ]
            }
        )


def test_sequential_side_effects_x1(file_io_functions):
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "file.txt")
        builder.build(
            {
                "parts": [
                    [[[{"@type": "write_file", "path": file_path}]]],
                    {"@type": "assert_file_exists", "path": file_path},
                    [{"@type": "assert_file_exists", "path": file_path}],
                    [[{"@type": "assert_file_exists", "path": file_path}]],
                ]
            }
        )


def test_sequential_side_effects_x2(file_io_functions):
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "file.txt")
        builder.build(
            {
                "parts": [
                    [[{"inner": [{"@type": "write_file", "path": file_path}]}]],
                    {"@type": "assert_file_exists", "path": file_path},
                    [{"@type": "assert_file_exists", "path": file_path}],
                    [[{"@type": "assert_file_exists", "path": file_path}]],
                ]
            }
        )


def test_sequential_side_effects_x3(file_io_functions):
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "file.txt")
        builder.build(
            {
                "parts": [
                    {"@type": "dummy_f", "arg": [{"@type": "write_file", "path": file_path}]},
                    {"@type": "assert_file_exists", "path": file_path},
                ]
            }
        )


def test_sequential_side_effects_x4(file_io_functions):
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "file.txt")
        builder.build(
            {
                "parts": [
                    {"@type": "write_file", "path": file_path},
                    {
                        "@type": "dummy_f",
                        "arg": {"@type": "assert_file_exists", "path": file_path},
                    },
                ]
            }
        )
