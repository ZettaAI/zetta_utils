from __future__ import annotations

import base64
import os
import pickle
import subprocess
import sys
from functools import partial
from typing import Any, Callable, Mapping

import attrs

from zetta_utils import mazepa
from zetta_utils.builder.building import BuilderPartial
from zetta_utils.geometry.bbox import BBox3D
from zetta_utils.layer.volumetric.cloudvol.build import build_cv_layer
from zetta_utils.mazepa import taskable_operation_cls
from zetta_utils.mazepa.id_generation import generate_invocation_id as gen_id
from zetta_utils.mazepa_layer_processing.common import build_subchunkable_apply_flow


class ClassA:
    @staticmethod
    def method(a, b):
        return a + b


class ClassB:
    @staticmethod
    def method(a, b):
        return a * b


class ClassC:
    @staticmethod
    def method(a, b):
        return a * b


class ClassD1:
    a = 1

    @classmethod
    def method(cls, b):
        return cls.a * b


class ClassD2:
    a = 2

    @classmethod
    def method(cls, b):
        return cls.a * b


class ClassE:
    def __init__(self, a):
        self.a = a

    def method(self, b):
        return self.a * b

    @property
    def prop(self):
        return self.a


@taskable_operation_cls()
@attrs.mutable
class TaskableA:
    def __call__(self, a, b):
        return a + b


@taskable_operation_cls
@attrs.mutable
class TaskableB:
    def __call__(self, a, b):
        return a * b


@taskable_operation_cls
@attrs.mutable
class TaskableC:
    def __call__(self, x, y):
        return x * y


@taskable_operation_cls
@attrs.mutable
class TaskableD:
    a: int

    def __call__(self, b):
        return self.a * b


@mazepa.flow_schema_cls
@attrs.mutable
class FlowSchema:
    fn_kwargs: Mapping[Any, Any]
    callable_fn: Callable[..., Any]

    def __init__(self, fn_kwargs: Mapping[Any, Any], callable_fn: Callable[..., Any]):
        self.fn_kwargs = fn_kwargs
        self.callable_fn = callable_fn

    def flow(self, *args, **kwargs):
        return self.callable_fn(*args, **kwargs)


def subchunkable_flow():
    # Subchunkable is used so commonly that it warrants its own test
    return build_subchunkable_apply_flow(
        fn=BuilderPartial(spec={"@type": "invoke_lambda_str", "lambda_str": "lambda x: x"}),
        dst_resolution=[1, 1, 1],
        processing_chunk_sizes=[[1, 1, 1]],
        dst=build_cv_layer(
            path="/tmp/zutils/test/test_id_generation",
            info_type="image",
            info_data_type="int8",
            info_num_channels=1,
            info_scales=[[1, 1, 1]],
            info_chunk_size=[1, 1, 1],
            info_encoding="raw",
            info_bbox=BBox3D.from_coords(start_coord=[0, 0, 0], end_coord=[1, 1, 1]),
        ),
        bbox=BBox3D.from_coords(start_coord=[0, 0, 0], end_coord=[1, 1, 1], resolution=[1, 1, 1]),
        level_intermediaries_dirs=[
            "/tmp/zutils/test/test_id_generation/tmp",
        ],
    )


def test_generate_invocation_id_method() -> None:
    assert gen_id(ClassA().method, [], {}) != gen_id(ClassB().method, [], {})
    assert gen_id(ClassB().method, [], {}) != gen_id(ClassC().method, [], {})

    assert gen_id(ClassA().method, [4, 2], {}) == gen_id(ClassA().method, [4, 2], {})
    assert gen_id(ClassA().method, [], {"a": 1}) == gen_id(ClassA().method, [], {"a": 1})

    assert gen_id(ClassA().method, [4, 2], {}) != gen_id(ClassA().method, [6, 3], {})
    assert gen_id(ClassA().method, [], {"a": 1}) != gen_id(ClassA().method, [], {"a": 2})

    assert gen_id(ClassD1().method, [], {}) != gen_id(ClassD2().method, [], {})

    assert gen_id(ClassE(1).method, [], {}) != gen_id(ClassE(2).method, [], {})


def test_generate_invocation_id_partial() -> None:
    partial_a = partial(ClassB().method, 42)
    partial_b = partial(ClassB().method, 21)
    partial_c = partial(ClassC().method, 21)
    partial_d1 = partial(ClassD1().method, 42)
    partial_d2 = partial(ClassD2().method, 42)
    partial_e1 = partial(ClassE(1).method, 42)
    partial_e2 = partial(ClassE(2).method, 42)

    assert gen_id(partial_a, [], {}) != gen_id(partial_b, [], {})
    assert gen_id(partial_b, [], {}) != gen_id(partial_c, [], {})

    assert gen_id(partial_a, [4, 2], {}) == gen_id(partial_a, [4, 2], {})
    assert gen_id(partial_a, [], {"a": 1}) == gen_id(partial_a, [], {"a": 1})

    assert gen_id(partial_a, [4, 2], {}) != gen_id(partial_a, [6, 3], {})
    assert gen_id(partial_a, [], {"a": 1}) != gen_id(partial_a, [], {"a": 2})

    assert gen_id(partial_d1, [], {}) != gen_id(partial_d2, [], {})

    assert gen_id(partial_e1, [], {}) != gen_id(partial_e2, [], {})


def test_generate_invocation_id_taskable_op() -> None:
    assert gen_id(TaskableA(), [], {}) != gen_id(TaskableB(), [], {})
    assert gen_id(TaskableB(), [], {}) != gen_id(TaskableC(), [], {})

    assert gen_id(TaskableA(), [4, 2], {}) == gen_id(TaskableA(), [4, 2], {})
    assert gen_id(TaskableA(), [], {"a": 1}) == gen_id(TaskableA(), [], {"a": 1})

    assert gen_id(TaskableA(), [4, 2], {}) != gen_id(TaskableA(), [6, 3], {})
    assert gen_id(TaskableA(), [], {"a": 1}) != gen_id(TaskableA(), [], {"a": 2})

    assert gen_id(TaskableD(1), [], {}) == gen_id(TaskableD(1), [], {})
    assert gen_id(TaskableD(1), [], {}) != gen_id(TaskableD(2), [], {})


def test_generate_invocation_id_flow_schema() -> None:
    assert gen_id(FlowSchema({}, ClassA().method).flow, [], {}) != gen_id(
        FlowSchema({}, ClassB().method).flow, [], {}
    )
    assert gen_id(FlowSchema({}, ClassB().method).flow, [], {}) != gen_id(
        FlowSchema({}, ClassC().method).flow, [], {}
    )

    assert gen_id(FlowSchema({}, ClassA().method).flow, [4, 2], {}) == gen_id(
        FlowSchema({}, ClassA().method).flow, [4, 2], {}
    )
    assert gen_id(FlowSchema({}, ClassA().method).flow, [], {"a": 1}) == gen_id(
        FlowSchema({}, ClassA().method).flow, [], {"a": 1}
    )

    assert gen_id(FlowSchema({}, ClassA().method).flow, [4, 2], {}) != gen_id(
        FlowSchema({}, ClassA().method).flow, [6, 3], {}
    )
    assert gen_id(FlowSchema({}, ClassA().method).flow, [], {"a": 1}) != gen_id(
        FlowSchema({}, ClassA().method).flow, [], {"a": 2}
    )

    assert gen_id(FlowSchema({}, ClassD1().method).flow, [], {}) != gen_id(
        FlowSchema({}, ClassD2().method).flow, [], {}
    )
    assert gen_id(FlowSchema({}, ClassE(1).method).flow, [], {}) != gen_id(
        FlowSchema({}, ClassE(2).method).flow, [], {}
    )


def test_generate_invocation_id_subchunkable_flow() -> None:
    a = subchunkable_flow()
    b = subchunkable_flow()
    assert gen_id(a.fn, a.args, a.kwargs) == gen_id(b.fn, b.args, b.kwargs)


def _gen_id_calls(_) -> dict[str, str]:
    gen_ids = {
        'gen_id(ClassA().method, [], {"a": 1})': gen_id(ClassA().method, [], {"a": 1}),
        "gen_id(ClassD1().method, [], {})": gen_id(ClassD1().method, [], {}),
        "gen_id(ClassE(1).method, [], {})": gen_id(ClassE(1).method, [], {}),
        "gen_id(partial(ClassA().method, 42), [], {})": gen_id(
            partial(ClassA().method, 42), [], {}
        ),
        "gen_id(partial(ClassD1().method, 42), [], {})": gen_id(
            partial(ClassD1().method, 42), [], {}
        ),
        "gen_id(partial(ClassE(1).method, 42), [], {})": gen_id(
            partial(ClassE(1).method, 42), [], {}
        ),
        "gen_id(TaskableA(), [], {})": gen_id(TaskableA(), [], {}),
        "gen_id(TaskableD(1), [], {})": gen_id(TaskableD(1), [], {}),
        "gen_id(FlowSchema({}, ClassA().method).flow, [], {})": gen_id(
            FlowSchema({}, ClassA().method).flow, [], {}
        ),
        "gen_id(FlowSchema({}, ClassD1().method).flow, [], {})": gen_id(
            FlowSchema({}, ClassD1().method).flow, [], {}
        ),
        "gen_id(FlowSchema({}, ClassE(1).method).flow, [], {})": gen_id(
            FlowSchema({}, ClassE(1).method).flow, [], {}
        ),
        "gen_id(subchunkable_flow(), [], {})": gen_id(
            subchunkable_flow().fn, subchunkable_flow().args, subchunkable_flow().kwargs
        ),
    }
    return gen_ids


def test_persistence_across_sessions() -> None:
    # Create two separate processes with different PYTHONHASHSEED values
    # to ensure ID generation is consistent across different hash seeds.
    # This workaround is necessary due to fork being the default
    # multiprocessing method, meaning we cannot use 'spawn' method to test.

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    script = f"""
import pickle
import sys
import base64

# Add the project root to path so we can import the test module
sys.path.insert(0, "{project_root}")

# Import the test function and call it
from tests.unit.mazepa.test_id_generation import _gen_id_calls
result = _gen_id_calls(0)

# Output as base64 encoded pickle
encoded_result = base64.b64encode(pickle.dumps(result)).decode('ascii')
print(encoded_result)
    """

    results = []
    for seed in ["0", "42"]:  # Different hash seeds
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed

        # Run subprocess with different PYTHONHASHSEED
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        if proc.returncode != 0:
            raise RuntimeError(f"Subprocess failed with seed {seed}: {proc.stderr}")

        result_data = base64.b64decode(proc.stdout.strip())
        result = pickle.loads(result_data)
        results.append(result)

    assert results[0] == results[1]


def test_unpickleable_fn(mocker) -> None:
    # See https://github.com/uqfoundation/dill/issues/147 and possibly
    # https://github.com/uqfoundation/dill/issues/56

    unpickleable_fn = mocker.MagicMock()
    # gen_id will return a random UUID in case of pickle errors
    assert gen_id(unpickleable_fn, [], {}) != gen_id(unpickleable_fn, [], {})
