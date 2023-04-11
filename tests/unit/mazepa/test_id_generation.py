from __future__ import annotations

from functools import partial

import attrs

from zetta_utils.mazepa import taskable_operation_cls
from zetta_utils.mazepa.id_generation import generate_invocation_id as gen_id


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
