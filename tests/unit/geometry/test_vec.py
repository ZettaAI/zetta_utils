# pylint: disable=all
import typing
from math import ceil, floor, trunc

import numpy as np
import pytest
import typeguard

from zetta_utils import builder
from zetta_utils.geometry.vec import Vec3D, allclose, isclose

some_float = 42.42
some_int = 42

vec3d = Vec3D(1.0, 2.0, 3.0)
vec3d_fp1 = Vec3D(0.1, 0.6 / 3.0, 0.1 + 0.2)
vec3d_fp2 = Vec3D(0.1, 0.2, 0.3)
vec3d_neg = Vec3D(-1.0, -2.0, -3.0)
vec3d_lg = Vec3D(1.5, 2.5, 3.5)
vec3d_mx = Vec3D(0.5, 2.5, 3.0)
intvec3d = Vec3D[int](1, 2, 3)

intvec3d_from_float = vec3d.int()
vec3d_from_int = intvec3d.float()
vec3d_diff = Vec3D(4.0, 5.0, 6.0)


@pytest.mark.parametrize("arg, val", [[vec3d, 3], [vec3d_diff, 3]])
def test_len(arg, val):
    assert len(arg) == val


@pytest.mark.parametrize("arg, ind, val", [[vec3d, 2, 3], [vec3d_diff, 0, 4]])
def test_indexing(arg, ind, val):
    assert arg[ind] == val


@pytest.mark.parametrize(
    "arg1, arg2, ind1, ind2",
    [
        # [vec2d, vec3d, slice(0, 2), slice(0, 2)],
        [vec3d_diff, tuple(range(4, 6)), slice(0, 1), slice(0, 1)],
        [vec3d, tuple(range(3)), slice(0, 1), slice(1, 2)],
    ],
)
def test_slicing(arg1, arg2, ind1, ind2):
    assert arg1[ind1] == arg2[ind2]


@pytest.mark.parametrize(
    "arg, val",
    [
        [vec3d, (1.0, 2.0, 3.0)],
        [intvec3d, (1, 2, 3)],
        [vec3d_from_int, (1.0, 2.0, 3.0)],
        [vec3d_diff, (4.0, 5.0, 6.0)],
        [intvec3d_from_float, (1, 2, 3)],
    ],
)
def test_iter(arg, val):
    for e, v in zip(arg, val):
        assert e == v


@pytest.mark.parametrize(
    "arg1, arg2, is_equal",
    [
        # [vec2d, [1, 2, 3], False],
        # [vec2d, vec2d, True],
        # [vec2d, intvec2d, True],
        # [vec2d, vec3d, False],
        # [vec2d, intvec3d, False],
        # [vec2d, vec3d_diff, False],
        # [intvec2d, vec3d, False],
        # [intvec2d, intvec3d, False],
        # [intvec2d, vec3d_diff, False],
        [vec3d, intvec3d, True],
        [vec3d, vec3d_diff, False],
        [intvec3d, vec3d_diff, False],
        [vec3d, vec3d_from_int, True],
        [intvec3d, vec3d_from_int, True],
        [intvec3d, intvec3d_from_float, True],
        [intvec3d_from_float, vec3d_from_int, True],
        [intvec3d_from_float, None, False],
    ],
)
def test_eq(arg1, arg2, is_equal):
    assert (arg1 == arg2) == is_equal
    assert (arg2 == arg1) == is_equal


@pytest.mark.parametrize(
    "arg1, arg2, is_equal",
    [
        [vec3d, intvec3d, False],
        [vec3d, vec3d_from_int, True],
        [intvec3d, intvec3d_from_float, True],
    ],
)
def test_eq_types(arg1, arg2, is_equal):
    assert (type(arg1[0]) == type(arg2[0])) == is_equal


@pytest.mark.parametrize(
    "arg1, arg2, is_equal",
    [
        [vec3d, intvec3d, True],
        [vec3d, vec3d_from_int, True],
        [intvec3d, intvec3d_from_float, True],
        [intvec3d, vec3d_diff, False],
    ],
)
def test_hash(arg1, arg2, is_equal):
    assert (arg1[0].__hash__() == arg2[0].__hash__()) == is_equal


@pytest.mark.parametrize(
    "arg1, arg2, is_lt",
    [
        [vec3d, vec3d_lg, True],
        [vec3d, vec3d, False],
        [vec3d_lg, vec3d, False],
        [vec3d, vec3d_mx, False],
        [vec3d_mx, vec3d, False],
    ],
)
def test_lt(arg1, arg2, is_lt):
    assert (arg1 < arg2) == is_lt


@pytest.mark.parametrize(
    "arg1, arg2, is_le",
    [
        [vec3d, vec3d_lg, True],
        [vec3d, vec3d, True],
        [vec3d_lg, vec3d, False],
        [vec3d, vec3d_mx, False],
        [vec3d_mx, vec3d, False],
    ],
)
def test_le(arg1, arg2, is_le):
    assert (arg1 <= arg2) == is_le


@pytest.mark.parametrize(
    "arg1, arg2, is_gt",
    [
        [vec3d, vec3d_lg, False],
        [vec3d, vec3d, False],
        [vec3d_lg, vec3d, True],
        [vec3d, vec3d_mx, False],
        [vec3d_mx, vec3d, False],
    ],
)
def test_gt(arg1, arg2, is_gt):
    assert (arg1 > arg2) == is_gt


@pytest.mark.parametrize(
    "arg1, arg2, is_ge",
    [
        [vec3d, vec3d_lg, False],
        [vec3d, vec3d, True],
        [vec3d_lg, vec3d, True],
        [vec3d, vec3d_mx, False],
        [vec3d_mx, vec3d, False],
    ],
)
def test_ge(arg1, arg2, is_ge):
    assert (arg1 >= arg2) == is_ge


def test_neg():
    assert -vec3d == vec3d_neg
    assert vec3d == -vec3d_neg


def test_abs():
    assert abs(vec3d_neg) == vec3d
    assert abs(vec3d) == vec3d


def test_round():
    assert round(vec3d_fp1, 8) == vec3d_fp2


def test_floor():
    assert floor(Vec3D(0.5, -2.5, 3.0)) == Vec3D(0, -3, 3)


def test_trunc():
    assert trunc(Vec3D(0.5, -2.5, 3.0)) == Vec3D(0, -2, 3)


def test_ceil():
    assert ceil(Vec3D(0.5, -2.5, 3.0)) == Vec3D(1, -2, 3)


@pytest.mark.parametrize(
    "arg1, arg2, fname, dtype",
    [
        [vec3d, some_float, "+", float],
        [vec3d, some_int, "+", float],
        [vec3d, vec3d, "+", float],
        [vec3d, intvec3d, "+", float],
        [intvec3d, some_float, "+", float],
        [intvec3d, some_int, "+", int],
        [intvec3d, vec3d, "+", float],
        [intvec3d, intvec3d, "+", int],
        [vec3d, some_float, "-", float],
        [vec3d, some_int, "-", float],
        [vec3d, vec3d, "-", float],
        [vec3d, intvec3d, "-", float],
        [intvec3d, some_float, "-", float],
        [intvec3d, some_int, "-", int],
        [intvec3d, vec3d, "-", float],
        [intvec3d, intvec3d, "-", int],
        [vec3d, some_float, "*", float],
        [vec3d, some_int, "*", float],
        [vec3d, vec3d, "*", float],
        [vec3d, intvec3d, "*", float],
        [intvec3d, some_float, "*", float],
        [intvec3d, some_int, "*", int],
        [intvec3d, vec3d, "*", float],
        [intvec3d, intvec3d, "*", int],
        [vec3d, some_float, "/", float],
        [vec3d, some_int, "/", float],
        [vec3d, vec3d, "/", float],
        [vec3d, intvec3d, "/", float],
        [intvec3d, some_float, "/", float],
        [intvec3d, some_int, "/", float],
        [intvec3d, vec3d, "/", float],
        [intvec3d, intvec3d, "/", float],
        [vec3d, some_float, "//", float],
        [vec3d, some_int, "//", float],
        [vec3d, vec3d, "//", float],
        [vec3d, intvec3d, "//", float],
        [intvec3d, some_float, "//", float],
        [intvec3d, some_int, "//", int],
        [intvec3d, vec3d, "//", float],
        [intvec3d, intvec3d, "//", int],
        [vec3d, some_float, "%", float],
        [vec3d, some_int, "%", float],
        [vec3d, vec3d, "%", float],
        [vec3d, intvec3d, "%", float],
        [intvec3d, some_float, "%", float],
        [intvec3d, some_int, "%", int],
        [intvec3d, vec3d, "%", float],
        [intvec3d, intvec3d, "%", int],
    ],
)
def test_ops(arg1, arg2, fname, dtype):
    zt_val = eval(f"arg1 {fname} arg2")
    zt_val_r = eval(f"arg2 {fname} arg1")
    arg1_np = np.array(arg1[:])
    if isinstance(arg2, (float, int)):
        arg2_lit = arg2
        np_val = eval(f"arg1_np {fname} arg2_lit")
        np_val_r = eval(f"arg2_lit {fname} arg1_np")
    else:
        arg2_np = np.array(arg2[:])
        np_val = eval(f"arg1_np {fname} arg2_np")
        np_val_r = eval(f"arg2_np {fname} arg1_np")

    assert (np_val == np.array(zt_val[:])).all()
    assert (np_val_r == np.array(zt_val_r[:])).all()


@pytest.mark.parametrize(
    "arg1, arg2, fname, exc",
    [
        [vec3d, None, "+", typeguard.TypeCheckError],
        [None, vec3d, "-", TypeError],
        [vec3d, "hi", "*", typeguard.TypeCheckError],
        ["hi", vec3d, "/", typeguard.TypeCheckError],
        ["hi", vec3d, "//", TypeError],
    ],
)
def test_unimplemented_ops(arg1, arg2, fname, exc):
    with pytest.raises(exc):
        eval(f"arg1 {fname} arg2")


@pytest.mark.parametrize(
    "constructor, args, expected_exc",
    [
        [Vec3D, (1, 2, 3, 4), TypeError],
        # Will be fixed by: https://github.com/agronholm/typeguard/issues/21
        # [Vec3D[int], (1.5, 2.5, 3.5), TypeError],
    ],
)
def test_exc(constructor, args, expected_exc):
    with pytest.raises(expected_exc):
        constructor(*args)


@pytest.mark.parametrize(
    "constructor, args, expected_exc",
    [
        [Vec3D, (1, 2, 3, 4), TypeError],
        [Vec3D[int], (1, 2, 3, 4), TypeError],
    ],
)
def test_exc_tuple(constructor, args, expected_exc):
    with pytest.raises(expected_exc):
        constructor(args)


"""
The following tests test that the type hierarchy is correct and the type inference works
correctly using mypy; we cannot use fixtures since the variables need to be statically typed.
Furthermore, we cannot test whether a Vec fails to be an IntVec since mypy will throw an error
meaning that the CI / pre-commit tests will not pass.

Tests are contained in a dummy function since calling a function defined in the module scope
defaults to using fixtures.

Only the 3d case is tested since the parametrisations follow the same pattern.
"""


def test_subtyping() -> None:
    def test_subtyping_int3d_is_vec3d(x: Vec3D):
        pass

    test_subtyping_int3d_is_vec3d(intvec3d)

    def test_subtyping_int3d_is_intnd(x: Vec3D[int]):
        pass

    test_subtyping_int3d_is_intnd(intvec3d)

    def test_subtyping_int3d_is_vecnd(x: Vec3D):
        pass

    test_subtyping_int3d_is_vecnd(intvec3d)

    def test_subtyping_vec3d_is_vecnd(x: Vec3D):
        pass

    test_subtyping_vec3d_is_vecnd(vec3d)

    def test_subtyping_cast_to_int(x: Vec3D[int]):
        pass

    test_subtyping_cast_to_int(vec3d.int())

    def test_subtyping_cast_to_float(x: Vec3D):
        pass

    test_subtyping_cast_to_float(intvec3d.float())

    def test_inference_return_int3d(x: Vec3D[int]):
        pass

    # using a list for brevity - if one of them fails the entire line will fail
    for x in [
        intvec3d + intvec3d,
        intvec3d - intvec3d,
        intvec3d * intvec3d,
        intvec3d // intvec3d,
        intvec3d % intvec3d,
        intvec3d + some_int,
        intvec3d - some_int,
        intvec3d * some_int,
        intvec3d // some_int,
        intvec3d % some_int,
        some_int + intvec3d,
        some_int - intvec3d,
        some_int * intvec3d,
        some_int // intvec3d,
        some_int % intvec3d,
        abs(intvec3d),
        round(intvec3d),
        floor(intvec3d),
        trunc(intvec3d),
        ceil(intvec3d),
    ]:

        test_inference_return_int3d(x)

    def test_inference_return_vec3d(x: Vec3D):
        pass

    for y in [
        vec3d + intvec3d,
        vec3d - intvec3d,
        vec3d * intvec3d,
        vec3d // intvec3d,
        vec3d / intvec3d,
        vec3d % intvec3d,
        intvec3d + vec3d,
        intvec3d - vec3d,
        intvec3d * vec3d,
        intvec3d // vec3d,
        intvec3d / vec3d,
        vec3d % intvec3d,
        vec3d + some_int,
        vec3d - some_int,
        vec3d * some_int,
        vec3d // some_int,
        vec3d / some_int,
        vec3d % some_int,
        some_int + vec3d,
        some_int - vec3d,
        some_int * vec3d,
        some_int // vec3d,
        some_int / vec3d,
        some_int % vec3d,
        vec3d + some_float,
        vec3d - some_float,
        vec3d * some_float,
        vec3d // some_float,
        vec3d / some_float,
        vec3d % some_float,
        some_float + vec3d,
        some_float - vec3d,
        some_float * vec3d,
        some_float // vec3d,
        some_float / vec3d,
        some_float % vec3d,
        intvec3d + some_float,
        intvec3d - some_float,
        intvec3d * some_float,
        intvec3d // some_float,
        intvec3d / some_float,
        intvec3d / some_int,
        some_float + intvec3d,
        some_float - intvec3d,
        some_float * intvec3d,
        some_float // intvec3d,
        some_float / intvec3d,
        some_float % intvec3d,
        abs(vec3d),
        round(vec3d),
    ]:
        test_inference_return_vec3d(y)


def test_isclose():
    assert isclose(vec3d_fp1, vec3d_fp2) == (True, True, True)
    assert isclose(vec3d_fp1, vec3d_fp2, abs_tol=0.0, rel_tol=0.0) == (True, False, False)


def test_allclose():
    assert allclose(vec3d_fp1, vec3d_fp2) == True
    assert allclose(vec3d_fp1, vec3d_fp2, abs_tol=0.0, rel_tol=0.0) == False
