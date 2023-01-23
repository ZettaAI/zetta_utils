# pylint: disable=all
import typing

import numpy as np
import pytest

from zetta_utils import builder
from zetta_utils.geometry.vec import (
    IntVec1D,
    IntVec2D,
    IntVec3D,
    IntVecND,
    Vec1D,
    Vec2D,
    Vec3D,
    VecND,
)

some_float = 42.42
some_int = 42
vec2d = Vec2D(1, 2)
vec2d_neg = Vec2D(-1, -2)
vec2d_lg = Vec2D(1.5, 2.5)
vec2d_mx = Vec2D(0.5, 2.5)
intvec2d = IntVec2D(1, 2)
vec3d = Vec3D(1.0, 2.0, 3.0)
intvec3d = IntVec3D(1, 2, 3)
intvec3d_from_float = vec3d.int()
vec3d_init_int = Vec3D(1, 2, 3)
vec3d_from_int = intvec3d.float()
vec3d_diff = Vec3D(4.0, 5.0, 6.0)


@pytest.mark.parametrize("arg, val", [[vec2d, 2], [vec3d, 3], [vec3d_diff, 3]])
def test_len(arg, val):
    assert len(arg) == val


@pytest.mark.parametrize("arg, ind, val", [[vec2d, 1, 2], [vec3d, 2, 3], [vec3d_diff, 0, 4]])
def test_indexing(arg, ind, val):
    assert arg[ind] == val


@pytest.mark.parametrize(
    "arg1, arg2, ind1, ind2",
    [
        [vec2d, vec3d, slice(0, 2), slice(0, 2)],
        [vec3d_diff, tuple(range(4, 6)), slice(0, 1), slice(0, 1)],
        [vec3d, tuple(range(3)), slice(0, 1), slice(1, 2)],
    ],
)
def test_slicing(arg1, arg2, ind1, ind2):
    assert arg1[ind1] == arg2[ind2]


@pytest.mark.parametrize(
    "arg, single, slc, dtype",
    [
        [vec3d, 1, slice(0, 2), float],
        [intvec3d, 2, slice(0, 2), int],
        [vec3d_from_int, 0, slice(0, 1), float],
        [vec3d_init_int, 1, slice(1, 2), float],
        [intvec3d_from_float, 0, slice(0, 2), int],
    ],
)
def test_index_type(arg, single, slc, dtype):
    assert type(arg[single]) == dtype
    for e in arg[slc]:
        assert type(e) == dtype


@pytest.mark.parametrize(
    "arg, val",
    [
        [vec3d, (1.0, 2.0, 3.0)],
        [intvec3d, (1, 2, 3)],
        [vec3d_from_int, (1.0, 2.0, 3.0)],
        [vec3d_init_int, (1.0, 2.0, 3.0)],
        [vec3d_diff, (4.0, 5.0, 6.0)],
        [intvec3d_from_float, (1, 2, 3)],
    ],
)
def test_iter(arg, val):
    for e, v in zip(arg, val):
        assert type(e) == type(v)
        assert e == v


@pytest.mark.parametrize(
    "arg1, arg2, is_equal",
    [
        [vec2d, [1, 2, 3], False],
        [vec2d, vec2d, True],
        [vec2d, intvec2d, True],
        [vec2d, vec3d, False],
        [vec2d, intvec3d, False],
        [vec2d, vec3d_diff, False],
        [intvec2d, vec3d, False],
        [intvec2d, intvec3d, False],
        [intvec2d, vec3d_diff, False],
        [vec3d, intvec3d, True],
        [vec3d, vec3d_diff, False],
        [intvec3d, vec3d_diff, False],
        [vec3d, vec3d_from_int, True],
        [intvec3d, vec3d_from_int, True],
        [intvec3d, intvec3d_from_float, True],
        [intvec3d_from_float, vec3d_from_int, True],
    ],
)
def test_eq(arg1, arg2, is_equal):
    assert (arg1 == arg2) == is_equal
    assert (arg2 == arg1) == is_equal


@pytest.mark.parametrize(
    "arg1, arg2, is_equal",
    [
        [vec2d, intvec2d, False],
        [vec2d, vec3d, True],
        [vec3d, intvec3d, False],
        [vec3d, vec3d_init_int, True],
        [vec3d, vec3d_from_int, True],
        [intvec3d, intvec3d_from_float, True],
    ],
)
def test_eq_types(arg1, arg2, is_equal):
    assert (arg1.dtype == arg2.dtype) == is_equal


@pytest.mark.parametrize(
    "arg1, arg2, is_lt",
    [
        [vec2d, vec2d_lg, True],
        [vec2d, vec2d, False],
        [vec2d_lg, vec2d, False],
        [vec2d, vec2d_mx, False],
        [vec2d_mx, vec2d, False],
    ],
)
def test_lt(arg1, arg2, is_lt):
    assert (arg1 < arg2) == is_lt


@pytest.mark.parametrize(
    "arg1, arg2, is_le",
    [
        [vec2d, vec2d_lg, True],
        [vec2d, vec2d, True],
        [vec2d_lg, vec2d, False],
        [vec2d, vec2d_mx, False],
        [vec2d_mx, vec2d, False],
    ],
)
def test_le(arg1, arg2, is_le):
    assert (arg1 <= arg2) == is_le


@pytest.mark.parametrize(
    "arg1, arg2, is_gt",
    [
        [vec2d, vec2d_lg, False],
        [vec2d, vec2d, False],
        [vec2d_lg, vec2d, True],
        [vec2d, vec2d_mx, False],
        [vec2d_mx, vec2d, False],
    ],
)
def test_gt(arg1, arg2, is_gt):
    assert (arg1 > arg2) == is_gt


@pytest.mark.parametrize(
    "arg1, arg2, is_ge",
    [
        [vec2d, vec2d_lg, False],
        [vec2d, vec2d, True],
        [vec2d_lg, vec2d, True],
        [vec2d, vec2d_mx, False],
        [vec2d_mx, vec2d, False],
    ],
)
def test_ge(arg1, arg2, is_ge):
    assert (arg1 >= arg2) == is_ge


@pytest.mark.parametrize(
    "arg",
    [
        [vec2d, vec2d_neg],
    ],
)
def test_neg(arg):
    assert -vec2d == vec2d_neg
    assert vec2d == -vec2d_neg


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
        arg2_np = arg2
    else:
        arg2_np = np.array(arg2[:])
    np_val = eval(f"arg1_np {fname} arg2_np")
    np_val_r = eval(f"arg2_np {fname} arg1_np")
    assert zt_val.dtype == dtype
    assert zt_val_r.dtype == dtype
    assert (np_val == np.array(zt_val[:])).all()
    assert (np_val_r == np.array(zt_val_r[:])).all()


@pytest.mark.parametrize(
    "arg1, arg2, fname",
    [
        [vec3d, None, "+"],
        [None, vec3d, "-"],
        [vec3d, "hi", "*"],
        ["hi", vec3d, "/"],
        ["hi", vec3d, "//"],
    ],
)
def test_unimplemented_ops(arg1, arg2, fname):
    with pytest.raises(NotImplementedError):
        eval(f"arg1 {fname} arg2")


@pytest.mark.parametrize(
    "constructor, args, expected_exc",
    [
        [VecND, (1, 2, 3), TypeError],
        [IntVecND, (1, 2, 3), TypeError],
        [Vec3D, (1, 2, 3, 4), ValueError],
        [IntVec3D, (1.5, 2.5, 3.5), TypeError],
    ],
)
def test_exc(constructor, args, expected_exc):
    with pytest.raises(expected_exc):
        constructor(*args)


@pytest.mark.parametrize(
    "constructor, args, expected_exc",
    [
        [Vec3D, (1, 2, 3, 4), ValueError],
        [IntVec3D, (1, 2, 3, 4), ValueError],
        [Vec1D, (1, 2, 3), TypeError],
        [IntVec1D, (1, 2, 3), TypeError],
    ],
)
def test_exc_tuple(constructor, args, expected_exc):
    with pytest.raises(expected_exc):
        constructor(args)


@pytest.mark.parametrize(
    "vec, idx, val",
    [
        [vec3d, 1, 4.5],
        [intvec3d, 0, 2],
    ],
)
def test_setitem(vec, idx, val):
    vec[idx] = val
    assert vec[idx] == val


@pytest.mark.parametrize(
    "vec, idx, val, expected_exc",
    [
        [intvec3d, 10, 1.0, IndexError],
        [intvec3d, 0, 1.5, TypeError],
    ],
)
def test_exc_setitem(vec, idx, val, expected_exc):
    with pytest.raises(expected_exc):
        vec[idx] = val


vec1d = Vec1D(1.0)
intvec1d = IntVec1D(1)
"""
The following tests test that the type hierarchy is correct and the type inference works
correctly using mypy; we cannot use fixtures since the variables need to be statically typed.
Furthermore, we cannot test whether a Vec fails to be an IntVec since mypy will throw an error
meaning that the CI / pre-commit tests will not pass.

Tests are contained in a dummy function since calling a function defined in the module scope
defaults to using fixtures.

Only the 1d case is tested since the parametrisations follow the same pattern.
"""


def test_subtyping() -> None:
    def test_subtyping_int1d_is_vec1d(x: Vec1D):
        pass

    test_subtyping_int1d_is_vec1d(intvec1d)

    def test_subtyping_int1d_is_intnd(x: IntVecND):
        pass

    test_subtyping_int1d_is_intnd(intvec1d)

    def test_subtyping_int1d_is_vecnd(x: VecND):
        pass

    test_subtyping_int1d_is_vecnd(intvec1d)

    def test_subtyping_vec1d_is_vecnd(x: VecND):
        pass

    test_subtyping_vec1d_is_vecnd(vec1d)

    def test_subtyping_cast_to_int(x: IntVec1D):
        pass

    test_subtyping_cast_to_int(vec1d.int())

    def test_subtyping_cast_to_float(x: Vec1D):
        pass

    test_subtyping_cast_to_float(intvec1d.float())

    def test_inference_return_int1d(x: IntVec1D):
        pass

    # using a list for brevity - if one of them fails the entire line will fail
    for x in [
        intvec1d + intvec1d,
        intvec1d - intvec1d,
        intvec1d * intvec1d,
        intvec1d // intvec1d,
        intvec1d % intvec1d,
        intvec1d + some_int,
        intvec1d - some_int,
        intvec1d * some_int,
        intvec1d // some_int,
        intvec1d % some_int,
        some_int + intvec1d,
        some_int - intvec1d,
        some_int * intvec1d,
        some_int // intvec1d,
        some_int % intvec1d,
    ]:

        test_inference_return_int1d(x)

    def test_inference_return_vec1d(x: Vec1D):
        pass

    for y in [
        vec1d + intvec1d,
        vec1d - intvec1d,
        vec1d * intvec1d,
        vec1d // intvec1d,
        vec1d / intvec1d,
        vec1d % intvec1d,
        intvec1d + vec1d,
        intvec1d - vec1d,
        intvec1d * vec1d,
        intvec1d // vec1d,
        intvec1d / vec1d,
        vec1d % intvec1d,
        vec1d + some_int,
        vec1d - some_int,
        vec1d * some_int,
        vec1d // some_int,
        vec1d / some_int,
        vec1d % some_int,
        some_int + vec1d,
        some_int - vec1d,
        some_int * vec1d,
        some_int // vec1d,
        some_int / vec1d,
        some_int % vec1d,
        vec1d + some_float,
        vec1d - some_float,
        vec1d * some_float,
        vec1d // some_float,
        vec1d / some_float,
        vec1d % some_float,
        some_float + vec1d,
        some_float - vec1d,
        some_float * vec1d,
        some_float // vec1d,
        some_float / vec1d,
        some_float % vec1d,
        intvec1d + some_float,
        intvec1d - some_float,
        intvec1d * some_float,
        intvec1d // some_float,
        intvec1d / some_float,
        intvec1d / some_int,
        some_float + intvec1d,
        some_float - intvec1d,
        some_float * intvec1d,
        some_float // intvec1d,
        some_float / intvec1d,
        some_float % intvec1d,
    ]:
        test_inference_return_vec1d(y)


def test_builder_autoconvert():
    built = builder.build({"a": [1, 2, 3], "b": [1.1, 2, 3], "c": [1, 2, 3, 4]})
    expected = {"a": IntVec3D(1, 2, 3), "b": Vec3D(1.1, 2, 3), "c": [1, 2, 3, 4]}
    assert built == expected
