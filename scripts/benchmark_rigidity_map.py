import math
import timeit

import torch
import torchfields

from zetta_utils.alignment import field


def rotation_tensor(degrees):
    rads = math.radians(degrees)
    return torch.Tensor(
        [[math.cos(rads), -math.sin(rads), 0], [math.sin(rads), math.cos(rads), 0]]
    )


# pylint: disable=invalid-name
displacement_field = None
weight_map = None


def setup(size=2048):
    # Create a displacement field, a 45° rotation, with one extreme element
    fld = torchfields.Field.affine_field(rotation_tensor(45), size=(1, 2, size, size))
    fld[0, 0, 7, 11] = 5
    fld[0, 1, 7, 11] = -1

    # Create a weight map, all ones except for a zero at the location of the extreme vector
    wm = torch.ones(1, size, size)
    wm[0, 7, 11] = 0

    return fld, wm


def test_no_weights():
    assert displacement_field is not None, "displacement_field must be a Tensor"
    field.get_rigidity_map_zcxy(displacement_field, weight_map=None)  # type: ignore


def test_with_weights():
    assert displacement_field is not None, "displacement_field must be a Tensor"
    field.get_rigidity_map_zcxy(displacement_field, weight_map=weight_map)  # type: ignore


def run_benchmark():
    # Number of times to run the function for the benchmark
    number_of_executions = 100

    print("-----------------------------------------------")
    print(f"Starting benchmark on device: {device}")

    # Use timeit.timeit to benchmark
    execution_time = timeit.timeit(
        "test_no_weights()", globals=globals(), number=number_of_executions
    )
    print(
        f"WITHOUT WEIGHTS: Mean time over {number_of_executions} \
            runs: {execution_time/number_of_executions} seconds"
    )

    execution_time = timeit.timeit(
        "test_with_weights()", globals=globals(), number=number_of_executions
    )
    print(
        f"   WITH WEIGHTS: Mean time over {number_of_executions} \
            runs: {execution_time/number_of_executions} seconds"
    )


displacement_field, weight_map = setup(2048)
device = "cpu"
run_benchmark()
if torch.cuda.is_available():
    device = "cuda"
    assert displacement_field is not None, "displacement_field is a Tensor at this point"
    assert weight_map is not None, "weight_map is a Tensor at this point"
    displacement_field = displacement_field.to(device)
    weight_map = weight_map.to(device)
    run_benchmark()
else:
    print("(No GPU available.)")
