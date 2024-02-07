import sys
from typing import Sequence

import zetta_utils
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.mazepa_layer_processing.common.subchunkable_apply_flow import (
    _expand_bbox_processing,
)

zetta_utils.log.set_verbosity("DEBUG")


def test_one(src_size: Sequence[int], gap: Sequence[int], expected_result: Sequence[int]):
    print()
    print("-" * 70)
    chunk_sizes = [Vec3D[int](1024, 1024, 5), Vec3D[int](512, 512, 3), Vec3D[int](256, 256, 1)]
    bbox = BBox3D(bounds=((0, src_size[0]), (0, src_size[1]), (0, src_size[2])))
    result = _expand_bbox_processing(bbox, Vec3D[int](1, 1, 1), chunk_sizes, Vec3D[int](*gap))
    print(f"Expanded {bbox.shape} with {gap} gap to: {result.shape}")
    if result.shape == Vec3D[int](*expected_result):
        print("OK!")
    else:
        print(f"INCORRECT RESULT: should be {expected_result}")
        sys.exit()


# existing cases: no gap, so expand to chunk boundary
test_one((1000, 1000, 5), (0, 0, 0), (1024, 1024, 5))
test_one((2000, 2000, 5), (0, 0, 0), (2048, 2048, 5))

# with a gap, expand instead to fit chunks + gaps
test_one((2048, 2048, 5), (512, 512, 0), (2560, 2560, 5))

# this one's already fine, so should not expand
test_one((2560, 2560, 5), (512, 512, 0), (2560, 2560, 5))

# more examples from issue #640
test_one((3000, 3000, 5), (512, 512, 0), (4096, 4096, 5))
