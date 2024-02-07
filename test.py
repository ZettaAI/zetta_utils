# Joe's temporary test code for the new gap feature.
# Delete before making PR.
import os
import sys
from pprint import pprint  # pylint: disable=unused-import

from cloudvolume import CloudVolume as CV

import zetta_utils
import zetta_utils.mazepa
from zetta_utils.geometry import Vec3D

zetta_utils.log.set_verbosity("WARNING")
os.chdir("tests/integration")
spec = zetta_utils.parsing.cue.load("subchunkable/specs/test_gap.cue")

print(f"spec gap: {spec['target']['gap']}")

spec["target"]["bbox"]["start_coord"] = [0, 0, 0]
spec["target"]["bbox"]["end_coord"] = [2048, 2048, 5]
spec["target"]["print_summary"] = True  # works only when logging INFO
expectedShape = Vec3D[int](2560, 2560, 5)

pprint(spec)
print()
print("Building flow...")
flow = zetta_utils.builder.build(spec["target"])
print(f"Expected index shape: {expectedShape}")
actualShape = flow.args[0].shape  # flow.args[0] is the VolumetricIndex
print(f"  Actual index shape: {actualShape}")
if expectedShape == actualShape:
    print("OK!")
else:
    print("Oops.  Expected expansion did not occur.")
    sys.exit()

ans = input("Run flow [Y/n]? ").lower()
if not ans in ("", "y"):
    sys.exit()

print("Running flow...")
zetta_utils.mazepa.execute(flow)

print()
print("Validation:")
vol = CV("file://assets/outputs/test_gap", mip=[4, 4, 40])
print(f"volume shape: {vol.shape}")
print(f"Entry at 5,5,2: {vol[5,5,2].flatten()}")
print(f"Entry at 1020,5,2: {vol[1020,5,2].flatten()}")
print(f"Entry at 1040,5,2: {vol[1040,5,2].flatten()}")  # should be null or zero


def check_row(row_y: int):
    print(f"Contents of row {row_y} (z=2):")
    row = vol[0:3072, row_y, 2].flatten()
    prev_val = row[0]
    prev_start = 0
    for x in range(1, len(row)):
        if row[x] != prev_val:
            print(f"{prev_start}-{x} (a span of {x-prev_start}): value = {prev_val}")
            prev_val = row[x]
            prev_start = x
    print(f"{prev_start}-{len(row)-1}: {prev_val}")


check_row(100)
# check_row(500)
# check_row(1000)
# check_row(1500)
