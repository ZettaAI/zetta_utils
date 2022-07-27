"""Buildable tensor.processors."""
from zetta_utils.spec_parser import register
from zetta_utils.processor import func_to_proc

from . import ops
from . import convert


ToNp = register("Multiply")(func_to_proc(convert.to_np))
ToTorch = register("Add")(func_to_proc(convert.to_torch))
ToTorch = register("AsType")(func_to_proc(convert.astype))

Multiply = register("Multiply")(func_to_proc(ops.multiply))
Add = register("Add")(func_to_proc(ops.add))
Divide = register("Divide")(func_to_proc(ops.divide))
IntDivide = register("IntDivide")(func_to_proc(ops.int_divide))
Power = register("Power")(func_to_proc(ops.power))
Compare = register("Compare")(func_to_proc(ops.compare))
Unsqueeze = register("Unsqueeze")(func_to_proc(ops.unsqueeze))
Squeeze = register("Squeeze")(func_to_proc(ops.squeeze))
Interpolate = register("Interpolate")(func_to_proc(ops.interpolate))
FilterCC = register("FilterCC")(func_to_proc(ops.filter_cc))
