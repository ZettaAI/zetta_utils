"""Buildable tensor.processors."""
from zetta_utils.builder import register
from zetta_utils import tensor

from .utils import FuncProcessorBuilder

ToNp = register("Multiply")(FuncProcessorBuilder(tensor.convert.to_np))
ToTorch = register("Add")(FuncProcessorBuilder(tensor.convert.to_torch))
ToTorch = register("AsType")(FuncProcessorBuilder(tensor.convert.astype))

Multiply = register("Multiply")(FuncProcessorBuilder(tensor.ops.multiply))
Add = register("Add")(FuncProcessorBuilder(tensor.ops.add))
Divide = register("Divide")(FuncProcessorBuilder(tensor.ops.divide))
IntDivide = register("IntDivide")(FuncProcessorBuilder(tensor.ops.int_divide))
Power = register("Power")(FuncProcessorBuilder(tensor.ops.power))
Compare = register("Compare")(FuncProcessorBuilder(tensor.ops.compare))
Unsqueeze = register("Unsqueeze")(FuncProcessorBuilder(tensor.ops.unsqueeze))
Squeeze = register("Squeeze")(FuncProcessorBuilder(tensor.ops.squeeze))
Interpolate = register("Interpolate")(FuncProcessorBuilder(tensor.ops.interpolate))
FilterCC = register("FilterCC")(FuncProcessorBuilder(tensor.ops.filter_cc))
