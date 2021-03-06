# type: ignore
"""Buildable data processors."""
import zetta_utils as zu

from .common import func_to_proc


# basic_ops
Multiply = zu.spec_parser.register("Multiply")(func_to_proc(zu.data.basic_ops.multiply))
Add = zu.spec_parser.register("Add")(func_to_proc(zu.data.basic_ops.add))
Divide = zu.spec_parser.register("Divide")(func_to_proc(zu.data.basic_ops.divide))
IntDivide = zu.spec_parser.register("IntDivide")(
    func_to_proc(zu.data.basic_ops.int_divide)
)
Power = zu.spec_parser.register("Power")(func_to_proc(zu.data.basic_ops.power))
Compare = zu.spec_parser.register("Compare")(func_to_proc(zu.data.basic_ops.compare))
Unsqueeze = zu.spec_parser.register("Unsqueeze")(
    func_to_proc(zu.data.basic_ops.unsqueeze)
)
Squeeze = zu.spec_parser.register("Squeeze")(func_to_proc(zu.data.basic_ops.squeeze))
Interpolate = zu.spec_parser.register("Interpolate")(
    func_to_proc(zu.data.basic_ops.interpolate)
)

# mask_ops
FilterCC = zu.spec_parser.register("FilterCC")(func_to_proc(zu.data.mask_ops.filter_cc))
