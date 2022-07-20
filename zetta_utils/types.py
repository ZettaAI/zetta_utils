"""
All of the declared types for _external_ annotation use.
This file is imported last, so should not be used within the library.
"""
import zetta_utils as zu

Array = zu.typing.Array

TorchInterpolationModes = zu.data.basic_ops.TorchInterpolationModes
CustomInterpolationModes = zu.data.basic_ops.CustomInterpolationModes
InterpolationModes = zu.data.basic_ops.InterpolationModes

DimOrder = zu.data.layers.volumetric.DimOrder
