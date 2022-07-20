"""
All of the declared types for _external_ annotation use.
This file is imported last, so should not be used within the library.
"""
import zetta_utils as zu

Array = zu.typing.Array

TorchInterpolationMode = zu.data.basic_ops.TorchInterpolationMode
CustomInterpolationMode = zu.data.basic_ops.CustomInterpolationMode
InterpolationMode = zu.data.basic_ops.InterpolationMode
CompareMode = zu.data.basic_ops.CompareMode

VolumetricResolution = zu.bcube.VolumetricResolution
VolumetricCoord = zu.bcube.VolumetricCoord
VolumetricSlices = zu.bcube.VolumetricSlices
VolumetricDimOrder = zu.data.layers.volumetric.VolumetricDimOrder
