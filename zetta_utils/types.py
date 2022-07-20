"""
All of the declared types for _external_ annotation use.
This file is imported last, so should not be used within the library.
"""
import zetta_utils as zu

Array = zu.typing.Array
IntVec3D = zu.typing.Array
FloatVec3D = zu.typing.Array
Vec3D = zu.typing.Array
Slice3D = zu.typing.Array

TorchInterpolationMode = zu.data.basic_ops.TorchInterpolationMode
CustomInterpolationMode = zu.data.basic_ops.CustomInterpolationMode
InterpolationMode = zu.data.basic_ops.InterpolationMode
CompareMode = zu.data.basic_ops.CompareMode

VolumetricCoord = zu.bcube.VolumetricCoord
VolumetricDimOrder = zu.data.layers.volumetric.VolumetricDimOrder
