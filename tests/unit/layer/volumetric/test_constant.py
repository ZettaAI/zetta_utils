from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import (
    VolumetricIndex,
    build_constant_volumetric_layer,
)


def test_constant_read():
    num_channels = 2
    value = 3.14
    layer = build_constant_volumetric_layer(
        value=value,
        num_channels=num_channels,
    )
    idx = VolumetricIndex(
        resolution=Vec3D[float](2, 4, 8),
        bbox=BBox3D(
            (
                (0, 128),
                (0, 128),
                (0, 128),
            )
        ),
    )

    data = layer[idx]
    assert data.shape == (num_channels, 128 / 2, 128 / 4, 128 / 8)
    assert (data != value).sum() == 0
