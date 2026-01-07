import numpy as np

from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.layer.volumetric.seg_contact import SegContact


def make_seg_contact(
    contact_id: int = 1,
    seg_a: int = 100,
    seg_b: int = 200,
    com: tuple[float, float, float] = (100.0, 100.0, 100.0),
    n_faces: int = 3,
) -> SegContact:
    """Helper to create a SegContact for testing."""
    contact_faces = np.array(
        [[com[0] + i, com[1] + i, com[2] + i, 0.5] for i in range(n_faces)],
        dtype=np.float32,
    )
    return SegContact(
        id=contact_id,
        seg_a=seg_a,
        seg_b=seg_b,
        com=Vec3D(*com),
        contact_faces=contact_faces,
    )


# --- Basic instantiation tests ---


def test_seg_contact_instantiation():
    """Test basic SegContact creation."""
    contact = make_seg_contact(contact_id=42, seg_a=100, seg_b=200, com=(50.0, 60.0, 70.0))

    assert contact.id == 42
    assert contact.seg_a == 100
    assert contact.seg_b == 200
    assert contact.com == Vec3D(50.0, 60.0, 70.0)
    assert contact.contact_faces.shape == (3, 4)


def test_seg_contact_with_optional_fields():
    """Test SegContact with all optional fields."""
    contact = SegContact(
        id=1,
        seg_a=100,
        seg_b=200,
        com=Vec3D(100.0, 100.0, 100.0),
        contact_faces=np.array([[100, 100, 100, 0.5]], dtype=np.float32),
        local_pointclouds={(1000, 2048): {100: np.zeros((10, 3)), 200: np.ones((10, 3))}},
        merge_decisions={"ground_truth": True, "model_v1": False},
        partner_metadata={100: {"type": "axon"}, 200: {"type": "dendrite"}},
    )

    assert contact.local_pointclouds is not None
    assert (1000, 2048) in contact.local_pointclouds
    assert contact.merge_decisions == {"ground_truth": True, "model_v1": False}
    assert contact.partner_metadata == {100: {"type": "axon"}, 200: {"type": "dendrite"}}


def test_seg_contact_defaults_to_none():
    """Test that optional fields default to None."""
    contact = make_seg_contact()

    assert contact.local_pointclouds is None
    assert contact.merge_decisions is None
    assert contact.partner_metadata is None


# --- in_bounds tests ---


def test_in_bounds_com_inside():
    """Test in_bounds returns True when COM is inside bbox."""
    # COM at (100, 100, 100) nm
    contact = make_seg_contact(com=(100.0, 100.0, 100.0))

    # Bbox from (0, 0, 0) to (200, 200, 200) in voxels at resolution (1, 1, 1)
    # So in nm: (0, 0, 0) to (200, 200, 200)
    idx = VolumetricIndex(
        resolution=Vec3D(1, 1, 1),
        bbox=BBox3D.from_slices((slice(0, 200), slice(0, 200), slice(0, 200))),
    )

    assert contact.in_bounds(idx) is True


def test_in_bounds_com_outside():
    """Test in_bounds returns False when COM is outside bbox."""
    # COM at (300, 300, 300) nm
    contact = make_seg_contact(com=(300.0, 300.0, 300.0))

    # Bbox from (0, 0, 0) to (200, 200, 200) nm
    idx = VolumetricIndex(
        resolution=Vec3D(1, 1, 1),
        bbox=BBox3D.from_slices((slice(0, 200), slice(0, 200), slice(0, 200))),
    )

    assert contact.in_bounds(idx) is False


def test_in_bounds_com_on_boundary_start():
    """Test in_bounds with COM exactly on start boundary (inclusive)."""
    contact = make_seg_contact(com=(100.0, 100.0, 100.0))

    # Bbox starts exactly at COM
    idx = VolumetricIndex(
        resolution=Vec3D(1, 1, 1),
        bbox=BBox3D.from_slices((slice(100, 200), slice(100, 200), slice(100, 200))),
    )

    assert contact.in_bounds(idx) is True


def test_in_bounds_com_on_boundary_end():
    """Test in_bounds with COM exactly on end boundary (exclusive)."""
    contact = make_seg_contact(com=(200.0, 200.0, 200.0))

    # Bbox ends exactly at COM
    idx = VolumetricIndex(
        resolution=Vec3D(1, 1, 1),
        bbox=BBox3D.from_slices((slice(100, 200), slice(100, 200), slice(100, 200))),
    )

    assert contact.in_bounds(idx) is False


def test_in_bounds_with_resolution():
    """Test in_bounds with non-unit resolution."""
    # COM at (1600, 1600, 4000) nm
    contact = make_seg_contact(com=(1600.0, 1600.0, 4000.0))

    # Bbox from (0, 0, 0) to (200, 200, 200) voxels at resolution (16, 16, 40)
    # In nm: (0, 0, 0) to (3200, 3200, 8000)
    idx = VolumetricIndex(
        resolution=Vec3D(16, 16, 40),
        bbox=BBox3D.from_slices((slice(0, 200), slice(0, 200), slice(0, 200))),
    )

    assert contact.in_bounds(idx) is True


def test_in_bounds_partial_outside():
    """Test in_bounds when COM is outside in one dimension only."""
    # COM at (100, 100, 300) nm - outside in z
    contact = make_seg_contact(com=(100.0, 100.0, 300.0))

    idx = VolumetricIndex(
        resolution=Vec3D(1, 1, 1),
        bbox=BBox3D.from_slices((slice(0, 200), slice(0, 200), slice(0, 200))),
    )

    assert contact.in_bounds(idx) is False
