import affine
import torch
import torchfields  # pylint: disable=unused-import


def get_affine_field(
    size,
    trans_x_px=0,
    trans_y_px=0,
    rot_deg=0,
    shear_x_deg=0,
    shear_y_deg=0,
    scale=1,
):
    """
    Return 2D displacement field that represents the given affine transformation.
    Transformations are applied in the following order -- translation->rotation->shear->scale.
    Note that the resulting field is represented in pixel magnitudes.

    :param size: Shape along the X and Y dimension of the resulting field.
    :param trans_x_px: X translation in pixels, from left to right.
    :param trans_y_px: Y translation in pixels, from top to bottom.
    :param rot_deg: Rotation degrees, clockwise
    :param shear_x_deg: X shear degrees.
    :param shear_y_deg: Y shear degrees.

    """
    aff = (
        affine.Affine.translation(-trans_x_px * 2 / size, -trans_y_px * 2 / size)
        * affine.Affine.rotation(-rot_deg)
        * affine.Affine.shear(-shear_x_deg, -shear_y_deg)
        * affine.Affine.scale(1 / scale)
    )
    mat = torch.tensor([[aff.a, aff.b, aff.c], [aff.d, aff.e, aff.f]]).unsqueeze(0)
    field = torch.Field.affine_field(mat, size=(1, 2, size, size))
    return field
