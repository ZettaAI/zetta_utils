import affine
import torch
import torchfields  # pylint: disable=unused-import


def get_affine_field(
    size,
    rot_deg=0,
    scale=1,
    shear_x_px=0,
    shear_y_px=0,
    trans_x_px=0,
    trans_y_px=0,
):
    aff = (
        affine.Affine.translation(trans_x_px * 2 / size, trans_y_px * 2 / size)
        * affine.Affine.rotation(rot_deg)
        * affine.Affine.shear(shear_x_px, shear_y_px)
        * affine.Affine.scale(scale)
    )
    mat = torch.tensor([[aff.a, aff.b, aff.c], [aff.d, aff.e, aff.f]]).unsqueeze(0)
    field = torch.Field.affine_field(mat, size=(1, 2, size, size))
    return field
