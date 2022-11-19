import einops
import metroem
import torch
import torchfields

from zetta_utils import builder, log, tensor_ops

logger = log.get_logger("zetta_utils")


@builder.register("align_with_online_finetunner")
def align_with_online_finetuner(
    src_data,  # (C, X, Y, Z)
    tgt_data,  # (C, X, Y, Z)
    sm=100,
    num_iter=200,
    lr=3e-1,
    src_field_data=None,  # (C, X, Y, Z)
):
    assert src_data.shape == tgt_data.shape
    # assert len(src_data.shape) == 4 # (1, C, X, Y,)
    # assert src_data.shape[0] == 1

    src_data = einops.rearrange(src_data, "C X Y 1 -> 1 C X Y")
    tgt_data = einops.rearrange(tgt_data, "C X Y 1 -> 1 C X Y")

    if src_field is None:
        src_field = torch.zeros(
            [1, 2, tgt_data.shape[2], tgt_data.shape[3]], device=tgt_data.device
        ).float()
    else:
        src_field = einops.rearrange(src_field, "C X Y 1 -> 1 C X Y")
        scales = [src_data.shape[i] / src_field.shape[i] for i in range(2, 4)]
        assert scales[0] == scales[1]
        src_field = tensor_ops.interpolate(src_field, scale_factor=scales, mode="field")

    orig_device = src_data.device

    if torch.cuda.is_available():
        src_data = src_data.cuda()
        tgt_data = tgt_data.cuda()

    # else:
    #    res_start = src_agg_field.field()

    with torchfields.set_identity_mapping_cache(True, clear_cache=True):
        result = metroem.finetuner.optimize_pre_post_ups(
            src_data,
            tgt_data,
            src_field_data,
            src_zeros=(src_data == -0.5),
            tgt_zeros=(tgt_data == -0.5),
            src_defects=(src_data == -0.5),
            tgt_defects=(tgt_data == -0.5),
            crop=2,
            num_iter=num_iter,
            lr=lr,
            sm=sm,
            verbose=True,
            opt_res_coarsness=0,
            normalize=True,
        )
    # convert result back to C X Y Z
    result = einops.rearrange(result, "1 C X Y -> C X Y 1")

    return result.detach().to(orig_device)
