import torch
import torchfields  # type: ignore

import metroem  # type: ignore
import einops
from zetta_utils import builder


@builder.register("align_with_online_finetunner")
def align_with_online_finetuner(
    src_data,
    tgt_data,
    sm=100,
    num_iter=200,
    lr=3e-1,
    defect_sm=0.0,
    mse_keys_to_apply=None,
    sm_keys_to_apply=None,
):
    assert src_data.shape == tgt_data.shape
    src_data = einops.rearrange(src_data, "C X Y 1 -> 1 C X Y")
    tgt_data = einops.rearrange(tgt_data, "C X Y 1 -> 1 C X Y")

    # if src_agg_field is None:
    if torch.cuda.is_available():
        src_data = src_data.cuda()
        tgt_data = tgt_data.cuda()

    res_start = torch.zeros(
        [1, 2, tgt_data.shape[-1], tgt_data.shape[-1]], device=tgt_data.device
    ).type_as(tgt_data)
    # else:
    #    res_start = src_agg_field.field()

    with torchfields.set_identity_mapping_cache(True, clear_cache=True):
        result = metroem.finetuner.optimize_pre_post_ups(
            src_data,
            tgt_data,
            res_start,
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
    return result.detach().cpu()
