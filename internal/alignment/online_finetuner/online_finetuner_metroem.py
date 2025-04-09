from __future__ import annotations

import einops
import metroem
import torch
import torchfields

from zetta_utils import builder, log, tensor_ops

logger = log.get_logger("zetta_utils")


@builder.register("align_with_online_finetuner")
def align_with_online_finetuner(
    src: torch.Tensor,  # (C, X, Y, Z)
    tgt: torch.Tensor,  # (C, X, Y, Z)
    sm: float = 100,
    num_iter: int = 200,
    lr: float = 5e-2,
    src_field: torch.Tensor | None = None,  # (C, X, Y, Z)
    src_zeros_sm_mult: float = 0.001,
    tgt_zeros_sm_mult: float = 0.001,
):
    assert src.shape == tgt.shape
    # assert len(src.shape) == 4 # (1, C, X, Y,)
    # assert src.shape[0] == 1
    src = einops.rearrange(src, "C X Y 1 -> 1 C X Y").float()
    tgt = einops.rearrange(tgt, "C X Y 1 -> 1 C X Y").float()

    if src_field is None:
        src_field = torch.zeros([1, 2, tgt.shape[2], tgt.shape[3]], device=tgt.device).float()
    else:
        src_field = einops.rearrange(src_field, "C X Y 1 -> 1 C X Y")
        scales = [src.shape[i] / src_field.shape[i] for i in range(2, 4)]
        assert scales[0] == scales[1]
        src_field = tensor_ops.interpolate(
            src_field, scale_factor=scales, mode="field", unsqueeze_input_to=4
        )
    assert isinstance(src_field, torch.Tensor)
    orig_device = src.device

    if torch.cuda.is_available():
        src = src.cuda()
        tgt = tgt.cuda()
        src_field = src_field.cuda()

    if src.abs().sum() == 0 or tgt.abs().sum() == 0:
        result = torch.zeros_like(src_field)
    elif num_iter <= 0:
        result = src_field
    else:
        sm_keys = {
            "src": [
                {
                    "name": "src_zeros",
                    "fm": 0,
                    "mask_value": src_zeros_sm_mult,
                    "binarization": {"strat": "eq", "value": 0},
                }
            ],
            "tgt": [
                {
                    "name": "tgt_zeros",
                    "fm": 0,
                    "mask_value": tgt_zeros_sm_mult,
                    "binarization": {"strat": "eq", "value": 0},
                }
            ],
        }
        mse_keys = {
            "src": [
                {
                    "name": "src_zeros",
                    "fm": 0,
                    "mask_value": 0,
                    "binarization": {"strat": "eq", "value": 0},
                }
            ],
            "tgt": [
                {
                    "name": "tgt_zeros",
                    "fm": 0,
                    "mask_value": 0,
                    "binarization": {"strat": "eq", "value": 0},
                }
            ],
        }

        with torchfields.set_identity_mapping_cache(True, clear_cache=True):
            result = metroem.finetuner.optimize_pre_post_ups(
                src,
                tgt,
                src_field,
                src_zeros=(src[:, 0] == 0.0).unsqueeze(1),
                tgt_zeros=(tgt[:, 0] == 0.0).unsqueeze(1),
                src_defects=torch.zeros((src.shape[0], 1, src.shape[1], src.shape[2])),
                tgt_defects=torch.zeros((src.shape[0], 1, src.shape[1], src.shape[2])),
                crop=2,
                num_iter=num_iter,
                lr=lr,
                sm=sm,
                l2=0,
                wd=0,
                max_bad=5,
                verbose=True,
                opt_res_coarsness=0,
                normalize=False,
                sm_keys_to_apply=sm_keys,
                mse_keys_to_apply=mse_keys,
            )
    result = einops.rearrange(result, "1 C X Y -> C X Y 1")
    result = result.detach().to(orig_device)
    result[result.abs() < 0.001] = 0
    return result
