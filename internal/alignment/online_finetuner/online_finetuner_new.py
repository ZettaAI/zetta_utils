from __future__ import annotations

import time

import einops
import torch
import torchfields

from zetta_utils import builder, log, tensor_ops
from zetta_utils.internal.alignment.field import get_rigidity_map_zcxy

logger = log.get_logger("zetta_utils")


def _standardize_values(data: torch.Tensor) -> torch.Tensor:
    if data.dtype == torch.uint8:
        return data / 255
    elif data.dtype == torch.int8:
        return data / 127
    else:
        return data


@builder.register("align_with_online_finetuner_new")
def align_with_online_finetuner(
    src: torch.Tensor,  # (C, X, Y, Z)
    tgt: torch.Tensor,  # (C, X, Y, Z)
    rig: float = 100,
    num_iter: int = 200,
    lr: float = 5e-2,
    src_field: torch.Tensor | None = None,  # (C, X, Y, Z)
    src_zeros_rig_mult: float = 0.001,
    tgt_zeros_rig_mult: float = 0.001,
    src_rig_weight: torch.Tensor | None = None,  # (C, X, Y, Z)
    tgt_rig_weight: torch.Tensor | None = None,  # (C, X, Y, Z)
    src_mse_weight: torch.Tensor | None = None,  # (C, X, Y, Z)
    tgt_mse_weight: torch.Tensor | None = None,  # (C, X, Y, Z)
    src_pinned_mask: torch.Tensor | None = None,  # (C, X, Y, Z)
):
    assert src.shape == tgt.shape
    orig_device = src.device

    src = einops.rearrange(_standardize_values(src), "C X Y 1 -> 1 C X Y").float()
    tgt = einops.rearrange(_standardize_values(tgt), "C X Y 1 -> 1 C X Y").float()

    if src_rig_weight is not None:
        src_rig_weight = einops.rearrange(src_rig_weight, "C X Y 1 -> 1 C X Y").float()
    if tgt_rig_weight is not None:
        tgt_rig_weight = einops.rearrange(tgt_rig_weight, "C X Y 1 -> 1 C X Y").float()
    if src_mse_weight is not None:
        src_mse_weight = einops.rearrange(src_mse_weight, "C X Y 1 -> 1 C X Y").float()
    if tgt_mse_weight is not None:
        tgt_mse_weight = einops.rearrange(tgt_mse_weight, "C X Y 1 -> 1 C X Y").float()
    if src_pinned_mask is not None:
        src_pinned_mask = einops.rearrange(src_pinned_mask, "C X Y 1 -> 1 C X Y").bool()

    if src_field is None:
        src_field_final = torch.zeros(
            [1, 2, tgt.shape[2], tgt.shape[3]], device=tgt.device
        ).float()
    else:
        src_field_final = einops.rearrange(src_field, "C X Y 1 -> 1 C X Y")
        scales = [src.shape[i] / src_field_final.shape[i] for i in range(2, 4)]
        assert scales[0] == scales[1]

        src_field_final = tensor_ops.interpolate(
            src_field_final, scale_factor=scales, mode="field", unsqueeze_input_to=4
        )

    if torch.cuda.is_available():
        src = src.cuda()
        tgt = tgt.cuda()
        src_field_final = src_field_final.cuda()

    if src.abs().sum() == 0 or tgt.abs().sum() == 0 or num_iter <= 0:
        result = src_field_final
    else:
        with torchfields.set_identity_mapping_cache(True, clear_cache=True):
            result = _online_finetunner(
                src=src,
                tgt=tgt,
                initial_res=src_field_final,
                src_zeros_rig_mult=src_zeros_rig_mult,
                tgt_zeros_rig_mult=tgt_zeros_rig_mult,
                src_rig_weight=src_rig_weight,
                tgt_rig_weight=tgt_rig_weight,
                src_mse_weight=src_mse_weight,
                tgt_mse_weight=tgt_mse_weight,
                src_pinned_mask=src_pinned_mask,
                num_iter=num_iter,
                lr=lr,
                rig=rig,
                l2=0,
                wd=0,
                max_bad=5,
                verbose=True,
                opt_res_coarsness=0,
            )
    result = einops.rearrange(result, "1 C X Y -> C X Y 1")
    result = result.detach().to(orig_device)
    result[result.abs() < 0.001] = 0
    return result


def _get_online_ft_loss(
    src: torch.Tensor,
    tgt: torch.Tensor,
    src_field: torchfields.Field,
    rig: float,
    src_zeros_rig_mult: float,
    tgt_zeros_rig_mult: float,
    src_rig_weight: torch.Tensor | None = None,
    tgt_rig_weight: torch.Tensor | None = None,
    src_mse_weight: torch.Tensor | None = None,
    tgt_mse_weight: torch.Tensor | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    if src_rig_weight is None:
        src_rig_weight = torch.ones_like(src)
        src_rig_weight[src == 0] = src_zeros_rig_mult
    if tgt_rig_weight is None:
        tgt_rig_weight = torch.ones_like(tgt)
        tgt_rig_weight[tgt == 0] = tgt_zeros_rig_mult
    rig_weights = (
        src_field.from_pixels()
        .sample(src_rig_weight, padding_mode="border")
        .minimum(tgt_rig_weight)
    )

    if src_mse_weight is None:
        src_mse_weight = torch.ones_like(src)
        src_mse_weight[src == 0] = 0
    if tgt_mse_weight is None:
        tgt_mse_weight = torch.ones_like(tgt)
        tgt_mse_weight[tgt == 0] = 0
    mse_weights = (
        src_field.from_pixels()
        .sample(src_mse_weight, padding_mode="border")
        .minimum(tgt_mse_weight)
    )

    pred_tgt = src_field.from_pixels().sample(src, mode="bilinear")
    mse_map = (pred_tgt - tgt) ** 2
    mse_loss = (mse_map * mse_weights).mean()

    rig_map = get_rigidity_map_zcxy(src_field, weight_map=rig_weights)
    rig_loss = rig_map.mean()
    loss = mse_loss + rig * rig_loss
    if verbose:
        logger.info(f"{loss.item()}, {mse_loss.item()}, {rig_loss.item()}")
    return loss


def _online_finetunner(  # pylint: disable=too-many-locals,too-many-statements
    src: torch.Tensor,
    tgt: torch.Tensor,
    initial_res: torch.Tensor,
    rig: float,
    lr: float,
    num_iter: int,
    src_zeros_rig_mult: float,
    tgt_zeros_rig_mult: float,
    src_rig_weight: torch.Tensor | None = None,
    tgt_rig_weight: torch.Tensor | None = None,
    src_mse_weight: torch.Tensor | None = None,
    tgt_mse_weight: torch.Tensor | None = None,
    src_pinned_mask: torch.Tensor | None = None,
    noimpr_period: int = 50,
    opt_res_coarsness: int = 0,
    wd: float = 0,
    l2: float = 1e-4,
    verbose: bool = False,
    max_bad: int = 15,
) -> torch.Tensor:
    pred_res = initial_res.detach().field_()  # type: ignore
    if opt_res_coarsness > 0:
        pred_res = pred_res.down(opt_res_coarsness)
    pred_res.requires_grad = True

    trainable = [pred_res]
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=wd)

    prev_loss = []
    s = time.time()

    pred_res_curr = pred_res
    if opt_res_coarsness > 0:
        pred_res_curr = pred_res_curr.up(opt_res_coarsness)
    with torch.no_grad():
        loss = _get_online_ft_loss(
            src=src,
            tgt=tgt,
            src_field=pred_res_curr,
            rig=rig,
            src_zeros_rig_mult=src_zeros_rig_mult,
            tgt_zeros_rig_mult=tgt_zeros_rig_mult,
            src_rig_weight=src_rig_weight,
            tgt_rig_weight=tgt_rig_weight,
            src_mse_weight=src_mse_weight,
            tgt_mse_weight=tgt_mse_weight,
            verbose=verbose,
        )
        best_loss = loss.detach().cpu().item()
    new_best_ago = 0
    lr_halfed_count = 0
    no_impr_count = 0
    new_best_count = 0

    for epoch in range(num_iter):
        pred_res_curr = pred_res
        if opt_res_coarsness > 0:
            pred_res_curr = pred_res_curr.up(opt_res_coarsness)

        loss = _get_online_ft_loss(
            src=src,
            tgt=tgt,
            src_field=pred_res_curr,
            rig=rig,
            src_zeros_rig_mult=src_zeros_rig_mult,
            tgt_zeros_rig_mult=tgt_zeros_rig_mult,
            src_rig_weight=src_rig_weight,
            tgt_rig_weight=tgt_rig_weight,
            src_mse_weight=src_mse_weight,
            tgt_mse_weight=tgt_mse_weight,
        )
        if l2 > 0.0:
            loss += (pred_res_curr ** 2).mean() * l2
        curr_loss = loss.detach().cpu().item()

        min_improve = 1e-11
        if curr_loss + min_improve <= best_loss:
            # Improvement
            best_loss = curr_loss
            new_best_count += 1
            new_best_ago = 0
        else:
            new_best_ago += 1
            if new_best_ago > noimpr_period:
                # No improvement, reduce learning rate
                no_impr_count += 1
                lr /= 2
                lr_halfed_count += 1

                optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=wd)
                new_best_ago -= 5
            prev_loss.append(curr_loss)

        optimizer.zero_grad()
        loss.backward()
        if src_pinned_mask is not None:
            pred_res.grad *= (~src_pinned_mask).float()
        optimizer.step()

        if lr_halfed_count >= max_bad:
            break

    pred_res_curr = pred_res
    if opt_res_coarsness > 0:
        pred_res_curr = pred_res_curr.up(opt_res_coarsness)

    with torch.no_grad():
        loss = _get_online_ft_loss(
            src=src,
            tgt=tgt,
            src_field=pred_res_curr,
            rig=rig,
            src_zeros_rig_mult=src_zeros_rig_mult,
            tgt_zeros_rig_mult=tgt_zeros_rig_mult,
            src_rig_weight=src_rig_weight,
            tgt_rig_weight=tgt_rig_weight,
            src_mse_weight=src_mse_weight,
            tgt_mse_weight=tgt_mse_weight,
            verbose=verbose,
        )

    e = time.time()

    if verbose:
        logger.info(f"New best: {new_best_count}, No impr: {no_impr_count}, Iter: {epoch}")
        logger.info(e - s)
        logger.info("==========")

    return pred_res_curr
