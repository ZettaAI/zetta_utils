from typing import Dict, List, Literal, Optional, Tuple

import einops
import metroem
import torch
import torchfields  # pylint: disable=unused-import # monkeypatch

from zetta_utils import builder, log

logger = log.get_logger("zetta_utils")


def compute_aced_loss(
    pfields: Dict[Tuple[int, int], torch.Tensor], afields: List[torch.Tensor], rigidity_weight
) -> torch.Tensor:
    inter_loss = 0
    intra_loss = 0
    for i in range(1, len(afields)):
        inter_loss_map = (
            pfields[(i, i - 1)]
            .from_pixels()(  # type: ignore
                afields[i - 1].from_pixels()  # type: ignore
            )
            .pixels()
            - afields[i]
        )

        intra_loss_map = metroem.loss.rigidity(afields[i])
        this_inter_loss = (inter_loss_map ** 2).mean()
        this_intra_loss = intra_loss_map.mean()

        intra_loss += this_intra_loss
        inter_loss += this_inter_loss

    loss = inter_loss + rigidity_weight * intra_loss
    print(inter_loss, rigidity_weight * intra_loss, loss)
    return loss


@builder.register("perform_aced_relaxation")
def perform_aced_relaxation(
    field: torch.Tensor,
    num_iter=100,
    lr=0.3,
    rigidity_weight=10.0,
    fix: Optional[Literal["first", "last", "both"]] = "first",
) -> torch.Tensor:
    field_xy = einops.rearrange(field, "C X Y Z -> Z C X Y").field()  # type: ignore
    num_sections = field_xy.shape[0]
    afields = [torch.zeros_like(field_xy[0:1]).cuda() for _ in range(num_sections)]

    for i in range(num_sections):
        afields[i].requires_grad = True

    if fix is None:
        opt_range = range(num_sections)
    elif fix == "first":
        opt_range = range(1, num_sections)
    elif fix == "last":
        opt_range = range(num_sections - 1)
    else:
        assert fix == "both"
        opt_range = range(1, num_sections - 1)

    optimizer = torch.optim.Adam(
        [afields[i] for i in opt_range],
        lr=lr,
    )

    pfields = {(i, i - 1): field_xy[i : i + 1].cuda() for i in range(num_sections)}
    for i in range(num_iter):
        loss = compute_aced_loss(pfields=pfields, afields=afields, rigidity_weight=rigidity_weight)
        if loss < 0.005:
            break
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    result_xy = torch.cat(afields, 0)
    result = einops.rearrange(result_xy, "Z C X Y -> C X Y Z")
    return result
