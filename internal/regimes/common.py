from __future__ import annotations

from functools import reduce
from typing import Sequence

import wandb
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.loggers.wandb import WandbLogger
from torchvision.utils import make_grid

from zetta_utils import tensor_ops, viz
from zetta_utils.geometry import Vec3D
from zetta_utils.tensor_typing import Tensor


def is_2d_image(tensor):
    return len(tensor.squeeze().shape) == 2 or (
        len(tensor.squeeze().shape) == 3 and tensor.squeeze().shape[0] <= 3
    )


RENDERER = viz.rendering.Renderer()


def log_results(mode: str, title_suffix: str = "", logger: Logger | None = None, **kwargs):
    if all(is_2d_image(v) for v in kwargs.values()):
        row = [
            wandb.Image(RENDERER(v.squeeze().detach().cpu()), caption=k) for k, v in kwargs.items()
        ]
        if logger is None:
            wandb.log({f"results/{mode}_{title_suffix}_slider": row})
        else:
            logger.log_image(f"results/{mode}_{title_suffix}_slider", images=row)
    else:
        max_z = max(v.shape[-1] for v in kwargs.values())

        for z in range(max_z):
            row = []
            for k, v in kwargs.items():
                if is_2d_image(v):
                    rendered = viz.rendering.Renderer()(v.squeeze().detach().cpu())
                else:
                    rendered = viz.rendering.Renderer()(v[..., z].squeeze().detach().cpu())

                row.append(wandb.Image(rendered, caption=k))

            if logger is None:
                wandb.log({f"results/{mode}_{title_suffix}_slider_z{z}": row})
            else:
                logger.log_image(f"results/{mode}_{title_suffix}_slider_z{z}", images=row)


def render_3d_result(data: Tensor):
    assert 3 <= data.ndim <= 5
    data_ = tensor_ops.convert.to_torch(data, device="cpu")
    data_ = data_[0, ...] if data_.ndim > 4 else data_
    data_ = data_[0:3, ...] if data_.ndim > 3 else data_
    depth = data_.shape[-1]
    imgs = [data_[..., z] for z in range(depth)]
    return make_grid(imgs, nrow=depth, padding=2)


def log_3d_results(
    logger: Logger | None,
    mode: str,
    title_suffix: str = "",
    max_dims: Sequence[int] | None = None,
    **kwargs,
) -> None:
    sizes = [Vec3D(*v.shape[-3:]) for v in kwargs.values()]  # type: list[Vec3D]
    min_s = reduce(lambda acc, cur: cur if acc > cur else acc, sizes)
    min_s_ = list(min_s)  # mypy doesn't like a single combined line with above
    if max_dims is not None:
        assert len(max_dims) == len(min_s_)
        min_s_ = [min(x, y) if y is not None else x for x, y in zip(min_s_, max_dims)]

    row = []
    for k, v in kwargs.items():
        data = tensor_ops.crop_center(v, min_s_)
        rendered = render_3d_result(data)
        row.append(wandb.Image(rendered, caption=k))

    if logger is None:
        wandb.log({f"results/{mode}_{title_suffix}_slider": row})
    else:
        assert isinstance(logger, WandbLogger)
        logger.experiment.log({f"results/{mode}_{title_suffix}_slider": row})
