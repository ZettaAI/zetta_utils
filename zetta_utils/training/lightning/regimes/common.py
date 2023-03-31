from __future__ import annotations

from functools import reduce
from typing import Callable

import wandb
from pytorch_lightning.loggers.logger import Logger
from torchvision.utils import make_grid

from zetta_utils import tensor_ops, viz
from zetta_utils.geometry import Vec3D
from zetta_utils.tensor_typing import Tensor


def is_2d_image(tensor):
    return len(tensor.squeeze().shape) == 2 or (
        len(tensor.squeeze().shape) == 3 and tensor.squeeze().shape[0] <= 3
    )


def log_results(mode: str, title_suffix: str = "", logger: Logger | None = None, **kwargs):
    if all(is_2d_image(v) for v in kwargs.values()):
        row = [
            wandb.Image(viz.rendering.Renderer()(v.squeeze()), caption=k)
            for k, v in kwargs.items()
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
                    rendered = viz.rendering.Renderer()(v.squeeze())
                else:
                    rendered = viz.rendering.Renderer()(v[..., z].squeeze())

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
    return make_grid(imgs, nrow=depth, padding=0)


def log_3d_results(
    mode: str,
    transforms: dict[str, Callable],
    title_suffix: str = "",
    **kwargs,
) -> None:
    sizes = [Vec3D(*v.shape[-3:]) for v in kwargs.values()]  # type: list[Vec3D]
    min_s = reduce(lambda acc, cur: cur if acc > cur else acc, sizes)

    row = []
    for k, v in kwargs.items():
        data = tensor_ops.crop_center(v, min_s)
        data = transforms[k](data) if k in transforms else data        
        rendered = render_3d_result(data)
        row.append(wandb.Image(rendered, caption=k))

    wandb.log({f"results/{mode}_{title_suffix}_slider": row})
