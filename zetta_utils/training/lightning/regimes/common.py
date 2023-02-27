import wandb

from zetta_utils import viz


def is_2d_image(tensor):
    return len(tensor.squeeze().shape) == 2 or (
        len(tensor.squeeze().shape) == 3 and tensor.squeeze().shape[0] <= 3
    )


def log_results(mode: str, title_suffix: str = "", **kwargs):
    if all(is_2d_image(v) for v in kwargs.values()):
        row = [
            wandb.Image(viz.rendering.Renderer()(v.squeeze()), caption=k)  # type: ignore
            for k, v in kwargs.items()
        ]
        wandb.log({f"results/{mode}_{title_suffix}_slider": row})
    else:
        max_z = max(v.shape[-1] for v in kwargs.values())

        for z in range(max_z):
            row = []
            for k, v in kwargs.items():
                if is_2d_image(v):
                    rendered = viz.rendering.Renderer()(v.squeeze())  # type: ignore
                else:
                    rendered = viz.rendering.Renderer()(v[..., z].squeeze())  # type: ignore

                row.append(wandb.Image(rendered, caption=k))

            wandb.log({f"results/{mode}_{title_suffix}_slider_z{z}": row})
