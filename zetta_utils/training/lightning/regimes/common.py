import wandb

from zetta_utils import viz


def log_results(mode: str, title_suffix: str = "", **kwargs):
    wandb.log(
        {
            f"results/{mode}_{title_suffix}_slider": [
                wandb.Image(viz.rendering.Renderer()(v.squeeze()), caption=k)  # type: ignore
                for k, v in kwargs.items()
            ]
        }
    )
