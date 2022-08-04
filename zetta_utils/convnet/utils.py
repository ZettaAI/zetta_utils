import torch


def save_model(model: torch.nn.Module, path: str):  # pragma: no cover
    torch.save(model.state_dict(), path)
