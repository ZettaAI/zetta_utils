import torch


def save_model(model: torch.nn.Module, path: str):
    torch.save(model.state_dict(), path)
