import torch
import numpy as np


def to_np(x):
    """
    """
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        pass

