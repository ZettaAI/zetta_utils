import torch
import numpy as np


def get_np(x):
    """
    adsfads
    adf
    aaa
    ##
    """
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        pass

