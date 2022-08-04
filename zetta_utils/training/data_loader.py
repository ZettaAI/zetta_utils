import torch
from zetta_utils import builder

builder.register("TorchDataLoader")(torch.utils.data.DataLoader)
