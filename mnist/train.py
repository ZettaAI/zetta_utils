# pylint: disable=missing-docstring # type: ignore

from timeit import default_timer as timer

import pytorch_lightning as pl
import torch
from pytorch_lightning.plugins.environments.torchelastic_environment import (
    TorchElasticEnvironment,
)
from pytorch_lightning.strategies import ddp
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from .module import Decoder, Encoder, LitAutoEncoder

transform = transforms.ToTensor()
train_set = MNIST(root="MNIST", download=True, train=True, transform=transform)
test_set = MNIST(root="MNIST", download=True, train=False, transform=transform)


train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size


seed = torch.Generator().manual_seed(42)
train_set, valid_set = random_split(train_set, [train_set_size, valid_set_size], generator=seed)

train_loader = DataLoader(train_set, num_workers=4)
valid_loader = DataLoader(valid_set, num_workers=4)

autoencoder = LitAutoEncoder(Encoder(), Decoder())

print("start training")

trainer = pl.Trainer(
    accelerator="cuda",
    max_epochs=1,
    num_nodes=5,
    strategy=ddp.DDPStrategy(
        find_unused_parameters=False,
        cluster_environment=TorchElasticEnvironment(),
    ),
)

start = timer()
trainer.fit(autoencoder, train_loader, valid_loader)
print(f"total time: {timer() - start}")
