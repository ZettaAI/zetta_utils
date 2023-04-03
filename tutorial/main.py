# pylint: disable=missing-docstring # type: ignore

import os
from timeit import default_timer as timer

import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from .module import Decoder, Encoder, LitAutoEncoder

dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset, num_workers=4)


autoencoder = LitAutoEncoder(Encoder(), Decoder())

trainer = pl.Trainer(
    accelerator="cuda",
    devices=1,
    max_epochs=1,
    strategy="ddp",
)

start = timer()
trainer.fit(model=autoencoder, train_dataloaders=train_loader)
print(f"total time: {timer() - start}")
