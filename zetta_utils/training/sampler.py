from typing import Iterator

import torch.utils.data

from zetta_utils import builder

builder.register("TorchRandomSampler")(torch.utils.data.RandomSampler)


# Needed for DDP + RandomSampler to work with pytorch-lightning, which
# overwrites Sequential and RandomSampler with DistributedSampler.
# With the wrapper below, it will apply its own DistributedSamplerWrapper instead.
@builder.register("SamplerWrapper")
class SamplerWrapper(torch.utils.data.Sampler[int]):
    sampler: torch.utils.data.Sampler[int]

    def __init__(self, sampler: torch.utils.data.Sampler[int]) -> None:
        super().__init__(None)
        self.sampler = sampler
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed + self.epoch)
        self.sampler.generator = generator  # type: ignore[attr-defined]

        return iter(self.sampler)

    def __len__(self) -> int:
        return len(self.sampler)  # type: ignore

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
