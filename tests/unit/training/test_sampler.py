from lightning_fabric import seed_everything
from torch.utils.data import RandomSampler

from zetta_utils.training.sampler import SamplerWrapper


def test_sampler_wrapper():
    sampler = RandomSampler(list(range(100)))
    wrapper = SamplerWrapper(sampler)

    assert len(wrapper) == len(sampler)

    wrapper.set_epoch(0)
    seed_everything(42)
    epoch_0 = list(wrapper)
    seed_everything(42)
    assert list(wrapper) == epoch_0

    wrapper.set_epoch(1)
    seed_everything(42)
    assert list(wrapper) != epoch_0
