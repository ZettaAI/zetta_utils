import random
from typing import Any, Dict, Literal

import attrs
import torch
from typeguard import typechecked

from zetta_utils import builder, log

logger = log.get_logger("zetta_utils")


@builder.register("JointDataset")
@typechecked
@attrs.mutable
class JointDataset(torch.utils.data.Dataset):
    """PyTorch dataset wrapper to allow using multiple :class:`torch.utils.data.Dataset` datasets
    simultaneously.

    :param mode: String indicating whether the dataset is horizontally or vertically joined.
        ``horizontal`` means that the LayerDatasets will be sampled all at once and returned
        in a dictionary.
        ``vertical`` means that the LayerDatasets will be treated as a single, large dataset.
    :param datasets: Dictionary containing the datasets that make up the JointDataset.
    :param sampling_order: By default, samples will be drawn starting from the first dataset in the
    dictionary until all its samples are exhausted, before moving on to the next dataset.
        ``ascending`` is the default behavior, with samples drawn from one dataset after another.
        ``shuffle`` will randomly draw a sample (without replacement) from all available samples
        across all datasets.

    """

    mode: Literal["horizontal", "vertical"]
    datasets: Dict[str, Any]
    # TODO: Make torch.utils.data.Dataset pass mypy checks
    # datasets: Dict[str, torch.utils.data.Dataset]
    sampling_order: Literal["ascending", "shuffle"] = "ascending"
    _order: list[int] = attrs.field(init=False)

    def __attrs_pre_init__(self):
        super().__init__()

    def __attrs_post_init__(self):
        num_samples = len(self)
        if self.mode == "horizontal":
            for key in self.datasets.keys():
                if num_samples == len(self.datasets[key]):
                    logger.warning(
                        f"JointDataset: Dataset '{key}' has {len(self.datasets[key])} samples, "
                        f"which is the minimum number of samples for this horizontally joint "
                        "dataset."
                    )
                if num_samples < len(self.datasets[key]):
                    logger.warning(
                        f"JointDataset: Dataset '{key}' has {len(self.datasets[key])} samples, "
                        f"but only {num_samples} samples will be used."
                    )

        self._order = list(range(num_samples))
        if self.sampling_order == "shuffle":
            random.shuffle(self._order)

    def __len__(self) -> int:
        if self.mode == "horizontal":
            num_samples = min([len(d) for d in self.datasets.values()])

        elif self.mode == "vertical":
            num_samples = sum([len(d) for d in self.datasets.values()])
        else:
            assert False, "Type checker error."  # pragma: no cover

        return num_samples

    def __getitem__(self, idx: int) -> Any:
        if self.sampling_order == "shuffle":
            idx = self._order[idx]

        if self.mode == "horizontal":
            sample = {}
            for key, dset in self.datasets.items():
                sample[key] = dset[idx]

        elif self.mode == "vertical":
            sum_num_samples = 0
            for dset in self.datasets.values():
                sum_num_samples_new = sum_num_samples + len(dset)
                if idx < sum_num_samples_new:
                    sample = dset[idx - sum_num_samples]
                    break
                sum_num_samples = sum_num_samples_new

        return sample
