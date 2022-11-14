from typing import Any, Dict, Literal

import attrs
import torch
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.log import logger


@builder.register("JointDataset")
@typechecked
@attrs.frozen
class JointDataset(torch.utils.data.Dataset):
    """PyTorch dataset wrapper to allow using multiple ``torch.utils.data.Dataset`` datasets
    simultaneously.

    :param mode: String indicating whether the dataset is horizontally or vertically joined.
        ``horizontal`` means that the LayerDatasets will be sampled all at once and returned
        in a dictionary.
        ``vertical`` means that the LayerDatasets will be sampled one after the other in the order
        given during initialization.
    :param datasets: Dictionary containing the datasets that make up the JointDataset.

    """

    mode: Literal["horizontal", "vertical"]
    datasets: Dict[str, Any]
    # TODO: Make torch.utils.data.Dataset pass mypy checks
    # datasets: Dict[str, torch.utils.data.Dataset]

    def __attrs_pre_init__(self):
        super().__init__()

    def __attrs_post_init__(self):
        if self.mode == "horizontal":
            num_samples = min([len(d) for d in self.datasets.values()])
            for key in self.datasets.keys():
                if num_samples == len(self.datasets[key]):
                    logger.warning(
                        f"JointDataset: Dataset {key} has {len(self.datasets[key])} samples, "
                        f"which is the minimum number of samples for this horizontally joint "
                        "dataset."
                    )
                if num_samples < len(self.datasets[key]):
                    logger.warning(
                        f"JointDataset: Dataset {key} has {len(self.datasets[key])} samples, "
                        f"but only {num_samples} samples will be used."
                    )

    def __len__(self) -> int:
        if self.mode == "horizontal":
            num_samples = min([len(d) for d in self.datasets.values()])

        elif self.mode == "vertical":
            num_samples = sum([len(d) for d in self.datasets.values()])
        else:
            assert False, "Type checker error."  # pragma: no cover

        return num_samples

    def __getitem__(self, idx: int) -> Any:
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
