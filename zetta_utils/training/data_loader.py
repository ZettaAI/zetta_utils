import torch
from torch.utils.data._utils.collate import default_collate

from zetta_utils import builder
from zetta_utils.layer.volumetric.annotation.annotations import Annotation


def _collate_with_annotations(batch):
    """
    Custom collate function that handles annotation objects.

    For most data types, uses default PyTorch collation.
    For lists of Annotation objects, keeps them as nested lists
    (one list per batch item) rather than trying to stack them.
    """
    if isinstance(batch[0], dict):
        result = {}
        for key in batch[0]:
            values = [d[key] for d in batch]
            # Check if this is a list of annotations
            if (
                isinstance(values[0], (list, tuple))
                and len(values[0]) > 0
                and isinstance(values[0][0], Annotation)
            ):
                # Keep as list of lists (one per batch item)
                result[key] = values
            elif isinstance(values[0], Annotation):
                # Single annotation per sample - keep as list
                result[key] = values
            else:
                # Default collation for tensors, numpy arrays, etc.
                result[key] = default_collate(values)
        return result
    else:
        return default_collate(batch)


@builder.register("TorchDataLoader")
class TorchDataLoader(torch.utils.data.DataLoader):
    """DataLoader that also handles annotation objects in collation."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("collate_fn", _collate_with_annotations)
        super().__init__(*args, **kwargs)
