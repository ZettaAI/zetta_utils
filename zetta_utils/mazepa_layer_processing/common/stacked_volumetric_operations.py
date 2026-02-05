from __future__ import annotations

import time
from typing import Any, Generic, Sequence, cast

import attrs
import numpy as np
import torch
from typing_extensions import ParamSpec

from zetta_utils import builder, log, mazepa
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer

from ..operation_protocols import StackableVolumetricOpProtocol

logger = log.get_logger("mazepa")
P = ParamSpec("P")


@builder.register("StackedVolumetricOperations")
@mazepa.taskable_operation_cls
@attrs.mutable
class StackedVolumetricOperations(Generic[P]):
    """
    Wrapper that processes multiple indices for a StackableVolumetricOpProtocol operation.
    Optimizes I/O by reading and writing multiple chunks in batch.

    The batching is controlled by the flow that creates tasks with this operation.
    This operation processes all indices it receives in a single batch.

    :param base_op: The stackable operation to wrap.
    """

    base_op: StackableVolumetricOpProtocol[P, None, VolumetricLayer]

    def __attrs_post_init__(self):
        if not isinstance(self.base_op, StackableVolumetricOpProtocol):
            raise TypeError(
                f"{type(self.base_op).__name__} does not implement StackableVolumetricOpProtocol. "
                f"Missing read/write methods required for stacking."
            )

    def get_operation_name(self) -> str:
        base_name = (
            self.base_op.get_operation_name()
            if hasattr(self.base_op, "get_operation_name")
            else type(self.base_op).__name__
        )
        return f"Stacked({base_name})"

    def get_input_resolution(self, dst_resolution: Vec3D) -> Vec3D:
        return self.base_op.get_input_resolution(dst_resolution)

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> StackedVolumetricOperations[P]:
        return attrs.evolve(self, base_op=self.base_op.with_added_crop_pad(crop_pad))

    def __call__(  # pylint: disable=keyword-arg-before-vararg,too-many-branches
        self,
        indices: Sequence[VolumetricIndex],
        dsts: Sequence[VolumetricLayer] | VolumetricLayer,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """
        Process multiple indices in a single batch.

        :param indices: List of VolumetricIndex objects to process.
        :param dsts: Destination layer(s) for writing results. Can be a single layer
            (same for all indices) or a sequence of layers (one per index).
        :param args: Additional positional arguments passed to base_op.
        :param kwargs: Keyword arguments passed to base_op for reading source data.
        """
        if len(indices) == 0:
            return

        # Normalize dsts to a sequence
        if not isinstance(dsts, Sequence):
            dsts_list = [dsts] * len(indices)
        else:
            dsts_list = list(dsts)
            if len(dsts_list) != len(indices):
                raise ValueError(
                    f"Length of dsts ({len(dsts_list)}) must match "
                    f"length of indices ({len(indices)})"
                )

        # Read all data
        read_start = time.time()
        data_list = [self.base_op.read(idx, *args, **kwargs) for idx in indices]
        read_time = time.time() - read_start

        # Stack tensors by key
        # Assumes all dicts have the same keys
        if not data_list:
            return

        keys = data_list[0].keys()
        stacked_kwargs: dict[str, Any] = {}

        for key in keys:
            tensors = [d[key] for d in data_list]

            # Stack tensors
            if isinstance(tensors[0], torch.Tensor):
                stacked_kwargs[key] = torch.stack(cast(list[torch.Tensor], tensors), dim=0)
            elif isinstance(tensors[0], np.ndarray):
                stacked_kwargs[key] = np.stack(cast(list[np.ndarray], tensors), axis=0)
            else:
                raise TypeError(
                    f"Read method returned unsupported type for key '{key}': {type(tensors[0])}. "
                    f"Only torch.Tensor and np.ndarray are supported for stacking."
                )

        # Process the batch with the base operation's processing function
        process_start = time.time()
        batched_result: Any = self.base_op.processing_fn(**stacked_kwargs)
        process_time = time.time() - process_start

        # Unstack and write results
        write_start = time.time()
        for i, (idx, dst) in enumerate(zip(indices, dsts_list)):
            result: Any
            if isinstance(batched_result, torch.Tensor):
                result = batched_result[i]
            elif isinstance(batched_result, np.ndarray):
                result = batched_result[i]
            else:
                raise TypeError(
                    f"Function returned unsupported type: {type(batched_result)}. "
                    f"Only torch.Tensor and np.ndarray are supported."
                )

            self.base_op.write(idx, dst, result, *args, **kwargs)
        write_time = time.time() - write_start

        total_time = read_time + process_time + write_time
        logger.info(
            f"StackedVolumetricOperations: Total time for {len(indices)} chunks: {total_time:.2f}s"
            f" (read: {read_time:.2f}s, process: {process_time:.2f}s, write: {write_time:.2f}s)"
        )
