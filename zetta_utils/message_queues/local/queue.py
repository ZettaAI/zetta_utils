from __future__ import annotations

import os
import shutil
from typing import Any, Sequence, TypeVar

import attrs
import taskqueue
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.common.partial import ComparablePartial
from zetta_utils.common.path import abspath, strip_prefix
from zetta_utils.log import get_logger
from zetta_utils.message_queues.base import MessageQueue

from .. import ReceivedMessage, TQTask, serialization

logger = get_logger("zetta_utils")
T = TypeVar("T")


@builder.register("LocalQueue")
@typechecked
@attrs.mutable
class LocalQueue(MessageQueue[T]):
    name: str
    _queue: Any = attrs.field(init=False, default=None)
    pull_wait_sec: float = 0.5
    pull_lease_sec: int = 10  # TODO: get a better value

    def __enter__(self) -> LocalQueue:
        queue_path = abspath(f"fq://./localqueue_{self.name}")
        queue_folder_path = strip_prefix(queue_path)
        if os.path.exists(queue_folder_path):
            raise RuntimeError(
                "Could not create LocalQueue: " f"{queue_folder_path} already exists."
            )
        self._queue = taskqueue.TaskQueue(queue_path)
        logger.info(f"Initialised LocalQueue at `{queue_folder_path}`.")
        return self

    def __exit__(self, *args) -> None:
        queue_path = abspath(f"fq://./localqueue_{self.name}")
        queue_folder_path = strip_prefix(queue_path)
        shutil.rmtree(queue_folder_path)
        logger.info(f"Cleaned up LocalQueue at `{queue_folder_path}`.")

    def _get_tq_queue(self) -> Any:
        if self._queue is None:
            self._queue = taskqueue.TaskQueue(abspath(f"fq://./localqueue_{self.name}"))
        return self._queue

    def push(self, payloads: Sequence[T]) -> None:
        if len(payloads) > 0:
            tq_tasks = []
            for e in payloads:
                tq_task = TQTask(serialization.serialize(e))
                tq_tasks.append(tq_task)
            self._get_tq_queue().insert(tq_tasks)

    def _delete_task(self, task_id: str) -> None:
        self._get_tq_queue().delete(task_id)

    def _extend_task_lease(self, duration_sec: int, task: TQTask):
        self._get_tq_queue().renew(task, duration_sec)

    def pull(self, max_num: int = 500) -> list[ReceivedMessage[T]]:
        results: list[ReceivedMessage[T]] = []
        try:
            lease_result = self._get_tq_queue().lease(
                num_tasks=max_num,
                seconds=self.pull_lease_sec,
                wait_sec=self.pull_wait_sec,
            )
            tasks = [lease_result] if isinstance(lease_result, TQTask) else lease_result
        except taskqueue.QueueEmptyError:
            return results

        for task in tasks:
            # Deserialize task object
            payload = serialization.deserialize(task.task_ser)
            acknowledge_fn = ComparablePartial(
                self._delete_task,
                task_id=task.id,
            )

            extend_lease_fn = ComparablePartial(
                self._extend_task_lease,
                task=task,
            )

            result = ReceivedMessage[T](
                payload=payload,
                approx_receive_count=1,
                acknowledge_fn=acknowledge_fn,
                extend_lease_fn=extend_lease_fn,
            )

            results.append(result)
        return results
