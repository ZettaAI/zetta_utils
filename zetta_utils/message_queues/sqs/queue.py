from __future__ import annotations

import json
from typing import Any, Sequence, TypeVar

import attrs
import taskqueue
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.common.partial import ComparablePartial
from zetta_utils.message_queues.base import MessageQueue

from .. import ReceivedMessage, serialization
from . import utils


class TQTask(taskqueue.RegisteredTask):
    """
    Wrapper that makes Mazepa tasks submittable with `python-task-queue`.
    """

    def __init__(self, task_ser: str):
        super().__init__(
            task_ser=task_ser,
        )

    def execute(self):  # pragma: no cover
        raise NotImplementedError()


def _delete_task_message(
    receipt_handle: str,
    queue_name: str,
    region_name: str,
    endpoint_url: str | None = None,
):
    utils.delete_msg_by_receipt_handle(
        receipt_handle=receipt_handle,
        queue_name=queue_name,
        region_name=region_name,
        endpoint_url=endpoint_url,
    )


T = TypeVar("T")


@builder.register("SQSQueue")
@typechecked
@attrs.mutable
class SQSQueue(MessageQueue[T]):
    name: str
    region_name: str = attrs.field(default=taskqueue.secrets.AWS_DEFAULT_REGION)
    endpoint_url: str | None = None
    insertion_threads: int = 5
    _queue: Any = attrs.field(init=False, default=None)
    pull_wait_sec: int = 0
    pull_lease_sec: int = 10  # TODO: get a better value

    def _get_tq_queue(self) -> Any:
        if self._queue is None:
            self._queue = taskqueue.TaskQueue(
                self.name,
                region_name=self.region_name,
                endpoint_url=self.endpoint_url,
                green=False,
                n_threads=self.insertion_threads,
            )
        return self._queue

    def push(self, payloads: Sequence[T]) -> None:
        if len(payloads) > 0:
            tq_tasks = []
            for e in payloads:
                tq_task = TQTask(serialization.serialize(e))
                tq_tasks.append(tq_task)
            self._get_tq_queue().insert(tq_tasks, parallel=self.insertion_threads)

    def _extend_msg_lease(self, duration_sec: int, msg: utils.SQSReceivedMsg):
        utils.change_message_visibility(
            receipt_handle=msg.receipt_handle,
            queue_name=self.name,
            region_name=self.region_name,
            endpoint_url=self.endpoint_url,
            visibility_timeout=duration_sec,
        )

    def pull(self, max_num: int = 1) -> list[ReceivedMessage[T]]:
        results = []
        msgs = utils.receive_msgs(
            queue_name=self.name,
            region_name=self.region_name,
            endpoint_url=self.endpoint_url,
            max_msg_num=max_num,
            max_time_sec=self.pull_wait_sec,
            visibility_timeout=self.pull_lease_sec,
        )

        for msg in msgs:
            # Deserialize task object
            tq_task = taskqueue.totask(json.loads(msg.body))
            payload = serialization.deserialize(tq_task.task_ser)
            acknowledge_fn = ComparablePartial(
                _delete_task_message,
                receipt_handle=msg.receipt_handle,
                queue_name=self.name,
                region_name=self.region_name,
                endpoint_url=self.endpoint_url,
            )

            extend_lease_fn = ComparablePartial(
                self._extend_msg_lease,
                msg=msg,
            )

            result = ReceivedMessage[T](
                payload=payload,
                approx_receive_count=msg.approx_receive_count - 1,
                acknowledge_fn=acknowledge_fn,
                extend_lease_fn=extend_lease_fn,
            )

            results.append(result)
        return results
