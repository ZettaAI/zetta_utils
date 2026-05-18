from __future__ import annotations

import json
import threading
from typing import Any, Sequence, TypeVar

import attrs
import taskqueue
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.common.partial import ComparablePartial
from zetta_utils.message_queues.base import MessageQueue

from .. import ReceivedMessage, TQTask, serialization
from . import utils


@attrs.mutable
class _BatchDeleter:
    """
    Thread-safe accumulator that batches receipt handles for a single SQS
    queue and issues ``delete_message_batch`` calls of up to
    ``utils.DELETE_BATCH_SIZE`` entries.
    """

    queue_name: str
    region_name: str
    endpoint_url: str | None = None
    expected: int = 0
    _pending: list[str] = attrs.field(factory=list)
    _added: int = 0
    _lock: threading.Lock = attrs.field(factory=threading.Lock)

    def add(self, receipt_handle: str) -> None:
        to_flush: list[list[str]] = []
        with self._lock:
            self._pending.append(receipt_handle)
            self._added += 1
            while len(self._pending) >= utils.DELETE_BATCH_SIZE:
                to_flush.append(self._pending[: utils.DELETE_BATCH_SIZE])
                self._pending = self._pending[utils.DELETE_BATCH_SIZE :]
            if self.expected > 0 and self._added >= self.expected and self._pending:
                to_flush.append(self._pending)
                self._pending = []
        for chunk in to_flush:
            self._send(chunk)

    def flush(self) -> None:
        with self._lock:
            to_flush = self._pending
            self._pending = []
        if to_flush:
            for i in range(0, len(to_flush), utils.DELETE_BATCH_SIZE):
                self._send(to_flush[i : i + utils.DELETE_BATCH_SIZE])

    def _send(self, receipts: list[str]) -> None:
        utils.delete_msg_batch(
            receipt_handles=receipts,
            queue_name=self.queue_name,
            region_name=self.region_name,
            endpoint_url=self.endpoint_url,
        )


def _delete_task_message(
    receipt_handle: str,
    deleter: _BatchDeleter,
):
    deleter.add(receipt_handle)


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
    pull_lease_sec: int = 30

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

    def pull(self, max_num: int = 500) -> list[ReceivedMessage[T]]:
        results = []
        msgs = utils.receive_msgs(
            queue_name=self.name,
            region_name=self.region_name,
            endpoint_url=self.endpoint_url,
            max_msg_num=max_num,
            max_time_sec=self.pull_wait_sec,
            visibility_timeout=self.pull_lease_sec,
        )

        deleter = _BatchDeleter(
            queue_name=self.name,
            region_name=self.region_name,
            endpoint_url=self.endpoint_url,
            expected=len(msgs),
        )

        for msg in msgs:
            tq_task = taskqueue.totask(json.loads(msg.body))
            payload = serialization.deserialize(tq_task.task_ser)
            acknowledge_fn = ComparablePartial(
                _delete_task_message,
                receipt_handle=msg.receipt_handle,
                deleter=deleter,
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
