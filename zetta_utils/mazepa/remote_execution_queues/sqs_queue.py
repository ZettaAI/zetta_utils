from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import attrs
import taskqueue
from typeguard import typechecked

# from zetta_utils.log import logger
from zetta_utils.partial import ComparablePartial

from .. import Task, TaskOutcome, serialization
from . import sqs_utils


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


@attrs.frozen
class OutcomeReport:
    task_id: str
    outcome: TaskOutcome


def _send_outcome_report(
    task: Task, queue_name: str, region_name: str, endpoint_url: Optional[str] = None
):
    sqs_utils.send_msg(
        queue_name=queue_name,
        region_name=region_name,
        endpoint_url=endpoint_url,
        msg_body=serialization.serialize(OutcomeReport(task_id=task.id_, outcome=task.outcome)),
    )


def _delete_task_message(
    task: Task,  # pylint: disable=unused-argument
    receipt_handle: str,
    queue_name: str,
    region_name: str,
    endpoint_url: Optional[str] = None,
):
    sqs_utils.delete_msg_by_receipt_handle(
        receipt_handle=receipt_handle,
        queue_name=queue_name,
        region_name=region_name,
        endpoint_url=endpoint_url,
    )


@typechecked
@attrs.mutable
class SQSExecutionQueue:
    name: str
    region_name: str = attrs.field(default=taskqueue.secrets.AWS_DEFAULT_REGION)
    endpoint_url: Optional[str] = None
    insertion_threads: int = 0
    outcome_queue_name: Optional[str] = None
    _queue: Any = attrs.field(init=False)
    pull_wait_sec: int = 0
    pull_lease_sec: int = 30

    def __attrs_post_init__(self):
        # Use TaskQueue for fast insertion
        self._queue = taskqueue.TaskQueue(
            self.name, region_name=self.region_name, endpoint_url=self.endpoint_url, green=False
        )

    def purge(self):  # pragma: no cover
        raise NotImplementedError()

    def push_tasks(self, tasks: Iterable[Task]):
        if self.outcome_queue_name is None:
            raise RuntimeError("Outcome queue name not specified.")

        for task in tasks:
            task._mazepa_callbacks.append(  # pylint: disable=protected-access
                ComparablePartial(
                    _send_outcome_report,
                    queue_name=self.outcome_queue_name,
                    region_name=self.region_name,
                    endpoint_url=self.endpoint_url,
                )
            )
        tq_tasks = [TQTask(serialization.serialize(e)) for e in tasks]
        self._queue.insert(tq_tasks, parallel=self.insertion_threads)

    def pull_task_outcomes(
        self, max_num: int = 100, max_time_sec: float = 2.5
    ) -> Dict[str, TaskOutcome]:
        if self.outcome_queue_name is None:
            raise RuntimeError(
                "Attempting to pull task oucomes without outcome queue beign specified"
            )

        msgs = sqs_utils.receive_msgs(
            queue_name=self.outcome_queue_name,
            region_name=self.region_name,
            endpoint_url=self.endpoint_url,
            max_msg_num=max_num,
            max_time_sec=max_time_sec,
        )
        task_outcomes = [serialization.deserialize(msg.body) for msg in msgs]
        result = {e.task_id: e.outcome for e in task_outcomes}
        sqs_utils.delete_received_msgs(msgs)

        return result

    def pull_tasks(self, max_num: int = 1):
        try:
            tq_tasks = self._queue.lease(
                seconds=self.pull_lease_sec, num_tasks=max_num, wait_sec=self.pull_wait_sec
            )
        except taskqueue.taskqueue.QueueEmptyError:
            tq_tasks = []

        if not isinstance(tq_tasks, list):
            tq_tasks = [tq_tasks]

        tasks = []
        for tq_task in tq_tasks:
            task = serialization.deserialize(tq_task.task_ser)
            task._mazepa_callbacks.append(  # pylint: disable=protected-access
                ComparablePartial(
                    _delete_task_message,
                    receipt_handle=tq_task.id,
                    queue_name=self.name,
                    region_name=self.region_name,
                    endpoint_url=self.endpoint_url,
                )
            )
            tasks.append(task)
        return tasks
