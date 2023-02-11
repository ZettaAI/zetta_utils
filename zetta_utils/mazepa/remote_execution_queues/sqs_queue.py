from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

import attrs
import taskqueue
from typeguard import typechecked

# from zetta_utils.log import logger
from zetta_utils.common.partial import ComparablePartial

from .. import Task, TaskOutcome, constants, serialization
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
    assert task.outcome is not None
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
    insertion_threads: int = 5
    outcome_queue_name: Optional[str] = None
    _queue: Any = attrs.field(init=False, default=None)
    pull_wait_sec: int = 0
    # This pull lease sec will work for tasks of ANY duration. Change only if expert/debugging
    pull_lease_sec: int = constants.DEFAULT_UPKEEP_INTERVAL * constants.DEFAULT_UPKEEPS_PER_LEASE

    def _get_tq_queue(self) -> Any:
        if self._queue is None:
            # Use TaskQueue for fast insertion
            self._queue = taskqueue.TaskQueue(
                self.name,
                region_name=self.region_name,
                endpoint_url=self.endpoint_url,
                green=False,
                n_threads=self.insertion_threads,
            )
        return self._queue

    def push_tasks(self, tasks: List[Task]):
        if len(tasks) == 0:
            return  # pragma: no cover
        if self.outcome_queue_name is None:
            raise RuntimeError("Outcome queue name not specified.")

        for task in tasks:
            task.completion_callbacks.append(  # pylint: disable=protected-access
                ComparablePartial(
                    _send_outcome_report,
                    queue_name=self.outcome_queue_name,
                    region_name=self.region_name,
                    endpoint_url=self.endpoint_url,
                )
            )
        tq_tasks = []
        for task in tasks:
            tq_task = TQTask(serialization.serialize(task))
            tq_tasks.append(tq_task)
        self._get_tq_queue().insert(tq_tasks, parallel=self.insertion_threads)

    def pull_task_outcomes(
        self, max_num: int = 500, max_time_sec: float = 2.5
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
        tasks = []
        msgs = sqs_utils.receive_msgs(
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
            task = serialization.deserialize(tq_task.task_ser)

            # Specify completion behavior through callbacks
            task.completion_callbacks.append(
                ComparablePartial(
                    _delete_task_message,
                    receipt_handle=msg.receipt_handle,
                    queue_name=self.name,
                    region_name=self.region_name,
                    endpoint_url=self.endpoint_url,
                )
            )

            # Specify upkeep behavior through callbacks
            if task.upkeep_settings.perform_upkeep:
                task.upkeep_settings.callbacks.append(
                    ComparablePartial(
                        sqs_utils.change_message_visibility,
                        receipt_handle=msg.receipt_handle,
                        queue_name=self.name,
                        region_name=self.region_name,
                        endpoint_url=self.endpoint_url,
                        visibility_timeout=math.ceil(task.upkeep_settings.interval_sec * 5),
                    )
                )

            tasks.append(task)
        return tasks
