from __future__ import annotations

from typing import Iterable

import attrs
from typeguard import typechecked

from zetta_utils import log
from zetta_utils.mazepa.worker import process_task
from zetta_utils.message_queues.base import MessageQueue, ReceivedMessage

from .task_outcome import OutcomeReport
from .tasks import Task

logger = log.get_logger("mazepa")


@typechecked
@attrs.mutable
class AutoexecuteTaskQueue(MessageQueue):
    name: str = "local_execution"
    tasks_todo: list[Task] = attrs.field(init=False, factory=list)
    debug: bool = False

    def push(self, payloads: Iterable[Task]):
        # TODO: Fix progress bar issue with multiple live displays in rich
        # for task in track(tasks, description="Local task execution..."):
        self.tasks_todo += list(payloads)

    def pull(
        self, max_num: int = 1, max_time_sec: float = 0  # pylint: disable=unused-argument
    ) -> list[ReceivedMessage[OutcomeReport]]:
        if max_time_sec != 0:
            raise NotImplementedError()

        results: list[ReceivedMessage[OutcomeReport]] = []

        for task in self.tasks_todo[:max_num]:
            # retries are counted for transient error handling
            curr_retry_count = 0
            while True:
                task.upkeep_settings.perform_upkeep = False
                finished_processing, outcome = process_task(
                    ReceivedMessage(
                        payload=task,
                        approx_receive_count=curr_retry_count,
                    ),
                    debug=self.debug,
                )
                if finished_processing:
                    task.outcome = outcome
                    break
                curr_retry_count += 1
            results.append(ReceivedMessage(OutcomeReport(task_id=task.id_, outcome=outcome)))
        self.tasks_todo = self.tasks_todo[max_num:]
        return results
