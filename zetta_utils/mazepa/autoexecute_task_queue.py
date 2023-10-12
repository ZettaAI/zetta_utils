from __future__ import annotations

from typing import Iterable

import attrs
from typeguard import typechecked

from zetta_utils import log
from zetta_utils.mazepa.worker import process_task_message
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
    handle_exceptions: bool = False

    def push(self, payloads: Iterable[Task]):
        # TODO: Fix progress bar issue with multiple live displays in rich
        # for task in track(tasks, description="Local task execution..."):
        self.tasks_todo += list(payloads)

    def pull(
        self,
        max_num: int = 128,
        max_time_sec: float = 0,
    ) -> list[ReceivedMessage[OutcomeReport]]:
        if max_time_sec != 0:
            raise NotImplementedError()
        if len(self.tasks_todo) == 0:
            return []
        else:
            results: list[ReceivedMessage[OutcomeReport]] = []
            for task in self.tasks_todo[:max_num]:
                results.append(execute_task(task, self.debug, self.handle_exceptions))
            self.tasks_todo = self.tasks_todo[max_num:]
            return results


def execute_task(
    task: Task, debug: bool, handle_exceptions: bool
) -> ReceivedMessage[OutcomeReport]:
    # retries are counted for transient error handling
    curr_retry_count = 0

    while True:
        task.upkeep_settings.perform_upkeep = False
        finished_processing, outcome = process_task_message(
            ReceivedMessage(
                payload=task,
                approx_receive_count=curr_retry_count,
            ),
            debug=debug,
            handle_exceptions=handle_exceptions,
        )
        if finished_processing:
            task.outcome = outcome
            break
        curr_retry_count += 1
    return ReceivedMessage(OutcomeReport(task_id=task.id_, outcome=outcome))
