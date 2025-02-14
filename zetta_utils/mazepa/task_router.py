from collections import defaultdict
from typing import Iterable, TypeVar

import attrs
from typeguard import typechecked

from zetta_utils.mazepa.tasks import Task
from zetta_utils.message_queues.base import PushMessageQueue

T = TypeVar("T")


def _is_compatible_task(task: Task, queue_name: str) -> bool:
    return (
        task.worker_type is None
        or queue_name.startswith("local_")
        or f"_{task.worker_type}" in queue_name
    )


@typechecked
@attrs.frozen
class TaskRouter(PushMessageQueue[Task]):
    name: str = attrs.field(init=False)
    queues: Iterable[PushMessageQueue[Task]]

    def __attrs_post_init__(self):
        name = "_".join(queue.name for queue in self.queues)
        object.__setattr__(self, "name", name)

    def push(self, payloads: Iterable[Task]):
        tasks_for_queue = defaultdict(list)
        for task in payloads:
            task_pushed = False
            for queue in self.queues:
                if _is_compatible_task(task, queue.name):
                    tasks_for_queue[queue.name].append(task)
                    task_pushed = True
                    break
            if not task_pushed:
                raise RuntimeError(
                    f"No queue from set {list(self.queues)} has the right worker "
                    f"type {task.worker_type}."
                )

        for queue in self.queues:
            queue.push(tasks_for_queue[queue.name])
