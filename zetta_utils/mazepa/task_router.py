from collections import defaultdict
from typing import Iterable, TypeVar

import attrs
from typeguard import typechecked

from zetta_utils.mazepa.tasks import Task
from zetta_utils.message_queues.base import PushMessageQueue

T = TypeVar("T")


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
            matching_queue_names = [
                queue.name for queue in self.queues if all(tag in queue.name for tag in task.tags)
            ]
            if len(matching_queue_names) == 0:
                raise RuntimeError(
                    f"No queue from set {list(self.queues)} matches " f"all tags {task.tags}."
                )
            tasks_for_queue[matching_queue_names[0]].append(task)

        for queue in self.queues:
            queue.push(tasks_for_queue[queue.name])
