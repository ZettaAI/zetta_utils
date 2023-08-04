from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional, Set, Union

import attrs
from typeguard import typechecked

from zetta_utils import log
from zetta_utils.mazepa import constants

from .exceptions import MazepaExecutionFailure
from .execution_checkpoint import read_execution_checkpoint
from .flows import Dependency, Flow
from .task_outcome import TaskOutcome, TaskStatus
from .tasks import Task

logger = log.get_logger("mazepa")


@attrs.frozen
class ProgressReport:
    submitted_count: int
    completed_count: int


class ExecutionState(ABC):
    raise_on_failed_task: bool = True

    @abstractmethod
    def get_ongoing_flows(self) -> list[Flow]:
        ...

    @abstractmethod
    def __init__(self, ongoing_flows: list[Flow], raise_on_failed_task: bool = True):
        ...

    @abstractmethod
    def get_ongoing_flow_ids(self) -> list[str]:
        ...

    @abstractmethod
    def update_with_task_outcomes(self, task_outcomes: dict[str, TaskOutcome]):
        ...

    @abstractmethod
    def get_task_batch(self, max_batch_len: int = ...) -> list[Task]:
        ...

    @abstractmethod
    def get_progress_reports(self) -> dict[str, ProgressReport]:
        ...

    @abstractmethod
    def get_completed_ids(self) -> set[str]:
        ...


@typechecked
@attrs.mutable
class InMemoryExecutionState(ExecutionState):  # pylint: disable=too-many-instance-attributes
    """
    ``ExecutionState`` implementation that keeps progress and dependency information
    as in-memory data structures.
    """

    ongoing_flows: list[Flow]
    ongoing_flows_dict: dict[str, Flow] = attrs.field(init=False)
    ongoing_exhausted_flow_ids: Set[str] = attrs.field(init=False, factory=set)
    ongoing_parent_map: dict[str, Set[str]] = attrs.field(
        init=False, factory=lambda: defaultdict(set)
    )
    ongoing_children_map: dict[str, Set[str]] = attrs.field(
        init=False, factory=lambda: defaultdict(set)
    )
    ongoing_tasks_dict: dict[str, Task] = attrs.field(init=False, factory=dict)
    raise_on_failed_task: bool = True
    completed_ids: Set[str] = attrs.field(
        init=False,
        factory=set,
    )
    dependency_map: dict[str, Set[str]] = attrs.field(init=False, factory=lambda: defaultdict(set))
    submitted_counts: dict[str, int] = attrs.field(init=False, factory=lambda: defaultdict(int))
    completed_counts: dict[str, int] = attrs.field(init=False, factory=lambda: defaultdict(int))
    leftover_ready_tasks: list[Task] = attrs.field(init=False, factory=list)
    checkpoint: Optional[str] = None

    def __attrs_post_init__(self):
        self.ongoing_flows_dict = {e.id_: e for e in self.ongoing_flows}
        if self.checkpoint is not None:
            self._load_completed_ids_from_file(self.checkpoint)

    def get_ongoing_flows(self) -> list[Flow]:
        return self.ongoing_flows

    def get_progress_reports(self) -> dict[str, ProgressReport]:
        result: dict[str, ProgressReport] = {}
        for op_name, submitted_count in self.submitted_counts.items():
            completed_count = 0
            if op_name in self.completed_counts:
                completed_count = self.completed_counts[op_name]

            result[op_name] = ProgressReport(
                submitted_count=submitted_count,
                completed_count=completed_count,
            )

        return result

    def get_ongoing_flow_ids(self) -> list[str]:
        """
        Return ids of the flows that haven't been completed.
        """
        return list(self.ongoing_flows_dict.keys())

    def update_with_task_outcomes(self, task_outcomes: dict[str, TaskOutcome]):
        """
        Given a mapping from tasks ids to task outcomes, update dependency and state of the
        execution. If any of the task outcomes indicates failure, will raise exception specified
        in the task outcome.

        :param task_ids: IDs of tasks indicated as completed.
        """
        for task_id, outcome in task_outcomes.items():
            if task_id == constants.UNKNOWN_TASK_ID:
                assert outcome.exception is not None
                if self.raise_on_failed_task:
                    logger.error(f"Task traceback: {outcome.traceback_text}")
                    raise MazepaExecutionFailure("Task failure.")
            elif task_id in self.ongoing_tasks_dict:
                self.ongoing_tasks_dict[task_id].outcome = outcome
                if outcome.exception is None:
                    self.ongoing_tasks_dict[task_id].status = TaskStatus.SUCCEEDED
                else:
                    self.ongoing_tasks_dict[task_id].status = TaskStatus.FAILED
                    if self.raise_on_failed_task:
                        logger.error(f"Task traceback: {outcome.traceback_text}")
                        raise MazepaExecutionFailure("Task failure.")

                self._update_completed_id(task_id)

    def get_task_batch(self, max_batch_len: int = 10000) -> list[Task]:
        """
        Generate the next batch of tasks that are ready for execution.

        :param max_batch_len: size limit after which no more flows will be querries for
            additional tasks. Note that the return length might be larger than
            ``max_batch_len``, as individual flows batches may not be subdivided.
        """

        result = self.leftover_ready_tasks  # type: list[Task]
        candidate_flows = list(self.ongoing_flows_dict.values())
        while len(candidate_flows) > 0:
            newly_added_flows = []
            for flow in candidate_flows:
                while (
                    flow.id_ in self.ongoing_flows_dict
                    and len(self.dependency_map[flow.id_]) == 0
                    and len(result) < max_batch_len
                    and flow.id_ not in self.ongoing_exhausted_flow_ids
                ):
                    flow_batch = self._get_batch_from_flow(flow)
                    for e in flow_batch:
                        if isinstance(e, Flow):
                            self.ongoing_flows_dict[e.id_] = e
                            newly_added_flows.append(e)
                        else:
                            assert isinstance(e, Task), "Typechecking error."
                            result.append(e)

                    if len(result) >= max_batch_len:
                        break

            if len(result) >= max_batch_len:
                break
            candidate_flows = newly_added_flows

        result_final = result[:max_batch_len]
        self.leftover_ready_tasks = result[max_batch_len:]

        for e in result_final:
            if e.id_ not in self.ongoing_tasks_dict:
                # Duplicate tasks don't count
                self.ongoing_tasks_dict[e.id_] = e
                self.submitted_counts[e.operation_name] += 1

        return result_final

    def get_completed_ids(self) -> set[str]:
        return self.completed_ids

    def _add_dependency(self, flow_id: str, dep: Dependency):
        if dep.ids is None:  # depend on all ongoing children
            self.dependency_map[flow_id].update(self.ongoing_children_map[flow_id])
        else:
            for id_ in dep.ids:
                if id_ not in self.completed_ids:
                    assert (
                        id_ in self.ongoing_children_map[flow_id]
                    ), f"Dependency on a non-child '{id_}' for flows '{flow_id}'"

                    self.dependency_map[flow_id].add(id_)

    def _update_completed_id(self, id_: str):
        self.completed_ids.add(id_)
        self.ongoing_exhausted_flow_ids.discard(id_)

        if id_ in self.ongoing_flows_dict:
            del self.ongoing_flows_dict[id_]
        else:
            assert id_ in self.ongoing_tasks_dict
            self.completed_counts[self.ongoing_tasks_dict[id_].operation_name] += 1
            del self.ongoing_tasks_dict[id_]

        parent_ids = self.ongoing_parent_map[id_]
        for parent_id in parent_ids:
            self.ongoing_children_map[parent_id].discard(id_)
            self.dependency_map[parent_id].discard(id_)
            if (
                parent_id in self.ongoing_exhausted_flow_ids
                and len(self.dependency_map[parent_id]) == 0
            ):
                self._update_completed_id(parent_id)

    def _get_batch_from_flow(self, flow: Flow) -> list[Union[Task, Flow]]:
        """
        Returns a batch of ready tasks from the flow.
        If the flow yields children flows, the children flows will be added
        to the execution state, and won't be returned by this function.
        If the flow yields dependencies, the dependencies will be added
        to the execution state, and won't be returned by this function.
        """
        flow_yield = flow.get_next_batch()

        result = []
        if flow_yield is None:  # Means the flows is exhausted
            self.ongoing_exhausted_flow_ids.add(flow.id_)
            self.dependency_map[flow.id_].update(self.ongoing_children_map[flow.id_])
            if len(self.dependency_map[flow.id_]) == 0:
                self._update_completed_id(flow.id_)

        elif isinstance(flow_yield, Dependency):
            self._add_dependency(flow.id_, flow_yield)
        else:
            for e in flow_yield:
                if e.id_ not in self.completed_ids:
                    self.ongoing_children_map[flow.id_].add(e.id_)
                    self.ongoing_parent_map[e.id_].add(flow.id_)
                    result.append(e)
                elif isinstance(e, Task):
                    # Task loaded from checkpoint - adjust the counter
                    self.submitted_counts[e.operation_name] += 1
                    self.completed_counts[e.operation_name] += 1
        return result

    def _load_completed_ids_from_file(self, filepath: str):
        completed_ids = read_execution_checkpoint(filepath, ignore_prefix="flow-")

        logger.info(f"Updating {len(completed_ids)} completed tasks from {self.checkpoint}")
        self.completed_ids.update(completed_ids)
