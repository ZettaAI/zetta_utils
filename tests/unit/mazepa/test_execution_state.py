# type: ignore # We're breaking mypy here
from __future__ import annotations

from typing import Any

import pytest

from zetta_utils.mazepa import (
    Dependency,
    Flow,
    InMemoryExecutionState,
    TaskOutcome,
    TaskStatus,
)

from .maker_utils import make_test_flow, make_test_task


def dummy_iter(iterable):
    return iter(iterable)


@pytest.mark.parametrize(
    "flows, max_batch_len, expected_batch_ids, completed_task_ids, expected_ongoing_flow_ids",
    [
        [
            [
                make_test_flow(
                    fn=dummy_iter,
                    iterable=[
                        make_test_task(
                            fn=lambda: None,
                            id_=id_,
                        )
                        for id_ in ["a", "b", "c"]
                    ],
                    id_="flow_0",
                )
            ],
            10,
            [["a", "b", "c"]],
            [[]],
            [["flow_0"]],
        ],
        [
            [
                make_test_flow(
                    fn=dummy_iter,
                    iterable=[make_test_task(fn=lambda: None, id_=id_) for id_ in ["a", "b", "c"]],
                    id_="flow_0",
                )
            ],
            10,
            [["a", "b", "c"]],
            [["a", "b"]],
            [["flow_0"]],
        ],
        [
            [
                make_test_flow(
                    fn=dummy_iter,
                    iterable=[make_test_task(fn=lambda: None, id_=id_) for id_ in ["a", "b", "c"]],
                    id_="flow_0",
                )
            ],
            10,
            [["a", "b", "c"]],
            [["a", "b", "c"]],
            [[]],
        ],
        [
            [
                make_test_flow(
                    fn=dummy_iter,
                    iterable=[make_test_task(fn=lambda: None, id_=id_) for id_ in ["a", "b", "c"]],
                    id_="flow_0",
                )
            ],
            1,
            [["a"]],
            [["a"]],
            [["flow_0"]],
        ],
        [
            [
                make_test_flow(
                    fn=dummy_iter,
                    iterable=[
                        make_test_task(fn=lambda: None, id_="a"),
                        Dependency(),
                        make_test_task(fn=lambda: None, id_="b"),
                    ],
                    id_="flow_0",
                )
            ],
            10,
            [["a"], ["b"]],
            [["a"], ["b"]],
            [["flow_0"], []],
        ],
        [
            [
                make_test_flow(
                    fn=dummy_iter,
                    iterable=[
                        make_test_task(fn=lambda: None, id_="a"),
                        Dependency(),
                        make_test_task(fn=lambda: None, id_="b"),
                    ],
                    id_="flow_0",
                )
            ],
            10,
            [["a"], ["b"], []],
            [["a"], [], ["b"]],
            [["flow_0"], ["flow_0"], []],
        ],
        [
            [
                make_test_flow(
                    fn=dummy_iter,
                    iterable=[
                        [
                            make_test_task(fn=lambda: None, id_="a"),
                            make_test_task(fn=lambda: None, id_="b"),
                            make_test_task(fn=lambda: None, id_="c"),
                        ],
                        Dependency([make_test_task(fn=lambda: None, id_="a")]),
                        [
                            make_test_task(fn=lambda: None, id_="d"),
                        ],
                    ],
                    id_="flow_0",
                )
            ],
            10,
            [["a", "b", "c"], [], ["d"]],
            [["b"], ["a"], ["c", "d"]],
            [["flow_0"], ["flow_0"], []],
        ],
        [
            [
                make_test_flow(
                    fn=dummy_iter,
                    iterable=[
                        [
                            make_test_task(fn=lambda: None, id_="a"),
                            make_test_task(fn=lambda: None, id_="b"),
                            make_test_task(fn=lambda: None, id_="c"),
                        ],
                        Dependency([make_test_task(fn=lambda: None, id_="a")]),
                        [
                            make_test_task(fn=lambda: None, id_="d"),
                        ],
                    ],
                    id_="flow_0",
                )
            ],
            10,
            [["a", "b", "c"], [], ["d"]],
            [["b"], ["a"], ["c", "d"]],
            [["flow_0"], ["flow_0"], []],
        ],
        [
            [
                make_test_flow(
                    fn=dummy_iter,
                    iterable=[
                        make_test_task(fn=lambda: None, id_="a"),
                        Dependency(),
                        make_test_task(fn=lambda: None, id_="b"),
                    ],
                    id_="flow_0",
                )
            ],
            10,
            [["a"], ["b"], []],
            [["a"], [], ["b"]],
            [["flow_0"], ["flow_0"], []],
        ],
        [
            [
                make_test_flow(
                    fn=dummy_iter,
                    iterable=[
                        make_test_flow(
                            fn=dummy_iter,
                            iterable=[
                                make_test_task(fn=lambda: None, id_="x"),
                                make_test_task(fn=lambda: None, id_="y"),
                                Dependency([make_test_task(fn=lambda: None, id_="x")]),
                                make_test_task(fn=lambda: None, id_="z"),
                            ],
                            id_="flow_1",
                        ),
                        make_test_task(fn=lambda: None, id_="a"),
                        Dependency([make_test_task(fn=lambda: None, id_="a")]),
                        make_test_task(fn=lambda: None, id_="b"),
                    ],
                    id_="flow_0",
                )
            ],
            10,
            [["a"], ["x", "y"], ["b"], ["z"]],
            [[], ["y", "a"], ["x"], ["z"]],
            [["flow_0", "flow_1"], ["flow_0", "flow_1"], ["flow_0", "flow_1"], ["flow_0"]],
        ],
    ],
)
def test_flow_execution_flow(
    flows: list[Flow],
    max_batch_len: int,
    expected_batch_ids,
    completed_task_ids: list[list[str]],
    expected_ongoing_flow_ids,
):
    state = InMemoryExecutionState(ongoing_flows=flows)

    for i, exp_id in enumerate(expected_batch_ids):  # pylint: disable=consider-using-enumerate
        batch = state.get_task_batch(max_batch_len=max_batch_len)
        assert [e.id_ for e in batch] == exp_id

        task_outcomes = {id_: TaskOutcome[Any]() for id_ in completed_task_ids[i]}
        state.update_with_task_outcomes(task_outcomes)
        assert state.get_ongoing_flow_ids() == expected_ongoing_flow_ids[i]


def test_task_outcome_setting():
    # type: () -> None
    task = make_test_task(fn=lambda: None, id_="a")
    flows = [make_test_flow(fn=dummy_iter, iterable=[task], id_="flow_0")]
    state = InMemoryExecutionState(ongoing_flows=flows)
    state.get_task_batch()
    outcomes = {"a": TaskOutcome(return_value=5566)}
    state.update_with_task_outcomes(outcomes)
    assert task.outcome == outcomes["a"]


def test_task_outcome_fail_raise():
    # type: () -> None
    task = make_test_task(fn=lambda: None, id_="a")
    flows = [make_test_flow(fn=dummy_iter, iterable=[task], id_="flow_0")]
    state = InMemoryExecutionState(ongoing_flows=flows, raise_on_failed_task=True)
    state.get_task_batch()
    outcomes = {"a": TaskOutcome[Any](exception=Exception())}
    with pytest.raises(Exception):
        state.update_with_task_outcomes(outcomes)


def test_task_outcome_fail_noraise():
    # type: () -> None
    task = make_test_task(fn=lambda: None, id_="a")
    flows = [make_test_flow(fn=dummy_iter, iterable=[task], id_="flow_0")]
    state = InMemoryExecutionState(ongoing_flows=flows, raise_on_failed_task=False)
    state.get_task_batch()
    outcomes = {"a": TaskOutcome[Any](exception=Exception())}
    state.update_with_task_outcomes(outcomes)
    assert task.status == TaskStatus.FAILED
