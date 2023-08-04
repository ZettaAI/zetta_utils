from __future__ import annotations

from typing import Any

import pytest

from zetta_utils.mazepa import (
    Dependency,
    Flow,
    InMemoryExecutionState,
    TaskOutcome,
    TaskStatus,
    constants,
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
            [["a", "x", "y"], [], ["z"], ["b"]],
            [[], ["x"], ["z", "y", "a"], []],
            [["flow_0", "flow_1"], ["flow_0", "flow_1"], ["flow_0"], ["flow_0"]],
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


def test_task_outcome_fail_unrelated_task_id():
    # type: () -> None
    task = make_test_task(fn=lambda: None, id_="a")
    flows = [make_test_flow(fn=dummy_iter, iterable=[task], id_="flow_0")]
    state = InMemoryExecutionState(ongoing_flows=flows, raise_on_failed_task=True)
    state.get_task_batch()
    outcomes = {"i'm not from this flow": TaskOutcome[Any](exception=Exception())}
    state.update_with_task_outcomes(outcomes)


def test_task_outcome_fail_raise_unknown_task():
    # type: () -> None
    task = make_test_task(fn=lambda: None, id_="a")
    flows = [make_test_flow(fn=dummy_iter, iterable=[task], id_="flow_0")]
    state = InMemoryExecutionState(ongoing_flows=flows, raise_on_failed_task=True)
    state.get_task_batch()
    outcomes = {constants.UNKNOWN_TASK_ID: TaskOutcome[Any](exception=Exception())}
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


def test_progress_report():
    # type: () -> None
    task_a = make_test_task(fn=lambda: None, id_="a", operation_name="A")
    task_b = make_test_task(fn=lambda: None, id_="b", operation_name="B")
    flows = [make_test_flow(fn=dummy_iter, iterable=[task_a, task_b], id_="flow_0")]
    state = InMemoryExecutionState(ongoing_flows=flows, raise_on_failed_task=False)
    report1 = state.get_progress_reports()
    assert len(report1) == 0

    state.get_task_batch()
    report2 = state.get_progress_reports()
    assert len(report2) == 2
    assert report2["A"].submitted_count == 1
    assert report2["A"].completed_count == 0
    assert report2["B"].submitted_count == 1
    assert report2["B"].completed_count == 0

    outcomes = {"a": TaskOutcome[Any]()}
    state.update_with_task_outcomes(outcomes)

    report3 = state.get_progress_reports()
    assert len(report3) == 2
    assert report3["A"].submitted_count == 1
    assert report3["A"].completed_count == 1
    assert report3["B"].submitted_count == 1
    assert report3["B"].completed_count == 0
