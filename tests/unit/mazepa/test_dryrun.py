# type: ignore # We're breaking mypy here
from __future__ import annotations

import pytest

from zetta_utils.mazepa import Dependency, Flow, dryrun

from .maker_utils import make_test_flow, make_test_task


def dummy_iter(iterable):
    return iter(iterable)


@pytest.mark.parametrize(
    "flows, expected",
    [
        [
            [
                make_test_flow(
                    fn=dummy_iter,
                    id_="flow_0",
                    iterable=[
                        make_test_task(fn=lambda: None, id_=id_, operation_name="OperationA")
                        for id_ in ["a", "b", "c"]
                    ],
                )
            ],
            {"OperationA": 3},
        ],
        [
            [
                make_test_flow(
                    fn=dummy_iter,
                    id_="flow_0",
                    iterable=[
                        make_test_flow(
                            fn=dummy_iter,
                            id_="flow_1",
                            iterable=[
                                make_test_task(
                                    fn=lambda: None, id_=id_, operation_name="OperationB"
                                )
                                for id_ in ["aa", "bb", "cc"]
                            ]
                            + [Dependency()]
                            + [
                                make_test_task(
                                    fn=lambda: None, id_=id_, operation_name="OperationB"
                                )
                                for id_ in ["aaa", "bbb", "ccc"]
                            ],
                        ),
                    ]
                    + [
                        make_test_task(fn=lambda: None, id_=id_, operation_name="OperationA")
                        for id_ in ["a", "b", "c"]
                    ],
                )
            ],
            {"OperationA": 3, "OperationB": 6},
        ],
    ],
)
def test_dryrun_execution_counts(
    flows: list[Flow],
    expected: dict[str, int],
):
    result = dryrun.get_expected_operation_counts(flows)
    assert result == expected


def test_in_dryrun_flag_observed_during_dryrun(mocker):
    observed: list[bool] = []

    original = dryrun._dryrun_for_task_ids  # pylint: disable=protected-access

    def fake(flows):
        observed.append(dryrun.in_dryrun())
        return original(flows)

    mocker.patch(
        "zetta_utils.mazepa.dryrun._dryrun_for_task_ids",
        side_effect=fake,
    )

    assert dryrun.in_dryrun() is False
    dryrun.dryrun_for_task_ids([])
    assert observed == [True]
    assert dryrun.in_dryrun() is False


def test_in_dryrun_flag_restored_on_nested_exit():
    # pylint: disable=protected-access
    inside_outer: list[bool] = []
    inside_inner: list[bool] = []

    with dryrun._dryrun_flag():
        inside_outer.append(dryrun.in_dryrun())
        with dryrun._dryrun_flag():
            inside_inner.append(dryrun.in_dryrun())
        inside_outer.append(dryrun.in_dryrun())

    assert inside_outer == [True, True]
    assert inside_inner == [True]
    assert dryrun.in_dryrun() is False
