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
