from __future__ import annotations
from zetta_utils.mazepa import (
    Flow,
    TaskExecutionEnv,
    FlowType,
    flow_type,
    flow_type_cls,
    FlowFnReturnType,
    Dependency,
)
from zetta_utils.mazepa.flows import _Flow, _FlowType


def test_make_flow_type_cls() -> None:
    @flow_type_cls
    class DummyFlowCls:
        x: str = "1"

        def generate(self) -> FlowFnReturnType:
            yield Dependency(self.x)

    obj = DummyFlowCls()
    # reveal_type(obj)
    assert isinstance(obj, FlowType)
    flow = obj()
    # reveal_type(flow)
    assert isinstance(flow, Flow)


def test_make_flow_type():
    @flow_type
    def dummy_flow_fn():
        yield []

    assert isinstance(dummy_flow_fn, FlowType)
    flow = dummy_flow_fn()
    assert isinstance(flow, Flow)


def test_ctx():
    j = _Flow(
        fn=lambda: None,
        task_execution_env=None,
        id_="flow_0",
    )
    env = TaskExecutionEnv()
    with j.task_execution_env_ctx(env):
        assert j.task_execution_env == env


def test_get_batch_env(mocker):
    fn = mocker.MagicMock(
        return_value=iter(
            [
                _FlowType(
                    fn=lambda: None,
                )()
            ]
        )
    )
    j = _FlowType(
        fn=fn,
    )()
    env = TaskExecutionEnv()
    with j.task_execution_env_ctx(env):
        result = j.get_next_batch()
        assert result[0].task_execution_env == env
