from __future__ import annotations

from zetta_utils.mazepa import (
    Dependency,
    Flow,
    FlowFnReturnType,
    FlowSchema,
    TaskExecutionEnv,
    flow_schema,
    flow_schema_cls,
)
from zetta_utils.mazepa.flows import Flow, _FlowSchema


def test_make_flow_schema_cls() -> None:
    @flow_schema_cls
    class DummyFlowCls:
        x: str = "1"

        def flow(self) -> FlowFnReturnType:
            yield Dependency(self.x)

    obj = DummyFlowCls()
    # reveal_type(obj)
    assert isinstance(obj, FlowSchema)
    flow = obj()
    # reveal_type(flow)
    assert isinstance(flow, Flow)


def test_make_flow_schema():
    @flow_schema
    def dummy_flow_fn():
        yield []

    assert isinstance(dummy_flow_fn, FlowSchema)
    flow = dummy_flow_fn()
    assert isinstance(flow, Flow)


def test_ctx():
    j = Flow(
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
                _FlowSchema(
                    fn=lambda: None,
                )()
            ]
        )
    )
    j = _FlowSchema(
        fn=fn,
    )()
    env = TaskExecutionEnv()
    with j.task_execution_env_ctx(env):
        result = j.get_next_batch()
        assert isinstance(result, list)
        assert isinstance(result[0], Flow)
        assert result[0].task_execution_env == env
