from __future__ import annotations

from zetta_utils.mazepa import (
    Dependency,
    Flow,
    FlowFnReturnType,
    FlowSchema,
    flow_schema,
    flow_schema_cls,
)
from zetta_utils.mazepa.flows import _FlowSchema


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


def test_get_batch_tags(mocker):
    fn = mocker.MagicMock(
        return_value=iter(
            [
                _FlowSchema(
                    fn=lambda: None,
                )()
            ]
        )
    )
    tag_list = ["tag1", "tag2"]
    flow = _FlowSchema(
        fn=fn,
        tags=tag_list,
    )()

    result = flow.get_next_batch()
    assert isinstance(result, list)
    assert isinstance(result[0], Flow)
    assert result[0].tags == tag_list
