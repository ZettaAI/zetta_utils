# pylint: disable=no-self-use
from __future__ import annotations

from zetta_utils.mazepa import (
    Dependency,
    Flow,
    FlowSchema,
    flow_schema,
    flow_schema_cls,
)
from zetta_utils.mazepa.flows import _FlowSchema


def test_make_flow_schema_cls() -> None:
    @flow_schema_cls
    class DummyFlowCls:
        def flow(self):
            yield Dependency()

    obj = DummyFlowCls()
    assert isinstance(obj, FlowSchema)
    flow = obj()
    assert isinstance(flow, Flow)


def test_make_flow_schema():
    @flow_schema
    def dummy_flow_fn():
        yield []

    assert isinstance(dummy_flow_fn, FlowSchema)
    flow = dummy_flow_fn()
    assert isinstance(flow, Flow)


def test_get_batch_tags(mocker):
    def do_nothing():
        return

    fn = mocker.MagicMock(
        return_value=iter(
            [
                _FlowSchema(
                    fn=do_nothing,
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
