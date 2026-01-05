# pylint: disable=no-self-use
from __future__ import annotations

from zetta_utils.mazepa import (
    Dependency,
    Flow,
    FlowSchema,
    flow_schema,
    flow_schema_cls,
)
from zetta_utils.mazepa.flows import _FlowSchema, sequential_flow


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


def test_get_batch(mocker):
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
    flow = _FlowSchema(
        fn=fn,
    )()

    result = flow.get_next_batch()
    assert isinstance(result, list)
    assert isinstance(result[0], Flow)


def test_sequential_flow_with_callable_functions(mocker):
    # Mock function to track calls
    mock_fn = mocker.MagicMock()

    # Create sequential flow with a callable function
    flow = sequential_flow([mock_fn])

    # Get first batch - should be dependency (after calling function)
    batch = flow.get_next_batch()
    assert isinstance(batch, Dependency)
    mock_fn.assert_called_once()

    # Get second batch - should be None (end of flow)
    batch = flow.get_next_batch()
    assert batch is None


def test_flow_schema_with_non_generator_function():
    call_count = 0

    def regular_function():
        nonlocal call_count
        call_count += 1
        return "test_result"

    # Decorate regular function with flow_schema
    flow_fn = flow_schema(regular_function)
    assert isinstance(flow_fn, FlowSchema)

    # Create flow and get batches
    flow = flow_fn()
    assert isinstance(flow, Flow)

    # First batch should yield the function result
    batch = flow.get_next_batch()
    assert batch is None
    assert call_count == 1

    # Second batch should be None (end of generator)
    batch = flow.get_next_batch()
    assert batch is None


def test_flow_schema_with_non_generator_function_returning_none():
    call_count = 0

    def void_function():
        nonlocal call_count
        call_count += 1

    # Decorate function that returns None
    flow_fn = flow_schema(void_function)
    flow = flow_fn()

    # Should end immediately since function returns None
    batch = flow.get_next_batch()
    assert batch is None
    assert call_count == 1
