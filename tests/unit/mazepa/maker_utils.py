from zetta_utils.mazepa.flows import _FlowSchema
from zetta_utils.mazepa.id_generation import get_literal_id_fn
from zetta_utils.mazepa.tasks import _TaskableOperation


def make_test_task(fn, id_, worker_type=None, operation_name="DummyTask"):  # TODO: type me
    return _TaskableOperation(
        fn=fn, operation_name=operation_name, id_fn=get_literal_id_fn(id_), worker_type=worker_type
    ).make_task()


def make_test_flow(fn, id_, **kwargs):  # TODO: type me
    return _FlowSchema(fn=fn, id_fn=get_literal_id_fn(id_))(**kwargs)
