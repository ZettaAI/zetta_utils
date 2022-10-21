from zetta_utils.mazepa.tasks import _TaskFactory
from zetta_utils.mazepa.flows import _FlowType
from zetta_utils.mazepa.id_generators import get_literal_id_fn


def make_test_task(fn, id_, task_execution_env=None):  # TODO: type me
    return _TaskFactory(
        fn=fn,
        id_fn=get_literal_id_fn(id_),
        task_execution_env=task_execution_env,
    ).make_task()


def make_test_flow(fn, id_, **kwargs):  # TODO: type me
    return _FlowType(fn=fn, id_fn=get_literal_id_fn(id_))(**kwargs)
