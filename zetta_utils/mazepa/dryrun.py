import copy
from collections import defaultdict

from typeguard import typechecked

from zetta_utils import log

from . import Dependency, Flow

logger = log.get_logger("zetta_utils")


@typechecked
def get_expected_operation_counts(flows: list[Flow]) -> dict[str, int]:
    operation_task_ids = dryrun_for_task_ids(flows)
    return {k: len(v) for k, v in operation_task_ids.items()}


@typechecked
def dryrun_for_task_ids(flows: list[Flow]) -> dict[str, set[str]]:
    dryrun_flows = copy.deepcopy(flows)
    logger.info("Starting dryrun....")
    result = _dryrun_for_task_ids(dryrun_flows)
    logger.info("Dryrun finished.")
    return result


@typechecked
def _dryrun_for_task_ids(flows: list[Flow]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)

    for flow in flows:
        flow_yield = flow.get_next_batch()

        while flow_yield is not None:
            if not isinstance(flow_yield, Dependency):
                for e in flow_yield:
                    if isinstance(e, Flow):
                        this_flow_result = _dryrun_for_task_ids([e])
                        result = defaultdict(
                            set,
                            {
                                k: result[k].union(this_flow_result[k])
                                for k in list(result.keys()) + list(this_flow_result.keys())
                            },
                        )
                    else:
                        result[e.operation_name].add(e.id_)
            flow_yield = flow.get_next_batch()

    return result
