# pylint: disable=all
def get_total_task_time(
    project_name: str, task_ids: str | list[str]
) -> int:  # pragma: no cover
    """
    Computes the total time spent (in seconds) for one or more tasks within the project.

    :param project_name: The name of the project.
    :param task_ids: A single task ID or a list of task IDs.
    :return: The total time spent on the task(s) in seconds.
    :raises KeyError: If any of the tasks do not exist.
    :raises RuntimeError: If the database transaction fails.
    """
    return 0


def get_total_task_cost(
    project_name: str, task_ids: str | list[str]
) -> float:  # pragma: no cover
    """
    Calculates the total cost for one or more tasks within the project.

    :param project_name: The name of the project.
    :param task_ids: A single task ID or a list of task IDs.
    :return: The total cost for the task(s).
    :raises KeyError: If any of the tasks do not exist.
    :raises RuntimeError: If the database transaction fails.
    """
    return 0
