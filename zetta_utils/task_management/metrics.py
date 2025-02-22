# pylint: disable=all
def get_total_subtask_time(
    project_name: str, subtask_ids: str | list[str]
) -> int:  # pragma: no cover
    """
    Computes the total time spent (in seconds) for one or more subtasks within the project.

    :param project_name: The name of the project.
    :param subtask_ids: A single subtask ID or a list of subtask IDs.
    :return: The total time spent on the subtask(s) in seconds.
    :raises KeyError: If any of the subtasks do not exist.
    :raises RuntimeError: If the Firestore transaction fails.
    """
    return 0


def get_total_subtask_cost(
    project_name: str, subtask_ids: str | list[str]
) -> float:  # pragma: no cover
    """
    Calculates the total cost for one or more subtasks within the project.

    :param project_name: The name of the project.
    :param subtask_ids: A single subtask ID or a list of subtask IDs.
    :return: The total cost for the subtask(s).
    :raises KeyError: If any of the subtasks do not exist.
    :raises RuntimeError: If the Firestore transaction fails.
    """
    return 0
