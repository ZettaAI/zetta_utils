import os.path


def abspath(path: str) -> str:  # pragma: no cover
    """
    Changes relative paths to absolute paths, and does so while respecting prefixes;
    adds 'file://' protocol if no protocol is specified.
    """
    split = path.split("://")
    prefixes = split[:-1]
    path_no_prefix = split[-1]
    if len(prefixes) == 0:
        prefixes = ["file"]
    if prefixes == ["file"]:
        path_no_prefix = os.path.abspath(os.path.expanduser(path_no_prefix))
    return "://".join(prefixes + [path_no_prefix])
