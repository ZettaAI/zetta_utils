class MazepaException(Exception):
    ...


class MazepaStop(MazepaException):
    ...


class MazepaCancel(MazepaException):
    ...


class MazepaExecutionFailure(MazepaException):
    ...
