import signal
from typing import Optional

from rich.prompt import Confirm, Prompt

from zetta_utils import log

logger = log.get_logger("zetta_utils")


class InputTimedOut(Exception):
    pass


def timeout_handler(signum, frame):  # pragma: no cover
    raise InputTimedOut


def get_user_input(prompt: str, timeout: int = 0) -> Optional[str]:  # pragma: no cover
    signal.signal(signal.SIGALRM, timeout_handler)
    result: Optional[str] = None
    try:
        signal.alarm(timeout)
        result = Prompt.ask(prompt)
        signal.alarm(0)
    except InputTimedOut:
        pass

    return result


def get_user_confirmation(prompt: str, timeout: int = 0) -> Optional[bool]:  # pragma: no cover
    signal.signal(signal.SIGALRM, timeout_handler)
    result: Optional[bool] = None
    try:
        signal.alarm(timeout)
        result = Confirm.ask(prompt)
        signal.alarm(0)
    except InputTimedOut:
        pass

    return result
