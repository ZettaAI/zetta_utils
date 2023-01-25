import contextlib
import os
import sys
from typing import Callable


@contextlib.contextmanager
def breakpointhook_ctx_mngr(
    new_hook_maker: Callable[[Callable], Callable[[], None]]
):  # pragma: no cover # Any good ideas on how to test this?
    old_hook = sys.breakpointhook
    sys.breakpointhook = new_hook_maker(old_hook)
    yield
    sys.breakpointhook = old_hook


@contextlib.contextmanager
def set_env_ctx_mngr(**environ):
    """
    Temporarily set the process environment variables.

    >>> with set_env(PLUGINS_DIR=u'test/plugins'):
    ...   "PLUGINS_DIR" in os.environ
    True

    >>> "PLUGINS_DIR" in os.environ
    False

    :type environ: dict[str, unicode]
    :param environ: Environment variables to set
    """

    old_environ = dict(os.environ)
    os.environ.update(environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)
