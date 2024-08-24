# pylint: disable=all # type: ignore

import traceback

from fastapi import Request, Response

from zetta_utils.log import get_logger

logger = get_logger("web_api")


def generic_exception_handler(request: Request, exc: Exception):
    logger.error(traceback.format_exc())
    if isinstance(exc, KeyError):
        if request.method == "GET":
            return Response(status_code=404, content=repr(exc))
    return Response(status_code=500, content=traceback.format_exc())
