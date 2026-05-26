# pylint: disable=all # type: ignore
import logging
import os
import sys

_SA_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"

log = logging.getLogger(__name__)


def assert_no_serviceaccount_token() -> None:
    session_id = os.environ.get("SESSION_ID", "<unknown>")
    if not os.path.exists(_SA_TOKEN_PATH):
        log.info(
            "sessions.worker.boot_self_check_passed",
            extra={"sessionId": session_id},
        )
        return

    # Print to stderr FIRST. Lifespan fires before logging is fully wired
    # to the container stdout pipeline in some hypercorn configurations;
    # the print() here guarantees the FATAL line reaches the container log.
    print(
        f"sessions.worker.boot_self_check_failed "
        f"sessionId={session_id} "
        f"reason=serviceaccount_token_present "
        f"tokenPath={_SA_TOKEN_PATH}",
        file=sys.stderr,
        flush=True,
    )
    log.critical(
        "sessions.worker.boot_self_check_failed",
        extra={
            "sessionId": session_id,
            "reason": "serviceaccount_token_present",
            "tokenPath": _SA_TOKEN_PATH,
        },
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)  # pragma: no cover
