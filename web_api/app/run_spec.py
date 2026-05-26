# pylint: disable=all # type: ignore
import asyncio
import json
import logging
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import fsspec
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from starlette.background import BackgroundTask

from zetta_utils import builder, parsing, run, setup_environment
from zetta_utils.common import ctx_managers
from zetta_utils.run import run_ctx_manager

api = FastAPI(redirect_slashes=False)
_run_spec_semaphore = asyncio.Semaphore(1)
_loaded_preload: str = os.environ.get("INITIAL_PRELOAD", "try")
_dispatch_count: int = 0
_PRELOAD_RANK = {"none": 0, "try": 1, "inference": 2, "training": 3, "all": 4}
_MAX_DISPATCHES = int(os.environ.get("MAX_DISPATCHES_PER_POD", "50"))
_RUN_SPEC_DOWNLOAD_TIMEOUT_SEC = int(os.environ.get("RUN_SPEC_DOWNLOAD_TIMEOUT_SEC", "30"))

log = logging.getLogger(__name__)


class RunSpecBody(BaseModel):
    specUrl: str
    runId: str
    jobType: str
    requiredPreload: Literal["none", "try", "inference", "training", "all"] = "try"


class RunSpecResponse(BaseModel):
    result: object
    dispatchCount: int
    nearRecycle: bool


@api.post("/")
async def run_spec(body: RunSpecBody) -> Response:
    global _loaded_preload, _dispatch_count

    session_id = os.environ.get("SESSION_ID", "<unknown>")
    log.info(
        "sessions.worker.dispatch_received",
        extra={
            "sessionId": session_id,
            "runId": body.runId,
            "jobType": body.jobType,
            "requiredPreload": body.requiredPreload,
        },
    )
    log.info(
        "sessions.dispatch.semaphore_wait_started",
        extra={"sessionId": session_id, "runId": body.runId},
    )

    queued_at = datetime.now(timezone.utc).timestamp()
    with run_ctx_manager(
        main_run_process=True,
        run_id=body.runId,
        spec={},
        queued_at=queued_at,
    ) as ctx:
        _log_dispatch_state(
            "queued", session_id=session_id, run_id=body.runId, job_type=body.jobType
        )

        async with _run_spec_semaphore:
            ctx.transition_to_running()
            _log_dispatch_state(
                "running", session_id=session_id, run_id=body.runId, job_type=body.jobType
            )
            try:
                _upgrade_preload_if_needed(body.requiredPreload, session_id)
                spec = await _download_and_parse_spec(body.specUrl)
                with ctx_managers.set_env_ctx_mngr(
                    ZETTA_RUN_SPEC_PATH=body.specUrl,
                    CURRENT_BUILD_SPEC=json.dumps(spec),
                ):
                    run.record_run(spec)
                    result = builder.build(spec)
                _dispatch_count += 1
                near_recycle = _dispatch_count >= _MAX_DISPATCHES - 1
                _log_dispatch_state(
                    "completed", session_id=session_id, run_id=body.runId, job_type=body.jobType
                )
                log.info(
                    "sessions.worker.dispatch_completed",
                    extra={
                        "sessionId": session_id,
                        "runId": body.runId,
                        "outcome": "completed",
                        "dispatchCount": _dispatch_count,
                        "nearRecycle": near_recycle,
                    },
                )
                _light_cleanup()

                response_body = RunSpecResponse(
                    result=result,
                    dispatchCount=_dispatch_count,
                    nearRecycle=near_recycle,
                ).model_dump_json()

                # Recycle is success-path-only AND must fire AFTER the
                # response body is flushed to the socket. BackgroundTask
                # runs post-response-write; os._exit() in `finally:`
                # would kill hypercorn before the bytes leave the kernel.
                background: BackgroundTask | None = None
                if _dispatch_count >= _MAX_DISPATCHES:
                    background = BackgroundTask(_recycle_after_response, session_id)

                return Response(
                    content=response_body,
                    media_type="application/json",
                    background=background,
                )
            except Exception:
                _log_dispatch_state(
                    "failed", session_id=session_id, run_id=body.runId, job_type=body.jobType
                )
                log.exception(
                    "sessions.worker.dispatch_completed",
                    extra={
                        "sessionId": session_id,
                        "runId": body.runId,
                        "outcome": "failed",
                        "dispatchCount": _dispatch_count,
                    },
                )
                _light_cleanup()
                # Failed dispatches do not count toward the per-pod cap. The
                # cap drives a graceful recycle that is structurally
                # success-only: the recycle os._exit fires from a success-path
                # BackgroundTask and nearRecycle is delivered only in a
                # success-path response body. Counting failures here would let
                # the threshold cross without the master ever being told.
                raise HTTPException(status_code=500, detail="dispatch failed")


def _log_dispatch_state(state: str, *, session_id: str, run_id: str, job_type: str) -> None:
    log.info(
        "sessions.dispatch.total",
        extra={
            "sessionId": session_id,
            "runId": run_id,
            "state": state,
            "jobType": job_type,
        },
    )


def _recycle_after_response(session_id: str) -> None:
    log.info(
        "sessions.worker.exit_at_max_dispatches",
        extra={"sessionId": session_id, "dispatchCount": _dispatch_count},
    )
    os._exit(0)  # pragma: no cover


def _upgrade_preload_if_needed(required: str, session_id: str) -> None:
    """Best-effort preload upgrade. On failure, os._exit(1) so K8s recycles.

    Rationale: ``setup_environment(load_mode=...)`` performs module imports
    and model downloads. A partial failure mid-import leaves ``sys.modules``
    in an inconsistent state; ``_loaded_preload`` would NOT be updated, and
    the next dispatch would retry the upgrade against a polluted process.
    Crash-loop is the safe option — K8s recreates the pod with a clean
    Python interpreter.
    """
    global _loaded_preload
    if _PRELOAD_RANK[required] <= _PRELOAD_RANK[_loaded_preload]:
        return
    log.info(
        "sessions.worker.preload_upgraded",
        extra={"sessionId": session_id, "from": _loaded_preload, "to": required},
    )
    try:
        setup_environment(load_mode=required)
    except Exception:
        log.exception(
            "sessions.worker.preload_upgrade_failed",
            extra={"sessionId": session_id, "from": _loaded_preload, "to": required},
        )
        os._exit(1)  # pragma: no cover
    _loaded_preload = required


async def _download_and_parse_spec(spec_url: str) -> dict:
    """Download CUE from GCS/HTTP and parse. Bounded by RUN_SPEC_DOWNLOAD_TIMEOUT_SEC.

    fsspec.open(...) is sync; wrap in ``asyncio.to_thread`` then bound with
    ``asyncio.wait_for``. On timeout, raises ``asyncio.TimeoutError`` which
    propagates as 500 via the outer handler.
    """

    def _blocking_download_and_parse() -> dict:
        with tempfile.NamedTemporaryFile("w", suffix=".cue", delete=False) as tmp_f:
            with fsspec.open(spec_url, "r", encoding="utf8") as f:
                tmp_f.write(f.read())
            tmp_path = tmp_f.name
        try:
            return parsing.cue.load(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    return await asyncio.wait_for(
        asyncio.to_thread(_blocking_download_and_parse),
        timeout=_RUN_SPEC_DOWNLOAD_TIMEOUT_SEC,
    )


def _light_cleanup() -> None:
    builder.building.BUILT_OBJECT_ID_REGISTRY.clear()
    builder.PARALLEL_BUILD_ALLOWED = False
    run.RUN_ID = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    random.seed()
    np.random.seed()
    torch.seed()
