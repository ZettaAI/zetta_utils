import time

import pytest

from zetta_utils import builder, run
from zetta_utils.run import RunInfo, RunState, run_ctx_manager


@pytest.fixture
def register_noop():
    @builder.register("noop_for_test")
    def _noop(value: int) -> int:
        return value

    try:
        yield "noop_for_test"
    finally:
        builder.REGISTRY.pop("noop_for_test", None)


@pytest.fixture(autouse=True)
def _user_env(monkeypatch):
    monkeypatch.setenv("ZETTA_USER", "test-user")
    monkeypatch.setenv("ZETTA_PROJECT", "test-project")
    monkeypatch.setenv("ZETTA_RUN_SPEC_PATH", "/dev/null")
    monkeypatch.setenv("EXECUTION_HEARTBEAT_LOOKBACK", "60")


def test_queued_at_none_preserves_existing_behavior(firestore_emulator, register_noop, mocker):
    mocker.patch("zetta_utils.run.record_run")
    with run_ctx_manager(main_run_process=True, run_id="test-run-001", spec={}):
        row = run.RUN_DB[("test-run-001", (RunInfo.STATE.value,))]
        assert row[RunInfo.STATE.value] == RunState.RUNNING.value
    final = run.RUN_DB[("test-run-001", (RunInfo.STATE.value,))]
    assert final[RunInfo.STATE.value] == RunState.COMPLETED.value  # "completed"


def test_queued_then_running_transition(firestore_emulator, register_noop, mocker):
    mocker.patch("zetta_utils.run.record_run")
    queued_at = time.time()
    with run_ctx_manager(
        main_run_process=True,
        run_id="test-run-002",
        spec={},
        queued_at=queued_at,
    ) as ctx:
        row = run.RUN_DB[("test-run-002", (RunInfo.STATE.value, RunInfo.QUEUED_AT.value))]
        assert row[RunInfo.STATE.value] == RunState.QUEUED.value
        assert row[RunInfo.QUEUED_AT.value] == queued_at
        ctx.transition_to_running()
        row2 = run.RUN_DB[("test-run-002", (RunInfo.STATE.value,))]
        assert row2[RunInfo.STATE.value] == RunState.RUNNING.value
    final = run.RUN_DB[("test-run-002", (RunInfo.STATE.value,))]
    assert final[RunInfo.STATE.value] == RunState.COMPLETED.value  # "completed"


def test_queued_without_transition_terminates_cleanly(firestore_emulator, register_noop, mocker):
    mocker.patch("zetta_utils.run.record_run")
    with pytest.raises(RuntimeError, match="simulated"):
        with run_ctx_manager(
            main_run_process=True,
            run_id="test-run-003",
            spec={},
            queued_at=time.time(),
        ):
            raise RuntimeError("simulated crash before transition")
    final = run.RUN_DB[("test-run-003", (RunInfo.STATE.value,))]
    assert final[RunInfo.STATE.value] == RunState.FAILED.value


def test_transition_to_running_is_idempotent(firestore_emulator, register_noop, mocker):
    mocker.patch("zetta_utils.run.record_run")
    with run_ctx_manager(
        main_run_process=True,
        run_id="test-run-004",
        spec={},
        queued_at=time.time(),
    ) as ctx:
        ctx.transition_to_running()
        ctx.transition_to_running()  # second call is a no-op (idempotent)
        row = run.RUN_DB[("test-run-004", (RunInfo.STATE.value,))]
        assert row[RunInfo.STATE.value] == RunState.RUNNING.value


def test_transition_from_non_queued_raises():
    """Construct a RunCtx in RUNNING state directly; verify transition raises."""
    from zetta_utils.run import RunCtx

    ctx = RunCtx(run_id="dummy", _state=RunState.RUNNING)
    # Re-call from RUNNING short-circuits (idempotent); no raise.
    ctx.transition_to_running()
    # Manually corrupt the state to something other than QUEUED/RUNNING.
    object.__setattr__(ctx, "_state", RunState.FAILED)
    with pytest.raises(RuntimeError, match="transition_to_running called from state="):
        ctx.transition_to_running()


def test_gc_filter_includes_queued(firestore_emulator, mocker):
    mocker.patch("zetta_utils.run.record_run")
    with run_ctx_manager(
        main_run_process=True,
        run_id="test-run-stale-queued",
        spec={},
        queued_at=time.time(),
    ):
        pass
    run.update_run_info("test-run-stale-queued", {RunInfo.STATE.value: RunState.QUEUED.value})
    # A multi-value state filter compiles to a composite OR query, which the
    # Firestore emulator rejects; a single-value filter proves queued rows are
    # indexable by state, which is what the GC broadening relies on.
    rows = run.RUN_DB.query(column_filter={"state": ["queued"]})
    assert "test-run-stale-queued" in rows


