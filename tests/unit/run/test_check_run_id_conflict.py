import pytest

from zetta_utils.run import RunInfo, RunState, _check_run_id_conflict


class _FakeRunDB:
    def __init__(self):
        self._rows: dict[str, dict] = {}

    def __contains__(self, key):
        return key in self._rows

    def __getitem__(self, key):
        run_id, _cols = key
        return self._rows.get(run_id, {})

    def __setitem__(self, key, value):
        run_id, _cols = key
        self._rows.setdefault(run_id, {}).update(value)


@pytest.fixture
def fake_run_db(mocker):
    fake = _FakeRunDB()
    mocker.patch("zetta_utils.run.RUN_DB", fake)
    return fake


def test_no_existing_row(fake_run_db):
    _check_run_id_conflict("fresh")


def test_existing_row_raises_without_allowed(fake_run_db):
    fake_run_db[("running-id", (RunInfo.STATE.value,))] = {
        RunInfo.STATE.value: RunState.RUNNING.value
    }
    with pytest.raises(ValueError, match="already exists"):
        _check_run_id_conflict("running-id")


def test_existing_queued_with_allowed_does_not_raise(fake_run_db):
    fake_run_db[("queued-id", (RunInfo.STATE.value,))] = {
        RunInfo.STATE.value: RunState.QUEUED.value
    }
    _check_run_id_conflict("queued-id", allowed_prior_state="queued")


def test_mismatched_prior_state_raises(fake_run_db):
    fake_run_db[("mismatch-id", (RunInfo.STATE.value,))] = {
        RunInfo.STATE.value: RunState.RUNNING.value
    }
    with pytest.raises(ValueError, match="state="):
        _check_run_id_conflict("mismatch-id", allowed_prior_state="queued")
