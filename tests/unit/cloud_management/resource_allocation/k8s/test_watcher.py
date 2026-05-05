# pylint: disable=redefined-outer-name,unused-argument,protected-access
"""Tests for the watcher infrastructure: resilient_watch + BatchedWarner."""
from __future__ import annotations

import threading

import pytest
from kubernetes.client.exceptions import ApiException

from zetta_utils.cloud_management.resource_allocation.k8s import watcher


class TestBatchedWarner:
    def test_flush_no_pending_emits_nothing(self, mocker):
        log_warning = mocker.patch.object(watcher.logger, "warning")
        bw = watcher.BatchedWarner("t", log_path=None)
        bw.flush()
        log_warning.assert_not_called()

    def test_flush_emits_one_sample_per_bracket_category(self, mocker):
        log_warning = mocker.patch.object(watcher.logger, "warning")
        bw = watcher.BatchedWarner("t", log_path=None)
        bw.pending = [
            "[A] msg1",
            "[A] msg2",
            "[B] msg3",
            "[A] msg4",
        ]
        bw.flush()
        log_warning.assert_called_once()
        msg = log_warning.call_args.args[0]
        assert "total 4 events" in msg
        assert "+2 more" in msg
        assert "[A] msg1" in msg
        assert "[B] msg3" in msg
        # Categories already shown should not appear again.
        assert "[A] msg2" not in msg
        assert "[A] msg4" not in msg

    def test_flush_omits_more_suffix_when_all_unique(self, mocker):
        log_warning = mocker.patch.object(watcher.logger, "warning")
        bw = watcher.BatchedWarner("t", log_path=None)
        bw.pending = ["[A] x", "[B] y"]
        bw.flush()
        msg = log_warning.call_args.args[0]
        assert "more" not in msg

    def test_add_does_not_flush_inside_interval(self, mocker):
        log_warning = mocker.patch.object(watcher.logger, "warning")
        mocker.patch.object(watcher.BatchedWarner, "FLUSH_INTERVAL_SEC", 9999)
        bw = watcher.BatchedWarner("t", log_path=None)
        bw.add("[X] hi")
        log_warning.assert_not_called()
        assert bw.pending == ["[X] hi"]

    def test_add_flushes_when_interval_elapsed(self, mocker):
        log_warning = mocker.patch.object(watcher.logger, "warning")
        mocker.patch.object(watcher.BatchedWarner, "FLUSH_INTERVAL_SEC", 0)
        bw = watcher.BatchedWarner("t", log_path=None)
        bw.add("[X] hi")
        log_warning.assert_called_once()
        assert not bw.pending


@pytest.fixture
def mock_watch_class(mocker):
    """Patch ``watcher.watch`` so each test controls ``Watch().stream``."""
    mock_module = mocker.patch.object(watcher, "watch")
    mock_w = mocker.MagicMock()
    mock_module.Watch.return_value = mock_w
    return mock_w


class TestResilientWatch:
    def test_rebuilds_via_factory_and_calls_on_error_after_exception(
        self, mocker, mock_watch_class
    ):
        """First stream attempt raises 401; second succeeds. Factory called
        twice; on_error fires once; on_event runs once with the recovered event."""
        stop_event = threading.Event()
        sentinel_obj = object()

        def on_event(obj):
            assert obj is sentinel_obj
            stop_event.set()

        on_error = mocker.MagicMock()
        list_fn = mocker.MagicMock()
        factory = mocker.MagicMock(return_value=list_fn)
        mocker.patch.object(stop_event, "wait", return_value=False)

        call_count = {"n": 0}

        def stream_side_effect(*_args, **_kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ApiException(status=401, reason="Unauthorized")
            yield {"object": sentinel_obj}

        mock_watch_class.stream.side_effect = stream_side_effect

        watcher.resilient_watch(
            factory,
            on_event,
            namespace="default",
            description="test watcher",
            stop_event=stop_event,
            on_error=on_error,
        )

        assert factory.call_count == 2
        on_error.assert_called_once()

    def test_does_not_emit_per_retry_warning_during_failure_streak(self, mocker, mock_watch_class):
        """5 consecutive errors with a high flush interval emit zero warnings."""
        stop_event = threading.Event()
        log_warning = mocker.patch.object(watcher.logger, "warning")
        mocker.patch.object(watcher.BatchedWarner, "FLUSH_INTERVAL_SEC", 9999)

        attempts = {"n": 0}

        def stream_side_effect(*_args, **_kwargs):
            class _RaisingIter:
                def __iter__(self):
                    return self

                def __next__(self):
                    attempts["n"] += 1
                    if attempts["n"] >= 5:
                        stop_event.set()
                    raise ApiException(status=401, reason="Unauthorized")

            return _RaisingIter()

        mock_watch_class.stream.side_effect = stream_side_effect
        mocker.patch.object(stop_event, "wait", return_value=False)

        watcher.resilient_watch(
            mocker.MagicMock(return_value=mocker.MagicMock()),
            mocker.MagicMock(),
            namespace="default",
            description="test watcher",
            stop_event=stop_event,
            on_error=mocker.MagicMock(),
        )

        log_warning.assert_not_called()
        assert attempts["n"] >= 5

    def test_flushes_error_batch_on_recovery(self, mocker, mock_watch_class):
        """Errors then a successful stream → exactly one summary warning."""
        stop_event = threading.Event()
        log_warning = mocker.patch.object(watcher.logger, "warning")
        mocker.patch.object(watcher.BatchedWarner, "FLUSH_INTERVAL_SEC", 9999)

        attempts = {"n": 0}

        def stream_side_effect(*_args, **_kwargs):
            attempts["n"] += 1
            if attempts["n"] <= 3:
                raise ApiException(status=401, reason="Unauthorized")
            stop_event.set()
            yield {"object": object()}

        mock_watch_class.stream.side_effect = stream_side_effect
        mocker.patch.object(stop_event, "wait", return_value=False)

        watcher.resilient_watch(
            mocker.MagicMock(return_value=mocker.MagicMock()),
            mocker.MagicMock(),
            namespace="default",
            description="test watcher",
            stop_event=stop_event,
            on_error=mocker.MagicMock(),
        )

        log_warning.assert_called_once()
        msg = log_warning.call_args.args[0]
        assert "test watcher errors" in msg
        assert "total 3 events" in msg
        assert "[ApiException:401]" in msg

    def test_stop_event_breaks_loop_without_starting_stream(self, mocker, mock_watch_class):
        stop_event = threading.Event()
        stop_event.set()
        factory = mocker.MagicMock()

        watcher.resilient_watch(
            factory,
            mocker.MagicMock(),
            namespace="default",
            description="test watcher",
            stop_event=stop_event,
        )

        factory.assert_not_called()
        mock_watch_class.stream.assert_not_called()
