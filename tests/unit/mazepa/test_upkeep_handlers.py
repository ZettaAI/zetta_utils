# pylint: disable=redefined-outer-name
import multiprocessing
import threading
import time
from typing import Any

from zetta_utils.common.partial import ComparablePartial
from zetta_utils.mazepa.upkeep_handlers import (
    SQSUpkeepHandlerManager,
    UpkeepCommand,
    extract_sqs_metadata,
    perform_direct_upkeep,
    run_sqs_upkeep_handler,
)


class TestPerformDirectUpkeep:
    def test_calls_extend_lease_fn(self, mocker):
        mock_extend_lease = mocker.MagicMock()
        task_start_time = time.time()

        perform_direct_upkeep(
            extend_lease_fn=mock_extend_lease,
            extend_duration=30,
            task_start_time=task_start_time,
        )

        mock_extend_lease.assert_called_once_with(30)


class TestExtractSqsMetadata:
    def test_non_comparable_partial_returns_none(self):
        def regular_fn():
            pass

        result = extract_sqs_metadata(regular_fn)
        assert result is None

    def test_comparable_partial_without_msg_returns_none(self):
        def some_fn():
            pass

        partial = ComparablePartial(some_fn, other_kwarg="value")
        result = extract_sqs_metadata(partial)
        assert result is None

    def test_msg_missing_required_attributes_returns_none(self):
        def some_fn():
            pass

        class IncompleteMsg:
            receipt_handle = "handle123"
            # Missing queue_name and region_name

        partial = ComparablePartial(some_fn, msg=IncompleteMsg())
        result = extract_sqs_metadata(partial)
        assert result is None

    def test_valid_sqs_metadata_extracted(self):
        def some_fn():
            pass

        class SQSMsg:
            receipt_handle = "handle123"
            queue_name = "test-queue"
            region_name = "us-east-1"
            endpoint_url = "http://localhost:9324"

        partial = ComparablePartial(some_fn, msg=SQSMsg())
        result = extract_sqs_metadata(partial)

        assert result == {
            "receipt_handle": "handle123",
            "queue_name": "test-queue",
            "region_name": "us-east-1",
            "endpoint_url": "http://localhost:9324",
        }

    def test_valid_sqs_metadata_with_none_endpoint_url(self):
        def some_fn():
            pass

        class SQSMsg:
            receipt_handle = "handle456"
            queue_name = "prod-queue"
            region_name = "us-west-2"
            # No endpoint_url attribute

        partial = ComparablePartial(some_fn, msg=SQSMsg())
        result = extract_sqs_metadata(partial)

        assert result == {
            "receipt_handle": "handle456",
            "queue_name": "prod-queue",
            "region_name": "us-west-2",
            "endpoint_url": None,
        }


class TestUpkeepCommand:
    def test_dataclass_fields(self):
        cmd = UpkeepCommand(
            action="start_upkeep",
            task_id="task-123",
            receipt_handle="handle-456",
            visibility_timeout=60,
            interval_sec=10.0,
            queue_name="my-queue",
            region_name="us-east-1",
            endpoint_url="http://localhost:9324",
        )

        assert cmd.action == "start_upkeep"
        assert cmd.task_id == "task-123"
        assert cmd.receipt_handle == "handle-456"
        assert cmd.visibility_timeout == 60
        assert cmd.interval_sec == 10.0
        assert cmd.queue_name == "my-queue"
        assert cmd.region_name == "us-east-1"
        assert cmd.endpoint_url == "http://localhost:9324"

    def test_dataclass_defaults(self):
        cmd = UpkeepCommand(action="shutdown")

        assert cmd.action == "shutdown"
        assert cmd.task_id is None
        assert cmd.receipt_handle is None
        assert cmd.visibility_timeout is None
        assert cmd.interval_sec is None
        assert cmd.queue_name is None
        assert cmd.region_name is None
        assert cmd.endpoint_url is None


class TestSQSUpkeepHandlerManager:
    def test_start_creates_process(self, mocker):
        mocker.patch("zetta_utils.mazepa.upkeep_handlers.multiprocessing.Queue")
        mock_process_cls = mocker.patch(
            "zetta_utils.mazepa.upkeep_handlers.multiprocessing.Process"
        )
        mock_process = mocker.MagicMock()
        mock_process_cls.return_value = mock_process

        manager = SQSUpkeepHandlerManager()
        manager.start()

        mock_process_cls.assert_called_once()
        mock_process.start.assert_called_once()

    def test_start_when_already_running_is_noop(self, mocker):
        mock_queue = mocker.MagicMock()
        mocker.patch(
            "zetta_utils.mazepa.upkeep_handlers.multiprocessing.Queue", return_value=mock_queue
        )
        mock_process_cls = mocker.patch(
            "zetta_utils.mazepa.upkeep_handlers.multiprocessing.Process"
        )
        mock_process = mocker.MagicMock()
        mock_process_cls.return_value = mock_process

        manager = SQSUpkeepHandlerManager()
        manager.start()
        manager.start()  # Second call should be no-op

        assert mock_process_cls.call_count == 1

    def test_shutdown_when_not_running_is_noop(self):
        manager = SQSUpkeepHandlerManager()
        manager.shutdown()  # Should not raise

    def test_shutdown_sends_command_and_joins(self, mocker):
        mock_queue = mocker.MagicMock()
        mocker.patch(
            "zetta_utils.mazepa.upkeep_handlers.multiprocessing.Queue", return_value=mock_queue
        )
        mock_process = mocker.MagicMock()
        mock_process.is_alive.return_value = False
        mocker.patch(
            "zetta_utils.mazepa.upkeep_handlers.multiprocessing.Process",
            return_value=mock_process,
        )

        manager = SQSUpkeepHandlerManager()
        manager.start()
        manager.shutdown()

        # Verify shutdown command was sent
        mock_queue.put.assert_called_once()
        cmd = mock_queue.put.call_args[0][0]
        assert cmd.action == "shutdown"

        # Verify process was joined
        mock_process.join.assert_called()

    def test_shutdown_terminates_if_process_alive(self, mocker):
        mock_queue = mocker.MagicMock()
        mocker.patch(
            "zetta_utils.mazepa.upkeep_handlers.multiprocessing.Queue", return_value=mock_queue
        )
        mock_process = mocker.MagicMock()
        mock_process.is_alive.return_value = True
        mocker.patch(
            "zetta_utils.mazepa.upkeep_handlers.multiprocessing.Process",
            return_value=mock_process,
        )

        manager = SQSUpkeepHandlerManager()
        manager.start()
        manager.shutdown(timeout=0.1)

        mock_process.terminate.assert_called_once()

    def test_start_upkeep_when_not_running_logs_warning(self, mocker):
        mock_logger = mocker.patch("zetta_utils.mazepa.upkeep_handlers.logger")

        manager = SQSUpkeepHandlerManager()
        manager.start_upkeep(
            task_id="task-123",
            receipt_handle="handle",
            visibility_timeout=60,
            interval_sec=10.0,
            queue_name="queue",
            region_name="us-east-1",
        )

        mock_logger.warning.assert_called_once()
        assert "not running" in mock_logger.warning.call_args[0][0]

    def test_stop_upkeep_when_not_running_logs_warning(self, mocker):
        mock_logger = mocker.patch("zetta_utils.mazepa.upkeep_handlers.logger")

        manager = SQSUpkeepHandlerManager()
        manager.stop_upkeep("task-123")

        mock_logger.warning.assert_called_once()
        assert "not running" in mock_logger.warning.call_args[0][0]

    def test_start_upkeep_sends_command(self, mocker):
        mock_queue = mocker.MagicMock()
        mocker.patch(
            "zetta_utils.mazepa.upkeep_handlers.multiprocessing.Queue", return_value=mock_queue
        )
        mock_process = mocker.MagicMock()
        mocker.patch(
            "zetta_utils.mazepa.upkeep_handlers.multiprocessing.Process",
            return_value=mock_process,
        )

        manager = SQSUpkeepHandlerManager()
        manager.start()
        manager.start_upkeep(
            task_id="task-123",
            receipt_handle="handle-456",
            visibility_timeout=60,
            interval_sec=10.0,
            queue_name="my-queue",
            region_name="us-east-1",
            endpoint_url="http://localhost:9324",
        )

        mock_queue.put_nowait.assert_called_once()
        cmd = mock_queue.put_nowait.call_args[0][0]
        assert cmd.action == "start_upkeep"
        assert cmd.task_id == "task-123"
        assert cmd.receipt_handle == "handle-456"
        assert cmd.visibility_timeout == 60
        assert cmd.interval_sec == 10.0
        assert cmd.queue_name == "my-queue"
        assert cmd.region_name == "us-east-1"
        assert cmd.endpoint_url == "http://localhost:9324"

    def test_stop_upkeep_sends_command(self, mocker):
        mock_queue = mocker.MagicMock()
        mocker.patch(
            "zetta_utils.mazepa.upkeep_handlers.multiprocessing.Queue", return_value=mock_queue
        )
        mock_process = mocker.MagicMock()
        mocker.patch(
            "zetta_utils.mazepa.upkeep_handlers.multiprocessing.Process",
            return_value=mock_process,
        )

        manager = SQSUpkeepHandlerManager()
        manager.start()
        manager.stop_upkeep("task-123")

        mock_queue.put_nowait.assert_called_once()
        cmd = mock_queue.put_nowait.call_args[0][0]
        assert cmd.action == "stop_upkeep"
        assert cmd.task_id == "task-123"

    def test_is_running_false_before_start(self):
        manager = SQSUpkeepHandlerManager()
        assert not manager.is_running

    def test_is_running_true_after_start(self, mocker):
        mock_queue = mocker.MagicMock()
        mocker.patch(
            "zetta_utils.mazepa.upkeep_handlers.multiprocessing.Queue", return_value=mock_queue
        )
        mock_process = mocker.MagicMock()
        mock_process.is_alive.return_value = True
        mocker.patch(
            "zetta_utils.mazepa.upkeep_handlers.multiprocessing.Process",
            return_value=mock_process,
        )

        manager = SQSUpkeepHandlerManager()
        manager.start()
        assert manager.is_running

    def test_is_running_false_when_process_dead(self, mocker):
        mock_queue = mocker.MagicMock()
        mocker.patch(
            "zetta_utils.mazepa.upkeep_handlers.multiprocessing.Queue", return_value=mock_queue
        )
        mock_process = mocker.MagicMock()
        mock_process.is_alive.return_value = False
        mocker.patch(
            "zetta_utils.mazepa.upkeep_handlers.multiprocessing.Process",
            return_value=mock_process,
        )

        manager = SQSUpkeepHandlerManager()
        manager.start()
        assert not manager.is_running


class TestRunSqsUpkeepHandler:
    def test_shutdown_command_exits_loop(self, mocker):
        # worker_init is imported inside the function, so patch at source
        mocker.patch("zetta_utils.mazepa.worker.worker_init")

        command_queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
        command_queue.put(UpkeepCommand(action="shutdown"))

        # Run in thread so we can verify it exits
        handler_thread = threading.Thread(
            target=run_sqs_upkeep_handler,
            args=(command_queue, "INFO"),
        )
        handler_thread.start()
        handler_thread.join(timeout=5.0)

        assert not handler_thread.is_alive()

    def test_start_and_stop_upkeep(self, mocker):
        mocker.patch("zetta_utils.mazepa.worker.worker_init")
        mock_change_visibility = mocker.patch(
            "zetta_utils.message_queues.sqs.utils.change_message_visibility"
        )

        command_queue: multiprocessing.Queue[Any] = multiprocessing.Queue()

        # Start upkeep with short interval
        command_queue.put(
            UpkeepCommand(
                action="start_upkeep",
                task_id="task-123",
                receipt_handle="handle-456",
                visibility_timeout=60,
                interval_sec=0.1,
                queue_name="my-queue",
                region_name="us-east-1",
                endpoint_url=None,
            )
        )

        handler_thread = threading.Thread(
            target=run_sqs_upkeep_handler,
            args=(command_queue, "INFO"),
        )
        handler_thread.start()

        # Wait for upkeep to fire at least once
        time.sleep(0.3)

        # Stop upkeep and shutdown
        command_queue.put(UpkeepCommand(action="stop_upkeep", task_id="task-123"))
        command_queue.put(UpkeepCommand(action="shutdown"))

        handler_thread.join(timeout=5.0)
        assert not handler_thread.is_alive()

        # Verify change_message_visibility was called
        assert mock_change_visibility.call_count >= 1
        mock_change_visibility.assert_called_with(
            receipt_handle="handle-456",
            visibility_timeout=60,
            queue_name="my-queue",
            region_name="us-east-1",
            endpoint_url=None,
        )

    def test_duplicate_start_upkeep_ignored(self, mocker):
        mocker.patch("zetta_utils.mazepa.worker.worker_init")
        mocker.patch("zetta_utils.message_queues.sqs.utils.change_message_visibility")

        command_queue: multiprocessing.Queue[Any] = multiprocessing.Queue()

        # Start upkeep twice for same task
        for _ in range(2):
            command_queue.put(
                UpkeepCommand(
                    action="start_upkeep",
                    task_id="task-123",
                    receipt_handle="handle-456",
                    visibility_timeout=60,
                    interval_sec=1.0,
                    queue_name="my-queue",
                    region_name="us-east-1",
                    endpoint_url=None,
                )
            )

        command_queue.put(UpkeepCommand(action="shutdown"))

        handler_thread = threading.Thread(
            target=run_sqs_upkeep_handler,
            args=(command_queue, "INFO"),
        )
        handler_thread.start()
        handler_thread.join(timeout=5.0)

        # Test passes if no crash - duplicate is logged and ignored
        assert not handler_thread.is_alive()

    def test_stop_nonexistent_upkeep_ignored(self, mocker):
        mocker.patch("zetta_utils.mazepa.worker.worker_init")

        command_queue: multiprocessing.Queue[Any] = multiprocessing.Queue()

        # Stop upkeep for task that doesn't exist
        command_queue.put(UpkeepCommand(action="stop_upkeep", task_id="nonexistent-task"))
        command_queue.put(UpkeepCommand(action="shutdown"))

        handler_thread = threading.Thread(
            target=run_sqs_upkeep_handler,
            args=(command_queue, "INFO"),
        )
        handler_thread.start()
        handler_thread.join(timeout=5.0)

        # Test passes if no crash - nonexistent stop is logged and ignored
        assert not handler_thread.is_alive()

    def test_unknown_action_logged(self, mocker):
        mocker.patch("zetta_utils.mazepa.worker.worker_init")

        command_queue: multiprocessing.Queue[Any] = multiprocessing.Queue()

        command_queue.put(UpkeepCommand(action="unknown_action"))
        command_queue.put(UpkeepCommand(action="shutdown"))

        handler_thread = threading.Thread(
            target=run_sqs_upkeep_handler,
            args=(command_queue, "INFO"),
        )
        handler_thread.start()
        handler_thread.join(timeout=5.0)

        # Test passes if no crash - unknown action is logged and ignored
        assert not handler_thread.is_alive()
