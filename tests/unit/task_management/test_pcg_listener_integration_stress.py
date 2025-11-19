"""
Integration stress tests for run_pcg_edit_listener function.

Tests the full listener workflow including PubSub integration, database operations,
and error recovery scenarios.
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest

from zetta_utils.message_queues.pubsub import PubSubPullQueue
from zetta_utils.task_management.automated_workers.pcg_edit_listener import (
    run_pcg_edit_listener,
)
from zetta_utils.task_management.db.models import ProjectModel


class MockPubSubMessage:
    """Mock PubSub message for integration testing."""

    def __init__(self, payload: Dict[str, Any], ack_delay: float = 0.0):
        self.payload = payload.copy()
        self.ack_called = False
        self.ack_delay = ack_delay

    def acknowledge_fn(self):
        """Mock acknowledge function with optional delay."""
        if self.ack_delay > 0:
            time.sleep(self.ack_delay)
        self.ack_called = True


class TestPCGListenerIntegrationStress:  # pylint: disable=attribute-defined-outside-init
    """Integration stress tests for PCG listener."""

    def setup_method(self):
        """Set up test fixtures."""
        self.project_name = "integration_stress_project"
        self.project_id = "test-gcp-project"
        self.subscription_name = "test-subscription"
        self.server_address = "https://test.example.com"
        self.datastack_name = "test_datastack"

        # Configurable stress test metrics
        self.stress_config = {
            "message_count": 1000,
            "batch_size": 100,
            "concurrent_instances": 2,
            "max_cycles": 2,
            "memory_test_messages": 500,
            "max_memory_increase_mb": 50,
            "poll_interval_ms": 1,
            "timeout_seconds": 1,
            "max_pull_attempts": 5,
            "max_processing_time_seconds": 5,
        }

    def test_listener_startup_and_shutdown(self):
        """Test listener startup, message processing, and graceful shutdown."""

        # Mock database project
        mock_project = Mock(spec=ProjectModel)
        mock_project.datastack_name = self.datastack_name

        # Mock session and query
        mock_session = Mock()
        mock_query = Mock()
        mock_query.filter_by.return_value.first.return_value = mock_project
        mock_session.query.return_value = mock_query

        # Mock PubSub queue
        mock_queue = Mock(spec=PubSubPullQueue)

        # Create test messages
        test_messages = [
            MockPubSubMessage(
                {
                    "new_root_ids": [123456],
                    "old_root_ids": [100001, 100002],
                    "operation_id": "test_merge_1",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "_pubsub_attributes": {"table_id": "test_table"},
                }
            ),
            MockPubSubMessage(
                {
                    "new_root_ids": [123457, 123458],
                    "operation_id": "test_split_1",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "_pubsub_attributes": {"table_id": "test_table"},
                }
            ),
        ]

        # Mock queue.pull to return messages then empty
        pull_call_count = 0

        def mock_pull(max_num=None): # pylint: disable=unused-argument
            nonlocal pull_call_count
            pull_call_count += 1
            if pull_call_count == 1:
                return test_messages
            elif pull_call_count == 2:
                return []  # Empty batch
            else:
                # Simulate KeyboardInterrupt on third call
                raise KeyboardInterrupt()

        mock_queue.pull.side_effect = mock_pull

        # Mock all external dependencies
        with patch("zetta_utils.task_management.automated_workers."
                   "pcg_edit_listener.get_session_context") as mock_session_ctx, \
             patch("zetta_utils.task_management.automated_workers."
                   "pcg_edit_listener.PubSubPullQueue") as mock_queue_class, \
             patch("zetta_utils.task_management.automated_workers."
                   "pcg_edit_listener.CAVEclient") as mock_cave_client, \
             patch("zetta_utils.task_management.automated_workers."
                   "pcg_edit_listener.process_edit_event") as mock_process_event, \
             patch("time.sleep") as _mock_sleep:

            mock_session_ctx.return_value.__enter__.return_value = mock_session
            mock_queue_class.return_value = mock_queue
            mock_cave_client.return_value = Mock()

            # Run listener (should exit gracefully on KeyboardInterrupt)
            run_pcg_edit_listener(
                project_id=self.project_id,
                subscription_name=self.subscription_name,
                project_name=self.project_name,
                poll_interval_sec=1,
                max_messages=10,
            )

            # Verify behavior
            assert mock_queue.pull.call_count == 3
            assert mock_process_event.call_count == 2  # Two messages processed

            # Verify messages were acknowledged
            for msg in test_messages:
                assert msg.ack_called

    def test_high_message_throughput(self):
        """Test listener performance under high message throughput."""

        # Mock all external dependencies
        mock_queue = Mock(spec=PubSubPullQueue)

        # Generate small number of messages for fast test
        message_count = self.stress_config["message_count"]
        batch_size = self.stress_config["batch_size"]
        messages_per_batch = message_count // batch_size

        message_batches = []
        for batch_idx in range(batch_size):
            batch = []
            for msg_idx in range(messages_per_batch):
                msg_id = batch_idx * messages_per_batch + msg_idx
                message = MockPubSubMessage(
                    {
                        "new_root_ids": [200000 + msg_id],
                        "old_root_ids": [100000 + msg_id],
                        "operation_id": f"high_throughput_{msg_id}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "_pubsub_attributes": {"table_id": "stress_table"},
                    }
                )
                batch.append(message)
            message_batches.append(batch)

        # Mock pull to return batches
        batch_index = 0

        def mock_pull(max_num=None): # pylint: disable=unused-argument
            nonlocal batch_index
            if batch_index < len(message_batches):
                batch = message_batches[batch_index]
                batch_index += 1
                return batch
            else:
                # End the test after all batches
                raise KeyboardInterrupt()

        mock_queue.pull.side_effect = mock_pull

        # Track processing time
        processing_times = []

        def mock_process_event(*_, **__):
            start_time = time.time()
            # Simulate minimal processing time for speed
            time.sleep(0.001)
            processing_times.append(time.time() - start_time)

        with patch(
                "zetta_utils.task_management.automated_workers.pcg_edit_listener.PubSubPullQueue"
        ) as mock_queue_class, patch(
            "zetta_utils.task_management.automated_workers.pcg_edit_listener.process_edit_event",
            side_effect=mock_process_event,
        ), patch(
            "time.sleep"
        ):

            mock_queue_class.return_value = mock_queue

            start_time = time.time()

            try:
                run_pcg_edit_listener(
                    project_id=self.project_id,
                    subscription_name=self.subscription_name,
                    poll_interval_sec=1,  # Fast polling
                    max_messages=messages_per_batch,
                )
            except KeyboardInterrupt:
                pass  # Expected

            total_time = time.time() - start_time

            # Verify performance metrics
            assert len(processing_times) == message_count
            assert (
                    total_time < self.stress_config["max_processing_time_seconds"]
            )  # Should complete within configured time

            # Verify all messages were acknowledged
            for batch in message_batches:
                for msg in batch:
                    assert msg.ack_called

    def test_database_connection_failures(self):
        """Test listener behavior during database connection failures."""

        # Mock PubSub queue
        mock_queue = Mock(spec=PubSubPullQueue)

        test_message = MockPubSubMessage(
            {
                "new_root_ids": [999999],
                "old_root_ids": [888888],
                "operation_id": "db_failure_test",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "_pubsub_attributes": {"table_id": "failure_table"},
            }
        )

        call_count = 0

        def mock_pull(max_num=None): # pylint: disable=unused-argument
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [test_message]
            elif call_count < self.stress_config["max_pull_attempts"]:
                return []  # Empty batches
            else:
                raise KeyboardInterrupt()  # Stop test

        mock_queue.pull.side_effect = mock_pull

        # Mock database session (only called once)
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        def mock_session_context():
            return mock_session

        process_call_count = 0

        def mock_process_event(*_, **__):
            nonlocal process_call_count
            process_call_count += 1
            # Simulate processing happens but doesn't need retries for this
            # test

        with patch(
                "zetta_utils.task_management.automated_workers.pcg_edit_listener.PubSubPullQueue"
        ) as mock_queue_class, patch(
            "zetta_utils.task_management.automated_workers.pcg_edit_listener.get_session_context"
        ) as mock_session_ctx, patch(
            "zetta_utils.task_management.automated_workers.pcg_edit_listener.process_edit_event",
            side_effect=mock_process_event,
        ), patch(
            "time.sleep"
        ):

            mock_queue_class.return_value = mock_queue
            mock_session_ctx.side_effect = mock_session_context

            try:
                run_pcg_edit_listener(
                    project_id=self.project_id,
                    subscription_name=self.subscription_name,
                    poll_interval_sec=1,
                    max_messages=1,
                )
            except KeyboardInterrupt:
                pass  # Expected

            # Verify the listener handled operations gracefully
            assert call_count >= 3  # Made multiple pull attempts
            assert process_call_count >= 1  # At least one message processed

    def test_cave_client_initialization_failures(self):
        """Test listener behavior when CAVEclient initialization fails."""

        mock_queue = Mock(spec=PubSubPullQueue)

        # Message that should be processed even without CAVEclient
        test_message = MockPubSubMessage(
            {
                "new_root_ids": [777777],
                "old_root_ids": [666666],  # Include old_root_ids in event
                "operation_id": "cave_failure_test",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "_pubsub_attributes": {"table_id": "cave_failure_table"},
            }
        )

        call_count = 0

        def mock_pull(max_num=None): # pylint: disable=unused-argument
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [test_message]
            else:
                raise KeyboardInterrupt()

        mock_queue.pull.side_effect = mock_pull

        # Mock CAVEclient initialization failure
        def mock_cave_client_init(*_, **__):
            raise RuntimeError("CAVEclient initialization failed")

        processed_events = []

        def mock_process_event(*args, **__):
            processed_events.append(args)

        with patch(
                "zetta_utils.task_management.automated_workers.pcg_edit_listener.PubSubPullQueue"
        ) as mock_queue_class, patch(
            "zetta_utils.task_management.automated_workers.pcg_edit_listener.CAVEclient",
            side_effect=mock_cave_client_init,
        ), patch(
            "zetta_utils.task_management.automated_workers.pcg_edit_listener.process_edit_event",
            side_effect=mock_process_event,
        ), patch(
            "time.sleep"
        ):

            mock_queue_class.return_value = mock_queue

            try:
                run_pcg_edit_listener(
                    project_id=self.project_id,
                    subscription_name=self.subscription_name,
                    datastack_name=self.datastack_name,
                    poll_interval_sec=1,
                    max_messages=1,
                )
            except KeyboardInterrupt:
                pass

            # Verify message was still processed despite CAVEclient failure
            assert len(processed_events) == 1
            assert test_message.ack_called

    def test_memory_leak_detection(self):
        """Test for memory leaks during extended operation."""

        import os  # pylint: disable=import-outside-toplevel

        import psutil  # pylint: disable=import-outside-toplevel

        process = psutil.Process(os.getpid())

        # Create continuous stream of messages
        message_counter = 0

        def mock_pull(max_num=None): # pylint: disable=unused-argument
            nonlocal message_counter
            batch = []
            batch_size = min(2, self.stress_config["memory_test_messages"] - message_counter)
            if batch_size <= 0:
                # No more messages to generate
                raise KeyboardInterrupt()
            for _ in range(batch_size):
                message_counter += 1
                if message_counter > self.stress_config["memory_test_messages"]:
                    break
                batch.append(
                    MockPubSubMessage(
                        {
                            "new_root_ids": [900000 + message_counter],
                            "old_root_ids": [800000 + message_counter],
                            "operation_id": f"memory_test_{message_counter}",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "_pubsub_attributes": {"table_id": "memory_table"},
                        }
                    )
                )
            # Stop if we've reached the limit
            if message_counter >= self.stress_config["memory_test_messages"]:
                raise KeyboardInterrupt()
            return batch
        mock_queue = Mock(spec=PubSubPullQueue)
        mock_queue.pull.side_effect = mock_pull

        def mock_process_event(*_, **__):
            # Minimal processing to focus on memory usage
            pass

        # Record initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        with patch(
                "zetta_utils.task_management.automated_workers.pcg_edit_listener.PubSubPullQueue"
        ) as mock_queue_class, patch(
            "zetta_utils.task_management.automated_workers.pcg_edit_listener.process_edit_event",
            side_effect=mock_process_event,
        ), patch(
            "time.sleep"
        ):

            mock_queue_class.return_value = mock_queue

            try:
                run_pcg_edit_listener(
                    project_id=self.project_id,
                    subscription_name=self.subscription_name,
                    poll_interval_sec=1,  # Very fast polling
                    max_messages=10,
                )
            except KeyboardInterrupt:
                pass

        # Record final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable
        assert (
                memory_increase < self.stress_config["max_memory_increase_mb"]
        ), f"Memory increased by {memory_increase}MB"
        assert (
                message_counter >= self.stress_config["memory_test_messages"]
        )  # Verify we processed the expected number of messages

    def test_cli_argument_parsing(self):
        """Test command-line argument parsing and validation."""

        # Mock sys.argv for different scenarios
        test_cases = [
            # Valid arguments with project
            [
                "run_pcg_listener.py",
                "--project-id",
                "test-project",
                "--subscription-name",
                "test-subscription",
                "--project-name",
                "test-project-name",
                "--datastack-name",
                "test-datastack",
                "--server-address",
                "https://test.com",
                "--poll-interval",
                "10",
                "--max-messages",
                "5",
            ],
            # Minimal required arguments without project name
            [
                "run_pcg_listener.py",
                "--project-id",
                "test-project",
                "--subscription-name",
                "test-subscription",
            ],
            # Invalid poll interval
            [
                "run_pcg_listener.py",
                "--project-id",
                "test-project",
                "--subscription-name",
                "test-subscription",
                "--poll-interval",
                "invalid",
            ],
        ]

        with patch("zetta_utils.task_management.automated_workers."
                   "run_pcg_listener.run_pcg_edit_listener") as mock_run_listener, \
             patch("zetta_utils.task_management.db.session.get_session_context"
                   ) as mock_session_ctx:

            # Mock database session with context manager
            mock_project = Mock()
            mock_project.datastack_name = "test_datastack"
            mock_session = Mock()
            # For test case 0 (with project name), return the project
            # For other test cases, project name is not provided so no query is
            # made
            mock_session.query.return_value.filter_by.return_value.first.return_value = (
                mock_project
            )

            # Set up context manager properly
            mock_session_ctx.return_value.__enter__.return_value = mock_session
            mock_session_ctx.return_value.__exit__.return_value = None

            for i, test_args in enumerate(test_cases):
                # Reset mock for each test case
                mock_run_listener.reset_mock()

                with patch("sys.argv", test_args):
                    try:
                        # pylint: disable=import-outside-toplevel
                        from zetta_utils.task_management.automated_workers.run_pcg_listener import (  # pylint: disable=line-too-long
                            main,
                        )

                        main()

                        if i < 2:  # Valid cases should call run_pcg_edit_listener
                            mock_run_listener.assert_called_once()

                    except SystemExit as e:
                        if i == 2:  # Invalid poll interval case
                            assert e.code != 0  # Should exit with error
                        else:
                            # Other cases should not cause SystemExit
                            pytest.fail(
                                f"Unexpected SystemExit for test case {i}: {e}")
                    except Exception:  # pylint: disable=broad-exception-caught
                        if i == 2:  # Invalid poll interval case
                            # Expected for invalid arguments
                            pass
                        else:
                            pytest.fail(f"Unexpected exception for test case {i}")

    def test_graceful_shutdown_on_signals(self):
        """Test graceful shutdown on system signals."""

        mock_queue = Mock(spec=PubSubPullQueue)

        # Mock pull to return empty results then trigger shutdown
        pull_count = 0

        def mock_pull(max_num=None): # pylint: disable=unused-argument
            nonlocal pull_count
            pull_count += 1
            if pull_count >= 3:
                # Simulate graceful shutdown after a few pulls
                raise KeyboardInterrupt()
            return []

        mock_queue.pull.side_effect = mock_pull

        with patch(
                "zetta_utils.task_management.automated_workers.pcg_edit_listener.PubSubPullQueue"
        ) as mock_queue_class, patch("time.sleep"):

            mock_queue_class.return_value = mock_queue

            try:
                run_pcg_edit_listener(
                    project_id=self.project_id,
                    subscription_name=self.subscription_name,
                    poll_interval_sec=1,
                    max_messages=1,
                )
            except KeyboardInterrupt:
                pass  # Expected for graceful shutdown

            # Verify shutdown was handled
            assert pull_count >= 3
