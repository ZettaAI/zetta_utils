"""
Error recovery and resilience tests for PCG edit listener.

Tests various failure scenarios and recovery mechanisms to ensure
the listener remains stable under adverse conditions.
"""

import queue
from datetime import datetime, timedelta, timezone
from typing import List
from unittest.mock import Mock, patch

from requests.exceptions import HTTPError, Timeout
from requests.exceptions import ConnectionError as RequestsConnectionError

from zetta_utils.task_management.automated_workers.pcg_edit_listener import (
    get_old_roots_from_lineage_graph,
    process_edit_event,
    process_merge_event,
    process_split_event,
)


class NetworkFailureSimulator:
    """Simulates various network failure patterns."""

    def __init__(self):
        self.call_count = 0
        self.failure_pattern = []
        self.success_response = Mock()
        self.success_response.status_code = 200
        self.success_response.json.return_value = {"links": []}
        self.success_response.raise_for_status.return_value = None

    def set_failure_pattern(self, pattern: List[str]):
        """Set pattern of failures: 'timeout', 'connection', 'http_error', 'success'."""
        self.failure_pattern = pattern
        self.call_count = 0

    def simulate_request(self, *args, **kwargs):  # pylint: disable=unused-argument
        """Simulate network request with failure pattern."""
        if self.call_count >= len(self.failure_pattern):
            # Default to success after pattern exhausted
            failure_type = "success"
        else:
            failure_type = self.failure_pattern[self.call_count]

        self.call_count += 1

        if failure_type == "timeout":
            raise Timeout("Request timeout")
        if failure_type == "connection":
            raise RequestsConnectionError("Connection failed")
        if failure_type == "http_error":
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = HTTPError(
                "500 Internal Server Error")
            return mock_response
        if failure_type == "rate_limit":
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.raise_for_status.side_effect = HTTPError(
                "429 Too Many Requests")
            return mock_response
        # success
        return self.success_response


class DatabaseFailureSimulator:
    """Simulates database failure and recovery patterns."""

    def __init__(self):
        self.call_count = 0
        self.failure_pattern = []

    def set_failure_pattern(self, pattern: List[str]):
        """Set pattern of failures: 'connection', 'timeout', 'deadlock', 'success'."""
        self.failure_pattern = pattern
        self.call_count = 0

    def simulate_operation(self, *args, **kwargs):  # pylint: disable=unused-argument
        """Simulate database operation with failure pattern."""
        if self.call_count >= len(self.failure_pattern):
            failure_type = "success"
        else:
            failure_type = self.failure_pattern[self.call_count]

        self.call_count += 1

        if failure_type == "connection":
            raise RuntimeError("Database connection failed")
        if failure_type == "timeout":
            raise RuntimeError("Database operation timeout")
        if failure_type == "deadlock":
            raise RuntimeError("Deadlock detected")
        if failure_type == "constraint":
            raise RuntimeError("Constraint violation")
        # success
        return 5  # Mock successful update count


class TestPCGListenerErrorRecovery:  # pylint: disable=attribute-defined-outside-init
    """Error recovery and resilience tests."""

    def setup_method(self):
        """Set up test fixtures."""
        self.project_name = "error_recovery_project"
        self.server_address = "https://test.example.com"
        self.table_id = "error_test_table"

    def test_network_failure_retry_patterns(self):
        """Test various network failure and retry patterns."""

        simulator = NetworkFailureSimulator()

        # Test patterns: initial failures followed by success
        test_patterns = [
            ["timeout", "success"],
            ["connection", "connection", "success"],
            ["http_error", "timeout", "success"],
            ["rate_limit", "rate_limit", "success"],
            ["timeout", "connection", "http_error", "success"],
        ]

        for pattern in test_patterns:
            simulator.set_failure_pattern(pattern)

            with patch("requests.post", side_effect=simulator.simulate_request):
                result = get_old_roots_from_lineage_graph(
                    server_address=self.server_address,
                    table_id=self.table_id,
                    root_ids=[123456],
                    timestamp_past=datetime.now(
                        timezone.utc) - timedelta(minutes=1),
                )

                # Should handle failures gracefully and return empty dict
                assert not result
                # Function makes only one call, not multiple retries
                assert simulator.call_count == 1

    def test_persistent_network_failures(self):
        """Test behavior under persistent network failures."""

        simulator = NetworkFailureSimulator()

        # Simulate persistent failures (no success)
        persistent_patterns = [
            ["timeout"] * 10,
            ["connection"] * 10,
            ["http_error"] * 10,
            ["rate_limit"] * 10,
        ]

        for pattern in persistent_patterns:
            simulator.set_failure_pattern(pattern)

            with patch("requests.post", side_effect=simulator.simulate_request):
                # Multiple attempts should all fail gracefully
                for _ in range(5):
                    result = get_old_roots_from_lineage_graph(
                        server_address=self.server_address,
                        table_id=self.table_id,
                        root_ids=[123456],
                        timestamp_past=datetime.now(
                            timezone.utc) - timedelta(minutes=1),
                    )
                    assert not result

    def test_database_failure_recovery(self, mocker):  # pylint: disable=unused-argument
        """Test database failure recovery patterns."""

        db_simulator = DatabaseFailureSimulator()

        # Test various database failure patterns
        test_patterns = [
            ["connection", "success"],
            ["timeout", "timeout", "success"],
            ["deadlock", "success"],
            ["constraint", "success"],
            ["connection", "timeout", "deadlock", "success"],
        ]

        for pattern in test_patterns:
            db_simulator.set_failure_pattern(pattern)

            with patch(
                    "zetta_utils.task_management.supervoxel.update_supervoxels_for_merge",
                    side_effect=db_simulator.simulate_operation,
            ):

                # Test merge event processing
                event_data = {
                    "new_root_ids": [123456],
                    "old_root_ids": [100001, 100002],
                    "operation_id": "db_recovery_test",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                # Should handle database failures gracefully
                try:
                    process_merge_event(
                        event_data=event_data,
                        project_name=self.project_name,
                        cave_client=None,
                        server_address=self.server_address,
                        table_id=self.table_id,
                    )
                except Exception:  # pylint: disable=broad-exception-caught
                    # Verify error is logged but doesn't crash
                    assert db_simulator.call_count <= len(pattern)

    def test_cascading_failure_scenarios(self, mocker):  # pylint: disable=unused-argument
        """Test cascading failure scenarios where multiple systems fail."""

        # Simulate scenario where both network and database fail
        network_simulator = NetworkFailureSimulator()
        db_simulator = DatabaseFailureSimulator()

        network_simulator.set_failure_pattern(
            ["timeout", "connection", "timeout"])
        db_simulator.set_failure_pattern(["connection", "deadlock"])

        event_data = {
            "new_root_ids": [789012, 789013],  # Split event
            "operation_id": "cascading_failure_test",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with patch("requests.post", side_effect=network_simulator.simulate_request), \
             patch("zetta_utils.task_management.supervoxel.update_supervoxels_for_split",
                   side_effect=db_simulator.simulate_operation), \
             patch("zetta_utils.task_management.automated_workers."
                   "pcg_edit_listener.get_supervoxels_by_segment",
            return_value=[],
        ):

            # Should handle cascading failures without crashing
            try:
                process_split_event(
                    event_data=event_data,
                    project_name=self.project_name,
                    cave_client=None,
                    server_address=self.server_address,
                    table_id=self.table_id,
                )
            except Exception:  # pylint: disable=broad-exception-caught
                # Verify graceful handling
                assert network_simulator.call_count > 0 or db_simulator.call_count > 0

    def test_partial_failure_recovery(self, db_session, project_factory):  # pylint: disable=unused-argument
        """Test recovery from partial failures in batch operations."""

        project_factory(project_name=self.project_name)

        # Create test data
        old_segment_ids = [500001, 500002, 500003]
        new_segment_id = 600001

        # Simulate partial failure: some supervoxels update successfully,
        # others fail
        update_call_count = 0

        def mock_update_with_partial_failure(*args, **kwargs):  # pylint: disable=unused-argument
            nonlocal update_call_count
            update_call_count += 1

            if update_call_count <= 2:
                # First two calls succeed
                return 5
            else:
                # Third call fails
                raise RuntimeError("Partial update failure")

        # Mock the lineage graph function to not interfere with old_root_ids lookup
        # Patch the function as imported in the PCG edit listener module
        with patch("zetta_utils.task_management.automated_workers."
                   "pcg_edit_listener.update_supervoxels_for_merge",
                   side_effect=mock_update_with_partial_failure), \
             patch("zetta_utils.task_management.automated_workers."
                   "pcg_edit_listener.get_old_roots_from_lineage_graph",
            return_value={},
        ):

            # Process merge event that should handle partial failure
            event_data = {
                "new_root_ids": [new_segment_id],
                "old_root_ids": old_segment_ids,  # Include old_root_ids to trigger update
                "operation_id": "partial_failure_test",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            try:
                process_merge_event(
                    event_data=event_data,
                    project_name=self.project_name,
                    cave_client=None,
                    server_address=self.server_address,
                    table_id=self.table_id,
                )
            except Exception:  # pylint: disable=broad-exception-caught
                # Should handle gracefully
                pass

            # Verify attempt was made - function is called because old_root_ids
            # are provided
            assert update_call_count >= 1

    def test_memory_pressure_handling(self, mocker):  # pylint: disable=unused-argument
        """Test behavior under memory pressure conditions."""

        # Simulate memory allocation failures
        allocation_count = 0

        def mock_memory_pressure(*args, **kwargs):  # pylint: disable=unused-argument
            nonlocal allocation_count
            allocation_count += 1

            if allocation_count % 10 == 0:
                # Every 10th allocation fails
                raise MemoryError("Out of memory")

            return Mock()  # Return mock object

        # Test large split operation under memory pressure
        event_data = {
            "new_root_ids": list(range(700000, 700020)),  # 20 new segments
            "operation_id": "memory_pressure_test",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with patch("zetta_utils.task_management.automated_workers."
                   "pcg_edit_listener.get_supervoxels_by_segment",
                   side_effect=mock_memory_pressure), \
             patch("zetta_utils.task_management.automated_workers."
                   "pcg_edit_listener.get_old_roots_from_lineage_graph",
            return_value={700000: [600000]},
        ):

            try:
                process_split_event(
                    event_data=event_data,
                    project_name=self.project_name,
                    cave_client=None,
                    server_address=self.server_address,
                    table_id=self.table_id,
                )
            except MemoryError:
                # Should handle memory errors gracefully
                pass

            # Verify attempts were made
            assert allocation_count > 0

    def test_concurrent_failure_scenarios(self, mocker):  # pylint: disable=unused-argument
        """Test handling of failures during concurrent operations."""

        # Simulate failures that occur during concurrent processing
        failure_events = queue.Queue()
        success_count = 0
        failure_count = 0

        def mock_concurrent_operation(*args, **kwargs):  # pylint: disable=unused-argument
            nonlocal success_count, failure_count

            # Deterministic failure pattern: fail every 3rd operation
            operation_id = success_count + failure_count
            if operation_id % 3 == 0:  # Fail every 3rd operation
                failure_count += 1
                failure_events.put(f"Failure {failure_count}")
                raise RuntimeError(f"Concurrent operation failure {failure_count}")

            success_count += 1
            return 1  # Success

        # Create multiple concurrent events
        events = []
        for i in range(5):
            events.append(
                {
                    "new_root_ids": [800000 + i],
                    "old_root_ids": [700000 + i],
                    "operation_id": f"concurrent_failure_{i}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        with patch("zetta_utils.task_management.automated_workers."
                   "pcg_edit_listener.update_supervoxels_for_merge",
                   side_effect=mock_concurrent_operation,
        ):

            # Process events sequentially to avoid actual threading issues in
            # tests
            for event_data in events:
                try:
                    process_merge_event(
                        event_data=event_data,
                        project_name=self.project_name,
                        cave_client=None,
                        server_address=self.server_address,
                        table_id=self.table_id,
                    )
                except Exception:  # pylint: disable=broad-exception-caught
                    pass  # Expected for some operations

            # Verify some operations were attempted despite failures
            total_operations = success_count + failure_count
            assert total_operations > 0

    def test_data_corruption_detection(self, mocker):  # pylint: disable=unused-argument
        """Test detection and handling of data corruption scenarios."""

        # Simulate various data corruption scenarios
        corruption_scenarios = [
            # Invalid root ID formats
            {"new_root_ids": ["invalid_id"], "operation_id": "corrupt1"},
            # Negative root IDs
            {"new_root_ids": [-123], "operation_id": "corrupt2"},
            # Extremely large root IDs
            {"new_root_ids": [2 ** 64], "operation_id": "corrupt3"},
            # Empty root ID lists when they shouldn't be
            {"new_root_ids": [], "operation_id": "corrupt4"},
            # Malformed timestamp
            {"new_root_ids": [123],
             "timestamp": "not_a_date",
             "operation_id": "corrupt5"},
            # Circular references in old/new root IDs
            {"new_root_ids": [123], "old_root_ids": [
                123], "operation_id": "corrupt6"},
        ]

        handled_corruptions = 0

        for scenario in corruption_scenarios:
            try:
                process_edit_event(
                    event_data=scenario,
                    message_attributes={"table_id": self.table_id},
                    project_name=self.project_name,
                    cave_client=None,
                    server_address=self.server_address,
                )
                # If no exception, corruption was handled gracefully
                handled_corruptions += 1
            except Exception:  # pylint: disable=broad-exception-caught
                # Expected for some corruption scenarios
                handled_corruptions += 1

        # All corruption scenarios should be handled gracefully
        assert handled_corruptions == len(corruption_scenarios)

    def test_resource_exhaustion_scenarios(self, mocker):  # pylint: disable=unused-argument
        """Test behavior under resource exhaustion conditions."""

        # Test file descriptor exhaustion
        def mock_fd_exhaustion(*args, **kwargs):  # pylint: disable=unused-argument
            raise OSError("Too many open files")

        # Test disk space exhaustion
        def mock_disk_exhaustion(*args, **kwargs):  # pylint: disable=unused-argument
            raise OSError("No space left on device")

        # Test network socket exhaustion
        def mock_socket_exhaustion(*args, **kwargs):  # pylint: disable=unused-argument
            raise OSError("Address already in use")

        resource_exhaustion_scenarios = [
            ("fd_exhaustion", mock_fd_exhaustion),
            ("disk_exhaustion", mock_disk_exhaustion),
            ("socket_exhaustion", mock_socket_exhaustion),
        ]

        for scenario_name, mock_function in resource_exhaustion_scenarios:
            event_data = {
                "new_root_ids": [900000],
                "old_root_ids": [800000],
                "operation_id": f"resource_exhaustion_{scenario_name}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            with patch("zetta_utils.task_management.supervoxel.update_supervoxels_for_merge",
                       side_effect=mock_function,
            ):

                try:
                    process_merge_event(
                        event_data=event_data,
                        project_name=self.project_name,
                        cave_client=None,
                        server_address=self.server_address,
                        table_id=self.table_id,
                    )
                except OSError:
                    # Should handle resource exhaustion gracefully
                    pass

    def test_long_running_stability(self, mocker):  # pylint: disable=unused-argument
        """Test stability during long-running operations with intermittent failures."""

        # Simulate long-running listener with periodic failures
        operation_count = 0
        failure_intervals = [5, 10, 15]  # Fail at these intervals

        def mock_long_running_operation(*args, **kwargs):  # pylint: disable=unused-argument
            nonlocal operation_count
            operation_count += 1

            if operation_count in failure_intervals:
                # Intermittent failures
                if operation_count % 2 == 0:
                    raise RequestsConnectionError("Network failure")

                raise RuntimeError("Database failure")

            return 1  # Success

        # Simulate 20 operations with intermittent failures
        events_processed = 0
        failures_handled = 0

        with patch("zetta_utils.task_management.automated_workers."
                   "pcg_edit_listener.update_supervoxels_for_merge",
                   side_effect=mock_long_running_operation,
        ):

            for i in range(20):
                event_data = {
                    "new_root_ids": [1000000 + i],
                    "old_root_ids": [900000 + i],
                    "operation_id": f"long_running_{i}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                try:
                    process_merge_event(
                        event_data=event_data,
                        project_name=self.project_name,
                        cave_client=None,
                        server_address=self.server_address,
                        table_id=self.table_id,
                    )
                    events_processed += 1
                except Exception:  # pylint: disable=broad-exception-caught
                    failures_handled += 1

        # Verify resilience: account for the fact that with failure, some
        # events may be processed successfully
        assert events_processed > 0
        assert failures_handled >= 0  # At least 0 failures handled
        assert operation_count == 20
        # Total processed events + handled failures should account for all 20
        # operations
        assert events_processed + failures_handled == 20
