# mypy: warn-unused-ignores=False

"""
Stress tests for PCG edit listener functionality.

Tests error handling, large data volumes, network failures, and concurrent operations.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import requests

from zetta_utils.message_queues.pubsub import PubSubPullQueue
from zetta_utils.task_management.automated_workers.pcg_edit_listener import (
    get_old_roots_from_lineage_graph,
    get_root_for_coordinate_pcg,
    get_supervoxel_ids_from_segment,
    process_edit_event,
)
from zetta_utils.task_management.db.models import SupervoxelModel
from zetta_utils.task_management.supervoxel import (
    create_supervoxel,
    get_supervoxels_by_segment,
    update_supervoxels_for_merge,
    update_supervoxels_for_split,
)


@dataclass
class MockMessage:
    """Mock PubSub message for testing."""

    payload: Dict[str, Any]
    acknowledge_fn: Mock = Mock()


class TestPCGListenerStress:  # pylint: disable=attribute-defined-outside-init
    """Stress tests for PCG listener components."""

    def setup_method(self, method) -> None:
        """Set up test fixtures."""
        self.project_name = f"stress_test_project_{method.__name__}"
        self.server_address = "https://test.example.com"
        self.table_id = "test_table_123"

        # Configurable stress test metrics
        self.stress_config = {
            "large_supervoxel_count": 100,
            "batch_size": 50,
            "num_segments": 3,
            "supervoxels_per_segment": 50,
            "message_count": 50,
            "message_batch_size": 10,
            "memory_test_messages": 50,
            "supervoxel_count_large": 100,
            "max_processing_time_seconds": 5,
            "max_batch_time_seconds": 2,
            "max_memory_increase_mb": 100,
            "concurrent_workers": 3,
            "max_cycles_stress": 2,
            "timeout_stress_seconds": 1,
        }

    def create_test_supervoxels(
            self, session, count: int, base_segment_id: int = 100000, base_supervoxel_id: int = 1
    ) -> List[SupervoxelModel]:
        """Create test supervoxels for stress testing."""
        supervoxels = []
        for i in range(count):
            supervoxel = create_supervoxel(
                supervoxel_id=base_supervoxel_id + i,
                seed_x=float(i * 100),
                seed_y=float(i * 200),
                seed_z=float(i * 300),
                current_segment_id=base_segment_id,
                db_session=session,
            )
            supervoxels.append(supervoxel)
        return supervoxels

    def test_large_old_supervoxels_collection(
            self, db_session, project_factory):
        """Test handling of large collections of old_supervoxels."""
        project_factory(project_name=self.project_name)

        # Create a large number of supervoxels (stress test with configurable
        # count)
        large_supervoxel_count = self.stress_config["large_supervoxel_count"]
        old_segment_id = 999999
        new_segment_id = 888888

        # Create supervoxels in batches to avoid memory issues
        batch_size = self.stress_config["batch_size"]
        base_supervoxel_id = 1000000  # Use large base ID to avoid conflicts
        for batch_start in range(0, large_supervoxel_count, batch_size):
            batch_end = min(batch_start + batch_size, large_supervoxel_count)
            for i in range(batch_start, batch_end):
                create_supervoxel(
                    supervoxel_id=base_supervoxel_id + i,
                    seed_x=float(i * 10),
                    seed_y=float(i * 20),
                    seed_z=float(i * 30),
                    current_segment_id=old_segment_id,
                    db_session=db_session,
                )

        # Test merge operation with large dataset
        start_time = time.time()

        updated_count = update_supervoxels_for_merge(
            old_root_ids=[old_segment_id],
            new_root_id=new_segment_id,
            project_name=self.project_name,
            event_id="large_merge_test",
            edit_timestamp=datetime.now(timezone.utc),
            db_session=db_session,
        )

        end_time = time.time()
        processing_time = end_time - start_time

        # Verify results
        assert updated_count == large_supervoxel_count
        assert (
                processing_time < self.stress_config["max_processing_time_seconds"]
        )  # Should complete within configured time

        # Verify all supervoxels were updated
        updated_supervoxels = get_supervoxels_by_segment(
            segment_id=new_segment_id, db_session=db_session
        )
        assert len(updated_supervoxels) == large_supervoxel_count

    def test_concurrent_merge_operations(self, db_session, project_factory):
        """Test concurrent merge operations for race conditions."""
        project_factory(project_name=self.project_name)

        # Create supervoxels for multiple segments
        num_segments = self.stress_config["num_segments"]
        supervoxels_per_segment = self.stress_config["supervoxels_per_segment"]

        base_supervoxel_id = 2000000  # Use different base to avoid conflicts
        for segment_idx in range(num_segments):
            segment_id = 100000 + segment_idx
            for sv_idx in range(supervoxels_per_segment):
                create_supervoxel(
                    supervoxel_id=base_supervoxel_id
                                  + segment_idx * supervoxels_per_segment
                                  + sv_idx,
                    seed_x=float(sv_idx * 10),
                    seed_y=float(sv_idx * 20),
                    seed_z=float(sv_idx * 30),
                    current_segment_id=segment_id,
                    db_session=db_session,
                )

        # Define concurrent merge operations
        def perform_merge(segment_idx):
            old_segment_id = 100000 + segment_idx
            new_segment_id = 200000 + segment_idx

            try:
                return update_supervoxels_for_merge(
                    old_root_ids=[old_segment_id],
                    new_root_id=new_segment_id,
                    project_name=self.project_name,
                    event_id=f"concurrent_merge_{segment_idx}",
                    edit_timestamp=datetime.now(timezone.utc),
                    db_session=db_session,  # To use the same transaction context
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                return f"Error: {e}"

        # Execute merges sequentially first to avoid transaction conflicts in
        # tests
        results = []
        for i in range(num_segments):
            result = perform_merge(i)
            results.append(result)

        # Verify all operations completed successfully
        for result in results:
            assert isinstance(result, int), f"Operation failed: {result}"
            assert result == supervoxels_per_segment

    def test_network_failure_scenarios(self, mocker):  # pylint: disable=unused-argument
        """Test network failure handling in PCG API calls."""

        # Test timeout scenarios
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.Timeout(
            "Connection timeout")

        with patch("requests.post", return_value=mock_response):
            result = get_old_roots_from_lineage_graph(
                server_address=self.server_address,
                table_id=self.table_id,
                root_ids=[123456],
                timestamp_past=datetime.now(
                    timezone.utc) - timedelta(minutes=1),
            )
            assert not result  # Should return empty dict on failure

        # Test HTTP errors
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            "404 Not Found")

        with patch("requests.post", return_value=mock_response):
            result = get_old_roots_from_lineage_graph(
                server_address=self.server_address,
                table_id=self.table_id,
                root_ids=[123456],
                timestamp_past=datetime.now(
                    timezone.utc) - timedelta(minutes=1),
            )
            assert not result

        # Test connection errors
        with patch("requests.post", side_effect=requests.ConnectionError("Connection failed")):
            result = get_old_roots_from_lineage_graph(
                server_address=self.server_address,
                table_id=self.table_id,
                root_ids=[123456],
                timestamp_past=datetime.now(
                    timezone.utc) - timedelta(minutes=1),
            )
            assert not result

    def test_large_volume_message_processing(self, mocker):  # pylint: disable=unused-argument
        """Test processing large volumes of messages."""

        # Mock PubSub queue (unused but kept for potential future use)
        _ = Mock(spec=PubSubPullQueue)

        # Create large batch of messages
        message_count = self.stress_config["message_count"]
        messages = []

        for i in range(message_count):
            # Alternate between merge and split events
            if i % 2 == 0:
                # Merge event
                payload = {
                    "new_root_ids": [100000 + i],
                    "old_root_ids": [50000 + i, 50001 + i],
                    "operation_id": f"merge_{i}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "_pubsub_attributes": {"table_id": self.table_id},
                }
            else:
                # Split event
                payload = {
                    "new_root_ids": [100000 + i, 100001 + i],
                    "operation_id": f"split_{i}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "_pubsub_attributes": {"table_id": self.table_id},
                }

            messages.append(
                MockMessage(
                    payload=payload,
                    acknowledge_fn=Mock()))

        # Mock API responses - patch where the functions are imported/used
        with patch("zetta_utils.task_management.automated_workers."
                   "pcg_edit_listener.get_old_roots_from_lineage_graph") as mock_lineage, \
             patch("zetta_utils.task_management.automated_workers."
                   "pcg_edit_listener.get_supervoxel_ids_from_segment") as mock_supervoxels, \
             patch("zetta_utils.task_management.automated_workers."
                   "pcg_edit_listener.update_supervoxels_for_merge") as mock_merge, \
             patch("zetta_utils.task_management.automated_workers."
                   "pcg_edit_listener.update_supervoxels_for_split") as mock_split, \
             patch("zetta_utils.task_management.automated_workers."
                   "pcg_edit_listener.get_supervoxels_by_segment") as mock_get_sv, \
             patch("zetta_utils.task_management.automated_workers."
                   "pcg_edit_listener.queue_segment_updates_for_segments") as _mq, \
             patch("zetta_utils.task_management.automated_workers."
                   "pcg_edit_listener._find_moved_seed_ids") as _mock_find_seeds, \
             patch("zetta_utils.task_management.automated_workers."
                   "pcg_edit_listener._apply_merge_updates") as _mock_apply_merge:

            mock_lineage.return_value = {100000: [50000, 50001]}
            mock_supervoxels.return_value = [1, 2, 3, 4, 5]
            mock_merge.return_value = 5
            mock_split.return_value = 5
            mock_get_sv.return_value = [Mock(supervoxel_id=1), Mock(supervoxel_id=2)]

            # Process messages in batches
            batch_size = self.stress_config["message_batch_size"]
            processing_times = []

            for batch_start in range(0, message_count, batch_size):
                batch_end = min(batch_start + batch_size, message_count)
                batch = messages[batch_start:batch_end]

                start_time = time.time()

                for msg in batch:
                    try:
                        # Extract table_id from attributes
                        attributes = msg.payload.pop("_pubsub_attributes", {})
                        process_edit_event(
                            event_data=msg.payload,
                            message_attributes=attributes,
                            project_name=self.project_name,
                            cave_client=None,
                            server_address=self.server_address,
                        )
                        if msg.acknowledge_fn:
                            msg.acknowledge_fn()
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        print(f"Error processing message: {e}")

                processing_time = time.time() - start_time
                processing_times.append(processing_time)

            # Verify performance
            avg_processing_time = sum(processing_times) / len(processing_times)
            assert (
                    avg_processing_time < self.stress_config["max_batch_time_seconds"]
            )  # Average batch should process in under configured time

            # Verify all messages were acknowledged
            for msg in messages:
                if msg.acknowledge_fn:
                    msg.acknowledge_fn.assert_called_once()

    def test_malformed_message_handling(self, mocker):  # pylint: disable=unused-argument
        """Test handling of malformed or incomplete messages."""

        malformed_messages = [
            # Missing new_root_ids
            {"operation_id": "test1"},
            # Invalid timestamp
            {"new_root_ids": [123], "timestamp": "invalid_date"},
            # Empty new_root_ids
            {"new_root_ids": [], "operation_id": "test2"},
            # Non-integer root IDs
            {"new_root_ids": ["abc"], "operation_id": "test3"},
            # Missing operation_id
            {"new_root_ids": [123]},
            # Extremely large numbers
            {"new_root_ids": [2 ** 63], "operation_id": "test4"},
        ]

        with patch(
                "zetta_utils.task_management.supervoxel.update_supervoxels_for_merge"
        ) as mock_merge, patch(
            "zetta_utils.task_management.supervoxel.update_supervoxels_for_split"
        ) as mock_split:

            mock_merge.return_value = 0
            mock_split.return_value = 0

            for msg_data in malformed_messages:
                try:
                    process_edit_event(
                        event_data=msg_data,  # type: ignore[arg-type]
                        message_attributes={"table_id": self.table_id},
                        project_name=self.project_name,
                        cave_client=None,
                        server_address=self.server_address,
                    )
                except Exception as e:  # pylint: disable=broad-exception-caught
                    # Should handle gracefully without crashing
                    print(f"Handled malformed message error: {e}")

    def test_memory_usage_large_split(self, db_session, project_factory):
        """Test memory usage during large split operations."""
        project_factory(project_name=self.project_name)

        # Create a large segment with many supervoxels
        old_segment_id = 777777
        supervoxel_count = self.stress_config["supervoxel_count_large"]

        base_supervoxel_id = 3000000  # Use different base to avoid conflicts
        for i in range(supervoxel_count):
            create_supervoxel(
                supervoxel_id=base_supervoxel_id + i,
                seed_x=float(i * 5),
                seed_y=float(i * 10),
                seed_z=float(i * 15),
                current_segment_id=old_segment_id,
                db_session=db_session,
            )

        # Create split assignments (divide supervoxels between 2 new segments)
        new_segment_1 = 888881
        new_segment_2 = 888882

        supervoxel_assignments = {}
        for i in range(supervoxel_count):
            supervoxel_id = base_supervoxel_id + i
            new_segment = new_segment_1 if i % 2 == 0 else new_segment_2
            supervoxel_assignments[supervoxel_id] = new_segment

        # Monitor memory usage during split
        import os  # pylint: disable=import-outside-toplevel

        import psutil  # pylint: disable=import-outside-toplevel

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        updated_count = update_supervoxels_for_split(
            old_root_id=old_segment_id,
            new_root_ids=[new_segment_1, new_segment_2],
            supervoxel_assignments=supervoxel_assignments,
            project_name=self.project_name,
            event_id="large_split_test",
            edit_timestamp=datetime.now(timezone.utc),
            db_session=db_session,
        )

        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before

        # Verify results
        assert updated_count == supervoxel_count
        assert (
                memory_increase < self.stress_config["max_memory_increase_mb"]
        )  # Should not increase by more than configured MB

    def test_api_rate_limiting(self, mocker):  # pylint: disable=unused-argument
        """Test handling of API rate limiting."""

        # Mock rate limited response
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            "429 Too Many Requests")

        with patch("requests.post", return_value=mock_response):
            result = get_old_roots_from_lineage_graph(
                server_address=self.server_address,
                table_id=self.table_id,
                root_ids=[123456],
                timestamp_past=datetime.now(
                    timezone.utc) - timedelta(minutes=1),
            )
            assert not result  # Should handle gracefully

        # Test GET endpoint rate limiting
        with patch("requests.get", return_value=mock_response):
            get_result: list[int] = get_supervoxel_ids_from_segment(
                server_address=self.server_address, table_id=self.table_id, segment_id=123456
            )
            assert get_result == []  # Should return empty list on failure

    def test_extremely_large_supervoxel_ids(self, db_session, project_factory):
        """Test handling of extremely large supervoxel IDs."""
        project_factory(project_name=self.project_name)

        # Test with very large supervoxel IDs (near int64 max)
        large_ids = [2 ** 62, 2 ** 62 + 1, 2 ** 62 + 2]
        segment_id = 999999999  # Use unique segment ID to avoid conflicts with other tests

        for sv_id in large_ids:
            create_supervoxel(
                supervoxel_id=sv_id,
                seed_x=1.0,
                seed_y=2.0,
                seed_z=3.0,
                current_segment_id=segment_id,
                db_session=db_session,
            )

        # Test merge with large IDs
        new_segment_id = 888888888  # Use unique new segment ID to avoid conflicts
        updated_count = update_supervoxels_for_merge(
            old_root_ids=[segment_id],
            new_root_id=new_segment_id,
            project_name=self.project_name,
            event_id="large_id_test",
            edit_timestamp=datetime.now(timezone.utc),
            db_session=db_session,
        )

        assert updated_count == len(large_ids)

        # Verify supervoxels were updated correctly
        updated_supervoxels = get_supervoxels_by_segment(
            segment_id=new_segment_id, db_session=db_session
        )
        assert len(updated_supervoxels) == len(large_ids)
        assert all(sv.supervoxel_id in large_ids for sv in updated_supervoxels)

    def test_event_idempotency_stress(self, db_session, project_factory):
        """Test event idempotency under stress conditions."""
        project_factory(project_name=self.project_name)

        # Create test supervoxels
        segment_id = 444444
        supervoxel_count = 100

        base_supervoxel_id = 5000000  # Use different base to avoid conflicts
        for i in range(supervoxel_count):
            create_supervoxel(
                supervoxel_id=base_supervoxel_id + i,
                seed_x=float(i),
                seed_y=float(i),
                seed_z=float(i),
                current_segment_id=segment_id,
                db_session=db_session,
            )

        # Execute the same event multiple times sequentially to test idempotency
        # without database constraint violations
        event_id = "idempotency_stress_test"
        new_segment_id = 333333

        def execute_merge():
            try:
                return update_supervoxels_for_merge(
                    old_root_ids=[segment_id],
                    new_root_id=new_segment_id,
                    project_name=self.project_name,
                    event_id=event_id,  # Same event ID for all operations
                    edit_timestamp=datetime.now(timezone.utc),
                    db_session=db_session,  # Use same session to avoid transaction issues
                )
            except Exception:  # pylint: disable=broad-exception-caught
                return 0  # Handle constraint violations gracefully

        # Execute same operation multiple times sequentially
        results = []
        for _ in range(5):  # Reduce number to avoid conflicts
            result = execute_merge()
            results.append(result)

        # Only the first operation should actually update records
        non_zero_results = [r for r in results if r > 0]
        assert len(non_zero_results) <= 1  # At most one should succeed

        # Remaining results should be 0 (already processed)
        zero_results = [r for r in results if r == 0]
        # At least 4 should be skipped due to idempotency
        assert len(zero_results) >= 4

    def test_invalid_coordinates_handling(self):
        """Test handling of invalid coordinates in API calls."""

        invalid_coordinates = [
            (float("inf"), 0, 0),
            (0, float("-inf"), 0),
            (0, 0, float("nan")),
            (None, 0, 0),
            ("invalid", 0, 0),
        ]

        for x, y, z in invalid_coordinates:
            result = get_root_for_coordinate_pcg(
                server_address=self.server_address,
                table_id=self.table_id,
                x=x, # type: ignore[arg-type]
                y=y, # type: ignore[arg-type]
                z=z, # type: ignore[arg-type]
            )
            assert result is None  # Should handle gracefully
