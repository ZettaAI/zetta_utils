"""Tests for automated_workers __init__.py module"""

# pylint: disable=unused-argument,redefined-outer-name

from zetta_utils.task_management import automated_workers


def test_automated_workers_imports():
    """Test that automated workers can be imported"""

    # Verify the submodules are accessible
    assert hasattr(automated_workers, "segmentation_auto_verifier")
    assert hasattr(automated_workers, "segmentation_stats_updater")
