# pylint: disable=redefined-outer-name,unused-argument,protected-access
"""Tests for the autoscaler nudger loop's pester-until-success behavior."""
from __future__ import annotations

import threading

import pytest

from zetta_utils.cloud_management.resource_allocation.k8s import autoscaler
from zetta_utils.cloud_management.resource_allocation.k8s.common import ClusterInfo


@pytest.fixture
def cluster_info():
    return ClusterInfo(name="test-cluster", project="test-proj", region="us-central1")


@pytest.fixture
def deployment_with(mocker):
    """Build a fake V1Deployment-like object with the given replica counts."""

    def _make(spec_replicas: int, ready_replicas: int):
        dep = mocker.MagicMock()
        dep.spec.replicas = spec_replicas
        dep.status.ready_replicas = ready_replicas
        return dep

    return _make


class TestRunNudgeLoop:
    def test_keeps_nudging_while_pending_then_stops_when_zero(
        self, mocker, cluster_info, deployment_with
    ):
        """Three iterations: first two have pending pods → nudge. Third has
        ``ready == spec`` → state cleared, no further nudge."""
        state = autoscaler._GroupNudgeState(attempted_pool="pool-a", attempted_target_per_zone=5)
        stop_event = threading.Event()

        deployments = [
            deployment_with(10, 2),  # pending=8
            deployment_with(10, 5),  # pending=5
            deployment_with(10, 10),  # pending=0 → state should clear
        ]
        apps_api = mocker.MagicMock()
        apps_api.read_namespaced_deployment.side_effect = deployments
        mocker.patch.object(autoscaler, "_get_apps_v1_api", return_value=apps_api)

        nudge_calls = []

        def fake_nudge(pool, target, _ci, _gap):
            nudge_calls.append((pool, target))
            if len(nudge_calls) == 3:
                stop_event.set()

        mocker.patch.object(autoscaler, "_nudge_pool", side_effect=fake_nudge)

        wait_calls = {"n": 0}

        def fake_wait(_t):
            wait_calls["n"] += 1
            if wait_calls["n"] > len(deployments):
                return True
            return stop_event.is_set()

        mocker.patch.object(stop_event, "wait", side_effect=fake_wait)

        autoscaler._run_nudge_loop(
            group_state=state,
            deployment_name="run-test-io",
            namespace="default",
            cluster_info=cluster_info,
            stop_event=stop_event,
            nudge_interval_sec=180,
            nudge_min_gap_sec=60,
        )

        # Two nudges fired (third iteration cleared state instead).
        assert nudge_calls == [("pool-a", 5), ("pool-a", 5)]
        # State cleared after pending hit 0.
        assert state.attempted_pool is None
        assert state.attempted_target_per_zone is None

    def test_skips_iteration_when_state_unset(self, mocker, cluster_info):
        state = autoscaler._GroupNudgeState()
        stop_event = threading.Event()

        apps_api = mocker.MagicMock()
        mocker.patch.object(autoscaler, "_get_apps_v1_api", return_value=apps_api)
        nudge = mocker.patch.object(autoscaler, "_nudge_pool")

        wait_calls = {"n": 0}

        def fake_wait(_t):
            wait_calls["n"] += 1
            return wait_calls["n"] >= 2

        mocker.patch.object(stop_event, "wait", side_effect=fake_wait)

        autoscaler._run_nudge_loop(
            group_state=state,
            deployment_name="run-test-io",
            namespace="default",
            cluster_info=cluster_info,
            stop_event=stop_event,
            nudge_interval_sec=180,
            nudge_min_gap_sec=60,
        )

        nudge.assert_not_called()
        apps_api.read_namespaced_deployment.assert_not_called()

    def test_keeps_state_when_nudge_call_raises(self, mocker, cluster_info, deployment_with):
        """An inner ``_nudge_pool`` exception is caught and logged, but the
        state must persist so the next cycle retries."""
        state = autoscaler._GroupNudgeState(attempted_pool="pool-a", attempted_target_per_zone=5)
        stop_event = threading.Event()

        apps_api = mocker.MagicMock()
        apps_api.read_namespaced_deployment.return_value = deployment_with(10, 2)
        mocker.patch.object(autoscaler, "_get_apps_v1_api", return_value=apps_api)

        mocker.patch.object(autoscaler, "_nudge_pool", side_effect=RuntimeError("boom"))

        wait_calls = {"n": 0}

        def fake_wait(_t):
            wait_calls["n"] += 1
            return wait_calls["n"] >= 2

        mocker.patch.object(stop_event, "wait", side_effect=fake_wait)

        autoscaler._run_nudge_loop(
            group_state=state,
            deployment_name="run-test-io",
            namespace="default",
            cluster_info=cluster_info,
            stop_event=stop_event,
            nudge_interval_sec=180,
            nudge_min_gap_sec=60,
        )

        assert state.attempted_pool == "pool-a"
        assert state.attempted_target_per_zone == 5
