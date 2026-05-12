# pylint: disable=redefined-outer-name,unused-argument
"""Tests for the master-startup preflight check."""
from __future__ import annotations

import pytest
from kubernetes.client.exceptions import ApiException

from zetta_utils.cloud_management.resource_allocation.k8s import preflight
from zetta_utils.cloud_management.resource_allocation.k8s.common import ClusterInfo


@pytest.fixture
def cluster_info():
    return ClusterInfo(name="test-cluster", project="test-proj", region="us-central1")


@pytest.fixture
def mock_get_cluster_data(mocker):
    return mocker.patch.object(
        preflight, "get_cluster_data", return_value=(mocker.MagicMock(), "")
    )


@pytest.fixture
def mock_apis(mocker, mock_get_cluster_data):
    """Patch the three k8s api classes preflight constructs."""
    core = mocker.MagicMock()
    apps = mocker.MagicMock()
    rbac = mocker.MagicMock()
    mocker.patch.object(preflight.k8s_client, "ApiClient", return_value=mocker.MagicMock())
    mocker.patch.object(preflight.k8s_client, "CoreV1Api", return_value=core)
    mocker.patch.object(preflight.k8s_client, "AppsV1Api", return_value=apps)
    mocker.patch.object(preflight.k8s_client, "RbacAuthorizationV1Api", return_value=rbac)
    return core, apps, rbac


def test_succeeds_when_all_resources_present(cluster_info, mock_apis):
    core, apps, rbac = mock_apis
    # All api calls succeed by default (MagicMock returns).
    preflight.verify_cluster_access(cluster_info)
    core.list_namespaced_pod.assert_called_once()
    apps.list_namespaced_deployment.assert_called_once()
    # 4 namespaced + 2 cluster-scoped reads.
    assert rbac.read_namespaced_role.call_count == 2
    assert rbac.read_namespaced_role_binding.call_count == 2
    assert rbac.read_cluster_role.call_count == 1
    assert rbac.read_cluster_role_binding.call_count == 1


def test_raises_on_master_401(cluster_info, mock_apis):
    core, _, _ = mock_apis
    core.list_namespaced_pod.side_effect = ApiException(status=401, reason="Unauthorized")

    with pytest.raises(PermissionError) as ei:
        preflight.verify_cluster_access(cluster_info)

    msg = str(ei.value)
    assert "test-cluster" in msg
    assert "401" in msg
    assert "gcloud auth application-default login" in msg


def test_raises_on_master_403(cluster_info, mock_apis):
    _, apps, _ = mock_apis
    apps.list_namespaced_deployment.side_effect = ApiException(status=403, reason="Forbidden")

    with pytest.raises(PermissionError) as ei:
        preflight.verify_cluster_access(cluster_info)

    assert "403" in str(ei.value)
    assert "roles/container.developer" in str(ei.value)


def test_raises_with_missing_rbac_resources(cluster_info, mock_apis):
    """404 on a Role + a ClusterRoleBinding: both names appear in the error."""
    _, _, rbac = mock_apis
    not_found = ApiException(status=404, reason="Not Found")

    def maybe_404(name, **_kwargs):
        if name in ("pod-metrics-reader", "node-reader-binding"):
            raise not_found

    rbac.read_namespaced_role.side_effect = maybe_404
    rbac.read_namespaced_role_binding.side_effect = maybe_404
    rbac.read_cluster_role.side_effect = maybe_404
    rbac.read_cluster_role_binding.side_effect = maybe_404

    with pytest.raises(PermissionError) as ei:
        preflight.verify_cluster_access(cluster_info)

    msg = str(ei.value)
    assert "Role/pod-metrics-reader" in msg
    assert "ClusterRoleBinding/node-reader-binding" in msg
    assert "kubectl apply -f scripts/gcp/rbac.yml" in msg


def test_raises_on_rbac_read_403(cluster_info, mock_apis):
    """A 403 (not 404) while reading RBAC bubbles up as the auth-error message."""
    _, _, rbac = mock_apis
    rbac.read_namespaced_role.side_effect = ApiException(status=403, reason="Forbidden")

    with pytest.raises(PermissionError) as ei:
        preflight.verify_cluster_access(cluster_info)

    assert "403" in str(ei.value)
    assert "roles/container.developer" in str(ei.value)
