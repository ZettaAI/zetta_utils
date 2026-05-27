# pylint: disable=protected-access,unused-argument,import-outside-toplevel
import pytest


def test_create_namespaced_pod_passes_through(mocker):
    from zetta_utils.cloud_management.resource_allocation.k8s import pod

    core_mock = mocker.patch(
        "zetta_utils.cloud_management.resource_allocation.k8s.pod.k8s_client.CoreV1Api"
    ).return_value
    body = {"metadata": {"name": "p", "namespace": "sessions"}}
    pod.create_namespaced_pod(namespace="sessions", body=body)
    core_mock.create_namespaced_pod.assert_called_once_with(
        namespace="sessions",
        body=body,
    )


def test_worker_template_substitution(master_env):
    """The rendered Pod body must carry the locked invariants."""
    from zetta_utils.session import master

    body = master._render_worker_template(initial_preload="try")
    assert body["metadata"]["name"] == "session-worker-test-uuid-001"
    assert body["metadata"]["namespace"] == "sessions"
    assert body["spec"]["automountServiceAccountToken"] is False
    or_ref = body["metadata"]["ownerReferences"][0]
    assert or_ref["uid"] == "pod-uid-xyz"
    assert or_ref["name"] == "session-master-test-uuid-001-abcd"
    assert or_ref["controller"] is True


def test_delete_swallows_404(mocker):
    from kubernetes.client.exceptions import ApiException

    from zetta_utils.cloud_management.resource_allocation.k8s import pod

    core_mock = mocker.patch(
        "zetta_utils.cloud_management.resource_allocation.k8s.pod.k8s_client.CoreV1Api"
    ).return_value
    core_mock.delete_namespaced_pod.side_effect = ApiException(status=404)
    # Must NOT raise.
    pod.delete_namespaced_pod(name="missing", namespace="sessions")


def test_delete_propagates_500(mocker):
    from kubernetes.client.exceptions import ApiException

    from zetta_utils.cloud_management.resource_allocation.k8s import pod

    core_mock = mocker.patch(
        "zetta_utils.cloud_management.resource_allocation.k8s.pod.k8s_client.CoreV1Api"
    ).return_value
    core_mock.delete_namespaced_pod.side_effect = ApiException(status=500)
    with pytest.raises(ApiException):
        pod.delete_namespaced_pod(name="p", namespace="sessions")
