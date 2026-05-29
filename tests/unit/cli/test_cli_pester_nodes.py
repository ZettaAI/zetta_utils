# pylint: disable=redefined-outer-name,unused-argument
import pytest
from click.testing import CliRunner

from zetta_utils.cli.run.cli_pester_nodes import pester_nodes_cli


def _make_deployment(mocker, name, spec_replicas, ready_replicas):
    dep = mocker.MagicMock()
    dep.metadata.name = name
    dep.metadata.labels = {"worker_group": "io"}
    dep.spec.replicas = spec_replicas
    dep.status.ready_replicas = ready_replicas
    return dep


def _make_pod(mocker, node_name):
    pod = mocker.MagicMock()
    pod.spec.node_name = node_name
    return pod


def _make_node(mocker, name, pool_name):
    node = mocker.MagicMock()
    node.metadata.name = name
    node.metadata.labels = {"cloud.google.com/gke-nodepool": pool_name}
    return node


def _make_pool(mocker, name, current, max_size, locations=None, total_max=0):
    pool = mocker.MagicMock()
    pool.name = name
    pool.initial_node_count = current
    pool.autoscaling.max_node_count = max_size
    pool.autoscaling.total_max_node_count = total_max
    pool.locations = locations or ["us-central1-a"]
    return pool


@pytest.fixture
def mock_load_kube_config(mocker):
    return mocker.patch("zetta_utils.cli.run.cli_pester_nodes.config.load_kube_config")


@pytest.fixture
def mock_core_api(mocker, mock_load_kube_config):
    core_api_class = mocker.patch("zetta_utils.cli.run.cli_pester_nodes.k8s_client.CoreV1Api")
    core_api = core_api_class.return_value
    core_api.list_namespaced_pod.return_value.items = []
    return core_api


@pytest.fixture
def mock_apps_api(mocker, mock_load_kube_config):
    apps_api_class = mocker.patch("zetta_utils.cli.run.cli_pester_nodes.k8s_client.AppsV1Api")
    apps_api = apps_api_class.return_value
    apps_api.list_namespaced_deployment.return_value.items = []
    return apps_api


@pytest.fixture
def mock_gke(mocker):
    return mocker.patch("zetta_utils.cli.run.cli_pester_nodes.gke")


@pytest.fixture
def mock_event_wait(mocker):
    """Loop's ``stop_event.wait(interval_sec)`` returns False instantly so tests don't sleep."""
    return mocker.patch("threading.Event.wait", return_value=False)


def _run(args=()):
    return CliRunner().invoke(
        pester_nodes_cli,
        [
            "pester-nodes",
            "test-run",
            "-g",
            "io",
            "-n",
            "5",
            "--cluster-name",
            "c",
            "--cluster-region",
            "us-central1",
            "--cluster-project",
            "proj",
            *args,
        ],
    )


def test_pester_exits_when_pending_already_zero(
    mocker, mock_core_api, mock_apps_api, mock_gke, mock_event_wait
):
    mock_apps_api.list_namespaced_deployment.return_value.items = [
        _make_deployment(mocker, "run-test-run-io", spec_replicas=10, ready_replicas=10),
    ]
    mock_apps_api.read_namespaced_deployment.return_value = _make_deployment(
        mocker, "run-test-run-io", spec_replicas=10, ready_replicas=10
    )
    mock_core_api.list_namespaced_pod.return_value.items = [_make_pod(mocker, "node-a")]
    mock_core_api.read_node.return_value = _make_node(mocker, "node-a", "pool-1")
    mock_gke.list_node_pools.return_value = [_make_pool(mocker, "pool-1", current=3, max_size=20)]

    result = _run()

    assert result.exit_code == 0, result.output
    assert "all 10 replicas ready" in result.output
    mock_gke.resize_node_pool.assert_not_called()


def test_pester_resizes_to_current_plus_additional(
    mocker, mock_core_api, mock_apps_api, mock_gke, mock_event_wait
):
    mock_apps_api.list_namespaced_deployment.return_value.items = [
        _make_deployment(mocker, "run-test-run-io", spec_replicas=10, ready_replicas=2),
    ]
    mock_apps_api.read_namespaced_deployment.side_effect = [
        _make_deployment(mocker, "run-test-run-io", spec_replicas=10, ready_replicas=2),
        _make_deployment(mocker, "run-test-run-io", spec_replicas=10, ready_replicas=10),
    ]
    mock_core_api.list_namespaced_pod.return_value.items = [_make_pod(mocker, "node-a")]
    mock_core_api.read_node.return_value = _make_node(mocker, "node-a", "pool-1")
    mock_gke.list_node_pools.return_value = [_make_pool(mocker, "pool-1", current=3, max_size=20)]

    result = _run()

    assert result.exit_code == 0, result.output
    mock_gke.resize_node_pool.assert_called_once_with("proj", "us-central1", "c", "pool-1", 8)


def test_pester_caps_target_at_max_node_count(
    mocker, mock_core_api, mock_apps_api, mock_gke, mock_event_wait
):
    mock_apps_api.list_namespaced_deployment.return_value.items = [
        _make_deployment(mocker, "run-test-run-io", spec_replicas=10, ready_replicas=2),
    ]
    mock_apps_api.read_namespaced_deployment.side_effect = [
        _make_deployment(mocker, "run-test-run-io", spec_replicas=10, ready_replicas=2),
        _make_deployment(mocker, "run-test-run-io", spec_replicas=10, ready_replicas=10),
    ]
    mock_core_api.list_namespaced_pod.return_value.items = [_make_pod(mocker, "node-a")]
    mock_core_api.read_node.return_value = _make_node(mocker, "node-a", "pool-1")
    mock_gke.list_node_pools.return_value = [_make_pool(mocker, "pool-1", current=18, max_size=20)]

    result = _run()

    assert result.exit_code == 0, result.output
    mock_gke.resize_node_pool.assert_called_once_with("proj", "us-central1", "c", "pool-1", 20)


def test_pester_skips_pool_already_at_max(
    mocker, mock_core_api, mock_apps_api, mock_gke, mock_event_wait
):
    mock_apps_api.list_namespaced_deployment.return_value.items = [
        _make_deployment(mocker, "run-test-run-io", spec_replicas=10, ready_replicas=2),
    ]
    mock_core_api.list_namespaced_pod.return_value.items = [_make_pod(mocker, "node-a")]
    mock_core_api.read_node.return_value = _make_node(mocker, "node-a", "pool-1")
    mock_gke.list_node_pools.return_value = [_make_pool(mocker, "pool-1", current=20, max_size=20)]

    result = _run()

    assert result.exit_code != 0
    assert "No pools left to pester" in result.output
    mock_gke.resize_node_pool.assert_not_called()


def test_pester_discovers_multiple_pools_from_pod_nodes(
    mocker, mock_core_api, mock_apps_api, mock_gke, mock_event_wait
):
    mock_apps_api.list_namespaced_deployment.return_value.items = [
        _make_deployment(mocker, "run-test-run-io", spec_replicas=10, ready_replicas=2),
    ]
    mock_apps_api.read_namespaced_deployment.side_effect = [
        _make_deployment(mocker, "run-test-run-io", spec_replicas=10, ready_replicas=2),
        _make_deployment(mocker, "run-test-run-io", spec_replicas=10, ready_replicas=10),
    ]
    mock_core_api.list_namespaced_pod.return_value.items = [
        _make_pod(mocker, "node-a"),
        _make_pod(mocker, "node-b"),
    ]
    nodes = {
        "node-a": _make_node(mocker, "node-a", "pool-1"),
        "node-b": _make_node(mocker, "node-b", "pool-2"),
    }
    mock_core_api.read_node.side_effect = lambda name: nodes[name]
    mock_gke.list_node_pools.return_value = [
        _make_pool(mocker, "pool-1", current=3, max_size=20),
        _make_pool(mocker, "pool-2", current=4, max_size=20),
    ]

    result = _run()

    assert result.exit_code == 0, result.output
    assert mock_gke.resize_node_pool.call_count == 2
    targets = {c.args[3]: c.args[4] for c in mock_gke.resize_node_pool.call_args_list}
    assert targets == {"pool-1": 8, "pool-2": 9}


def test_pester_errors_when_no_scheduled_pods(
    mocker, mock_core_api, mock_apps_api, mock_gke, mock_event_wait
):
    mock_apps_api.list_namespaced_deployment.return_value.items = [
        _make_deployment(mocker, "run-test-run-io", spec_replicas=10, ready_replicas=0),
    ]
    mock_core_api.list_namespaced_pod.return_value.items = [_make_pod(mocker, None)]

    result = _run()

    assert result.exit_code != 0
    assert "No scheduled pods found" in result.output
    mock_gke.list_node_pools.assert_not_called()


def test_pester_errors_when_no_deployment(mock_core_api, mock_apps_api, mock_gke, mock_event_wait):
    mock_apps_api.list_namespaced_deployment.return_value.items = []

    result = _run()

    assert result.exit_code != 0
    assert "No deployment found" in result.output


def test_pester_explicit_pool_skips_pod_discovery(
    mocker, mock_core_api, mock_apps_api, mock_gke, mock_event_wait
):
    mock_apps_api.list_namespaced_deployment.return_value.items = [
        _make_deployment(mocker, "run-test-run-io", spec_replicas=10, ready_replicas=2),
    ]
    mock_apps_api.read_namespaced_deployment.side_effect = [
        _make_deployment(mocker, "run-test-run-io", spec_replicas=10, ready_replicas=2),
        _make_deployment(mocker, "run-test-run-io", spec_replicas=10, ready_replicas=10),
    ]
    # No scheduled pods — pod-discovery would error, but --pool bypasses it.
    mock_core_api.list_namespaced_pod.return_value.items = [_make_pod(mocker, None)]
    mock_gke.list_node_pools.return_value = [_make_pool(mocker, "pool-x", current=3, max_size=20)]

    result = _run(args=["--pool", "pool-x"])

    assert result.exit_code == 0, result.output
    assert "Using explicit pools: ['pool-x']" in result.output
    mock_core_api.read_node.assert_not_called()
    mock_gke.resize_node_pool.assert_called_once_with("proj", "us-central1", "c", "pool-x", 8)


def test_pester_uses_total_max_when_per_zone_max_unset(
    mocker, mock_core_api, mock_apps_api, mock_gke, mock_event_wait
):
    """Regional pools set total_max_node_count instead of per-zone max_node_count.

    In that case `max_node_count` defaults to 0 in proto; without the
    fallback we'd incorrectly skip the pool as "already at max (0)".
    """
    mock_apps_api.list_namespaced_deployment.return_value.items = [
        _make_deployment(mocker, "run-test-run-io", spec_replicas=10, ready_replicas=2),
    ]
    mock_apps_api.read_namespaced_deployment.side_effect = [
        _make_deployment(mocker, "run-test-run-io", spec_replicas=10, ready_replicas=2),
        _make_deployment(mocker, "run-test-run-io", spec_replicas=10, ready_replicas=10),
    ]
    mock_core_api.list_namespaced_pod.return_value.items = [_make_pod(mocker, None)]
    mock_gke.list_node_pools.return_value = [
        _make_pool(mocker, "pool-r", current=1, max_size=0, total_max=300),
    ]

    result = _run(args=["--pool", "pool-r"])

    assert result.exit_code == 0, result.output
    mock_gke.resize_node_pool.assert_called_once_with("proj", "us-central1", "c", "pool-r", 6)


def test_pester_worker_group_optional(
    mocker, mock_core_api, mock_apps_api, mock_gke, mock_event_wait
):
    mock_apps_api.list_namespaced_deployment.return_value.items = [
        _make_deployment(mocker, "run-test-run", spec_replicas=10, ready_replicas=2),
    ]
    mock_apps_api.read_namespaced_deployment.side_effect = [
        _make_deployment(mocker, "run-test-run", spec_replicas=10, ready_replicas=2),
        _make_deployment(mocker, "run-test-run", spec_replicas=10, ready_replicas=10),
    ]
    mock_core_api.list_namespaced_pod.return_value.items = [_make_pod(mocker, "node-a")]
    mock_core_api.read_node.return_value = _make_node(mocker, "node-a", "pool-1")
    mock_gke.list_node_pools.return_value = [_make_pool(mocker, "pool-1", current=3, max_size=20)]

    result = CliRunner().invoke(
        pester_nodes_cli,
        [
            "pester-nodes",
            "test-run",
            "-n",
            "5",
            "--cluster-name",
            "c",
            "--cluster-region",
            "us-central1",
            "--cluster-project",
            "proj",
        ],
    )

    assert result.exit_code == 0, result.output
    # Selector must be run_id-only when worker_group omitted.
    list_kwargs = mock_apps_api.list_namespaced_deployment.call_args.kwargs
    assert list_kwargs["label_selector"] == "run_id=test-run"
    mock_gke.resize_node_pool.assert_called_once_with("proj", "us-central1", "c", "pool-1", 8)
