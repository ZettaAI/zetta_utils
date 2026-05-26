import asyncio

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if asyncio.iscoroutinefunction(getattr(item, "function", None)):
            item.add_marker(pytest.mark.anyio)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def mock_k8s_apis(mocker):
    """Returns a CoreV1Api mock with sane default returns.

    Master only uses CoreV1Api; BatchV1Api (Jobs) is the manager's concern and
    is intentionally absent.
    """
    core = mocker.patch(
        "zetta_utils.session.master.k8s_client.CoreV1Api", autospec=True
    ).return_value
    return core


@pytest.fixture
def master_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SESSION_ID", "test-uuid-001")
    monkeypatch.setenv("POD_NAME", "session-master-test-uuid-001-abcd")
    monkeypatch.setenv("POD_UID", "pod-uid-xyz")
    monkeypatch.setenv("WORKLOAD_NAMESPACE", "sessions")
    monkeypatch.setenv("SESSIONS_IMAGE_TAG", "web_api_gpu:test")
    worker_yaml = tmp_path / "session-worker-template.yaml"
    worker_yaml.write_text(
        """
apiVersion: v1
kind: Pod
metadata:
  name: session-worker-${SESSION_ID}
  namespace: sessions
  ownerReferences:
  - apiVersion: v1
    kind: Pod
    name: ${MASTER_POD_NAME}
    uid: ${MASTER_POD_UID}
    controller: true
spec:
  automountServiceAccountToken: false
  containers:
  - name: session-worker
    image: ${SESSIONS_IMAGE_TAG}
"""
    )
    monkeypatch.setenv("SESSION_WORKER_TEMPLATE_PATH", str(worker_yaml))

    worker_svc_yaml = tmp_path / "session-worker-service.yaml"
    worker_svc_yaml.write_text(
        """
apiVersion: v1
kind: Service
metadata:
  name: session-worker-${SESSION_ID}
  namespace: sessions
  ownerReferences:
  - apiVersion: v1
    kind: Pod
    name: ${MASTER_POD_NAME}
    uid: ${MASTER_POD_UID}
    controller: true
spec:
  type: ClusterIP
  selector:
    app: session-worker
    sessionId: ${SESSION_ID}
  ports:
  - port: 80
    targetPort: 80
"""
    )
    monkeypatch.setenv("SESSION_WORKER_SERVICE_TEMPLATE_PATH", str(worker_svc_yaml))

    from zetta_utils.session import master as _master_mod

    _master_mod._shutdown_started = False


@pytest.fixture
def mock_batch_v1(mocker):
    return mocker.patch("zetta_utils.session.manager.k8s_client.BatchV1Api").return_value


@pytest.fixture
def manager_env(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKLOAD_NAMESPACE", "sessions")
    monkeypatch.setenv("SESSIONS_IMAGE_TAG", "web_api_gpu:test")

    job_tmpl = tmp_path / "session-master-template.yaml"
    job_tmpl.write_text(
        """
apiVersion: batch/v1
kind: Job
metadata:
  name: session-master-${SESSION_ID}
  namespace: sessions
  labels:
    app: session-master
    sessionId: ${SESSION_ID}
spec:
  template:
    spec:
      containers:
      - name: session-master
        image: ${SESSIONS_IMAGE_TAG}
"""
    )
    monkeypatch.setenv("SESSION_MASTER_TEMPLATE_PATH", str(job_tmpl))

    svc_tmpl = tmp_path / "session-master-service.yaml"
    svc_tmpl.write_text(
        """
apiVersion: v1
kind: Service
metadata:
  name: session-master-${SESSION_ID}
  namespace: sessions
  ownerReferences:
  - apiVersion: batch/v1
    kind: Job
    name: session-master-${SESSION_ID}
    uid: ${MASTER_JOB_UID}
    controller: true
spec:
  type: ClusterIP
  selector:
    app: session-master
    sessionId: ${SESSION_ID}
  ports:
  - port: 80
    targetPort: 80
"""
    )
    monkeypatch.setenv("SESSION_MASTER_SERVICE_TEMPLATE_PATH", str(svc_tmpl))


@pytest.fixture
def aiohttp_mock_session(mocker):
    """Mock ``aiohttp.ClientSession`` used by master to probe and dispatch.

    Returns a thin handle exposing ``.get`` and ``.post`` whose return values
    are async context managers. Each verb-call's ``__aenter__`` yields a mock
    response. Tests configure responses via ``set_get_response`` /
    ``set_post_response`` or by directly assigning ``side_effect`` on the verb
    mocks for sequenced behavior.
    """

    def _make_response(status: int, json_payload: dict | None = None):
        response = mocker.AsyncMock()
        response.status = status
        if json_payload is not None:
            response.json = mocker.AsyncMock(return_value=json_payload)
        response.raise_for_status = mocker.MagicMock()
        cm = mocker.AsyncMock()
        cm.__aenter__.return_value = response
        cm.__aexit__.return_value = None
        return cm, response

    session = mocker.MagicMock()
    session.__aenter__ = mocker.AsyncMock(return_value=session)
    session.__aexit__ = mocker.AsyncMock(return_value=None)

    session.get = mocker.MagicMock()
    session.post = mocker.MagicMock()

    def set_get_response(status: int = 200, json_payload: dict | None = None):
        cm, _ = _make_response(status, json_payload)
        session.get.return_value = cm

    def set_post_response(status: int = 200, json_payload: dict | None = None):
        cm, _ = _make_response(status, json_payload)
        session.post.return_value = cm

    session.set_get_response = set_get_response
    session.set_post_response = set_post_response
    session._make_response = _make_response

    mocker.patch("aiohttp.ClientSession", return_value=session)
    return session
