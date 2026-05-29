# pylint: disable=import-error,wrong-import-position,import-outside-toplevel
import pytest

pytest.importorskip("fastapi")

from starlette.routing import Mount


def test_worker_app_exposes_healthz():
    from web_api.app import worker

    paths = {getattr(route, "path", None) for route in worker.app.routes}
    assert "/healthz" in paths


def test_worker_app_exposes_run_spec():
    from web_api.app import worker

    paths = {getattr(route, "path", None) for route in worker.app.routes}
    assert "/run_spec/" in paths


def test_worker_app_has_no_portal_routers():
    from web_api.app import worker

    mount_paths = {route.path for route in worker.app.routes if isinstance(route, Mount)}
    portal_mounts = {
        "/alignment",
        "/annotations",
        "/collections",
        "/layer_groups",
        "/layers",
        "/painting",
        "/precomputed",
        "/segmentation",
        "/sessions",
        "/tasks",
    }
    assert portal_mounts.isdisjoint(mount_paths)


def test_worker_app_has_auth_middleware():
    from web_api.app import worker
    from web_api.app.auth import check_authorized_user

    middleware_funcs = [
        getattr(m, "kwargs", {}).get("dispatch") for m in worker.app.user_middleware
    ]
    assert check_authorized_user in middleware_funcs
