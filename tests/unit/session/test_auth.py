# pylint: disable=import-error,wrong-import-position,import-outside-toplevel,unused-argument
import pytest

pytest.importorskip("fastapi")


def _make_request(mocker, *, method="GET", path="/run_spec/", headers=None):
    request = mocker.Mock()
    request.method = method
    request.url.path = path
    request.headers = headers if headers is not None else {}
    return request


def _make_call_next(mocker):
    return mocker.AsyncMock(return_value=mocker.Mock(name="downstream_response"))


async def test_missing_authorization_header_returns_401(monkeypatch, mocker):
    monkeypatch.setenv("OAUTH_CLIENT_ID", "client-id")
    from web_api.app import auth

    call_next = _make_call_next(mocker)
    resp = await auth.check_authorized_user(_make_request(mocker, headers={}), call_next)

    assert resp.status_code == 401
    call_next.assert_not_called()


async def test_garbled_authorization_header_returns_401(monkeypatch, mocker):
    monkeypatch.setenv("OAUTH_CLIENT_ID", "client-id")
    from web_api.app import auth

    call_next = _make_call_next(mocker)
    resp = await auth.check_authorized_user(
        _make_request(mocker, headers={"authorization": ""}), call_next
    )

    assert resp.status_code == 401
    call_next.assert_not_called()


async def test_healthz_passes_without_token(mocker):
    from web_api.app import auth

    call_next = _make_call_next(mocker)
    request = _make_request(mocker, path="/healthz", headers={})
    resp = await auth.check_authorized_user(request, call_next)

    call_next.assert_awaited_once_with(request)
    assert resp is call_next.return_value


async def test_options_passes_without_token(mocker):
    from web_api.app import auth

    call_next = _make_call_next(mocker)
    request = _make_request(mocker, method="OPTIONS", headers={})
    resp = await auth.check_authorized_user(request, call_next)

    call_next.assert_awaited_once_with(request)
    assert resp is call_next.return_value


async def test_non_zetta_email_rejected(monkeypatch, mocker):
    monkeypatch.setenv("OAUTH_CLIENT_ID", "client-id")
    from web_api.app import auth

    mocker.patch(
        "web_api.app.auth.id_token.verify_oauth2_token", return_value={"email": "user@evil.com"}
    )
    call_next = _make_call_next(mocker)
    resp = await auth.check_authorized_user(
        _make_request(mocker, headers={"authorization": "Bearer tok"}), call_next
    )

    assert resp.status_code == 401
    call_next.assert_not_called()


async def test_valid_zetta_token_accepted(monkeypatch, mocker):
    monkeypatch.setenv("OAUTH_CLIENT_ID", "client-id")
    from web_api.app import auth

    mocker.patch(
        "web_api.app.auth.id_token.verify_oauth2_token", return_value={"email": "user@zetta.ai"}
    )
    call_next = _make_call_next(mocker)
    request = _make_request(mocker, headers={"authorization": "Bearer tok"})
    resp = await auth.check_authorized_user(request, call_next)

    call_next.assert_awaited_once_with(request)
    assert resp is call_next.return_value
