# pylint: disable=protected-access,import-outside-toplevel


def test_get_sessions_db_constructs_once_with_env_vars(monkeypatch, mocker):
    """_get_sessions_db builds firestore.Client once and caches it."""
    import zetta_utils.session as session_mod

    monkeypatch.setattr(session_mod, "_sessions_db", None)
    monkeypatch.setenv("SESSIONS_FIRESTORE_PROJECT", "proj-x")
    monkeypatch.setenv("SESSIONS_FIRESTORE_DATABASE", "db-y")

    mock_client_cls = mocker.patch("zetta_utils.session.firestore.Client")
    sentinel = mocker.MagicMock()
    mock_client_cls.return_value = sentinel

    db1 = session_mod._get_sessions_db()
    db2 = session_mod._get_sessions_db()

    assert mock_client_cls.call_count == 1
    mock_client_cls.assert_called_once_with(project="proj-x", database="db-y")
    assert db1 is sentinel
    assert db1 is db2


def test_get_sessions_db_defaults_to_constants_project(monkeypatch, mocker):
    """Without env vars, Client is constructed with DEFAULT_PROJECT and database=None."""
    import zetta_utils.session as session_mod
    from zetta_utils import constants

    monkeypatch.setattr(session_mod, "_sessions_db", None)
    monkeypatch.delenv("SESSIONS_FIRESTORE_PROJECT", raising=False)
    monkeypatch.delenv("SESSIONS_FIRESTORE_DATABASE", raising=False)

    mock_client_cls = mocker.patch("zetta_utils.session.firestore.Client")

    session_mod._get_sessions_db()

    mock_client_cls.assert_called_once_with(project=constants.DEFAULT_PROJECT, database=None)
