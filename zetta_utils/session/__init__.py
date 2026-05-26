"""Per-session orchestration: manager service, master process, and reconcile backstop."""

import os

from google.cloud import firestore

from zetta_utils import constants

_sessions_db: firestore.Client | None = None


def _get_sessions_db() -> firestore.Client:
    """Lazily build and cache the Firestore client for the main (sessions) DB.

    Constructed once per process and shared by the manager, master, and
    reconcile components so they all address the same ``sessions/*`` documents.
    Construction is deferred to first use so importing a session module does not
    require GCP credentials. The session documents live in the main database,
    distinct from the run-info DB.
    """
    global _sessions_db
    if _sessions_db is None:
        _sessions_db = firestore.Client(
            project=os.environ.get("SESSIONS_FIRESTORE_PROJECT", constants.DEFAULT_PROJECT),
            database=os.environ.get("SESSIONS_FIRESTORE_DATABASE"),
        )
    return _sessions_db
