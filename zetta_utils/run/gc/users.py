"""Resolve ZETTA_USER to a Slack user id with a Firestore-backed cache.

Read path is lazy: the Firestore mapping (collection ``slack-user-mapping``,
row key = ``zetta_user``, column = ``slack_id``) is fetched on the first
:meth:`UserResolver.resolve` call within a process. Cycles that need no DMs
never touch Firestore.

Cache miss fallbacks, in order:

1. ``users.lookupByEmail`` on the raw ``ZETTA_USER`` (used when it already
   contains an ``@``).
2. ``users.lookupByEmail`` on ``ZETTA_USER@<domain>`` where ``<domain>`` is
   ``ZETTA_GC_USER_EMAIL_DOMAIN`` (default ``zetta.ai``).
3. A single paginated ``users.list`` scan, matching the bare username
   against each member's ``name`` / ``display_name`` / ``real_name``. The
   scan result is cached on the resolver so repeated misses within the
   same cycle don't re-scan.

On any hit, the resolved id is persisted to Firestore so the next cycle
finds it cached. On total miss, :meth:`resolve` returns ``None`` and logs
a warning.
"""

from __future__ import annotations

import os

from slack_sdk.errors import SlackApiError

from zetta_utils import constants
from zetta_utils.layer.db_layer.firestore import build_firestore_layer
from zetta_utils.log import get_logger
from zetta_utils.run.gc.slack import slack_client

logger = get_logger("zetta_utils")

SLACK_USER_MAPPING_COLLECTION = "slack-user-mapping"
SLACK_USER_MAPPING_DB = build_firestore_layer(
    SLACK_USER_MAPPING_COLLECTION,
    database=constants.RUN_DATABASE,
    project=constants.DEFAULT_PROJECT,
)

DEFAULT_EMAIL_DOMAIN = os.environ.get("ZETTA_GC_USER_EMAIL_DOMAIN", "zetta.ai")
_USERS_LIST_PAGE_LIMIT = 200


class UserResolver:
    """Lazy ZETTA_USER -> Slack user id resolver with Firestore cache."""

    def __init__(self) -> None:
        self._loaded = False
        self._cache: dict[str, str] = {}
        self._user_list_index: dict[str, str] | None = None

    def resolve(self, zetta_user: str) -> str | None:
        """Return the Slack user id for ``zetta_user`` or ``None`` on miss.

        :param zetta_user: The ``ZETTA_USER`` value recorded on the run.
        """
        if not zetta_user:
            return None
        if not self._loaded:
            self._cache = self._load_cache()
            self._loaded = True
        if zetta_user in self._cache:
            return self._cache[zetta_user]
        slack_id = self._lookup_via_slack(zetta_user)
        if slack_id is not None:
            self._persist(zetta_user, slack_id)
            self._cache[zetta_user] = slack_id
        else:
            logger.warning(f"Could not resolve zetta_user {zetta_user!r} to a Slack id.")
        return slack_id

    def _load_cache(self) -> dict[str, str]:
        try:
            rows = SLACK_USER_MAPPING_DB.query()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(f"Failed to load slack-user-mapping cache: {exc}")
            return {}
        cache: dict[str, str] = {}
        for zetta_user, row in rows.items():
            slack_id = row.get("slack_id")
            if isinstance(slack_id, str) and slack_id:
                cache[zetta_user] = slack_id
        return cache

    def _lookup_via_slack(self, zetta_user: str) -> str | None:
        for email in _candidate_emails(zetta_user):
            slack_id = self._lookup_by_email(email)
            if slack_id is not None:
                return slack_id
        return self._lookup_via_list_scan(zetta_user)

    def _lookup_by_email(self, email: str) -> str | None:
        try:
            response = slack_client.users_lookupByEmail(email=email)
        except SlackApiError as exc:
            error = exc.response.get("error", "") if exc.response else ""
            if error != "users_not_found":
                logger.warning(f"Slack lookupByEmail failed for {email!r}: {error}")
            return None
        user = response.get("user") if response else None
        return user.get("id") if user else None

    def _lookup_via_list_scan(self, zetta_user: str) -> str | None:
        if self._user_list_index is None:
            self._user_list_index = self._build_user_list_index()
        return self._user_list_index.get(zetta_user)

    def _build_user_list_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        cursor: str | None = None
        try:
            while True:
                response = slack_client.users_list(limit=_USERS_LIST_PAGE_LIMIT, cursor=cursor)
                members: list[dict] = response.get("members") or []
                for member in members:
                    if member.get("deleted"):
                        continue
                    user_id = member.get("id")
                    if not user_id:
                        continue
                    profile = member.get("profile") or {}
                    names = (
                        member.get("name"),
                        profile.get("display_name"),
                        profile.get("real_name"),
                    )
                    for name in names:
                        if name and name not in index:
                            index[name] = user_id
                meta = response.get("response_metadata") or {}
                cursor = meta.get("next_cursor") or None
                if not cursor:
                    break
        except SlackApiError as exc:
            error = exc.response.get("error", "") if exc.response else ""
            logger.warning(f"Slack users.list scan failed: {error}")
        return index

    def _persist(self, zetta_user: str, slack_id: str) -> None:
        try:
            SLACK_USER_MAPPING_DB[(zetta_user, ("slack_id",))] = {"slack_id": slack_id}
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(f"Failed to persist slack mapping for {zetta_user!r}: {exc}")


def _candidate_emails(zetta_user: str) -> list[str]:
    """Return the email forms to try against ``users.lookupByEmail``."""
    if "@" in zetta_user:
        return [zetta_user]
    if DEFAULT_EMAIL_DOMAIN:
        return [f"{zetta_user}@{DEFAULT_EMAIL_DOMAIN}"]
    return []
