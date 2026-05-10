"""Shared types and external clients for the run garbage collector.

Sits below :mod:`orchestrator`, :mod:`slack`, and :mod:`users` in the
dependency graph. Keeping the report dataclasses and the Slack
WebClient accessor here breaks what would otherwise be cycles between
those modules.

This module imports from :mod:`deleters`, :mod:`discovery`, and the
top-level :mod:`zetta_utils.run.resource`; nothing under ``gc/`` imports
back into it (other than from this module out), so it forms the
foundation layer of the submodule.
"""

from __future__ import annotations

import functools
import os

import attrs
from slack_sdk import WebClient

from zetta_utils.cloud_management.resource_allocation.k8s.common import ClusterInfo
from zetta_utils.run.gc.deleters import DeleteOutcome, DeleteStatus
from zetta_utils.run.gc.discovery import ClusterFailure
from zetta_utils.run.resource import Resource


@functools.cache
def get_slack_client() -> WebClient:
    """Return the process-wide :class:`WebClient`, built on first use.

    Reads ``SLACK_BOT_TOKEN`` from the environment on first call only.
    The result is cached for the process lifetime; we do not rotate
    tokens across cycles. Lazy build keeps importing siblings like
    :mod:`deleters` or :mod:`discovery` token-free.
    """
    return WebClient(token=os.environ["SLACK_BOT_TOKEN"])


@attrs.frozen
class ResourceOutcome:
    """One resource's delete attempt during a cleanup pass.

    :param resource_id: ``RESOURCE_DB`` row key.
    :param resource: Inflated :class:`Resource` object.
    :param outcome: The :class:`DeleteOutcome` returned by the deleter
        (or a synthesized one when discovery determined the resource was
        already absent).
    """

    resource_id: str
    resource: Resource
    outcome: DeleteOutcome


@attrs.frozen
class CleanupReport:
    """Aggregated outcome of a single stale run's cleanup pass.

    :param run_id: Stripped run id (no ``run-`` prefix).
    :param zetta_user: Run owner, as recorded in ``RUN_DB``.
    :param timestamp: Unix epoch the run was registered.
    :param heartbeat: Unix epoch of the run's last heartbeat.
    :param outcomes: Per-resource outcomes.
    :param cluster_failures: Clusters that were unreachable for this run.
    :param error_class: Dominant failure category for this cycle, used by
        the Slack layer to categorize and to decide whether to re-DM.
        Empty when the run cleaned up fully.
    """

    run_id: str
    zetta_user: str
    timestamp: float
    heartbeat: float
    outcomes: list[ResourceOutcome]
    cluster_failures: dict[ClusterInfo, ClusterFailure]
    error_class: str

    @property
    def fully_succeeded(self) -> bool:
        return not self.cluster_failures and all(
            o.outcome.status in (DeleteStatus.DELETED, DeleteStatus.NOT_FOUND)
            for o in self.outcomes
        )

    @property
    def status_label(self) -> str:
        """One of ``"OK"``, ``"WARN"``, ``"FAIL"``; matches the Slack
        channel-summary 4-char prefix once padded.
        """
        if self.fully_succeeded:
            return "OK"
        has_any_success = any(
            o.outcome.status in (DeleteStatus.DELETED, DeleteStatus.NOT_FOUND)
            for o in self.outcomes
        )
        return "WARN" if has_any_success else "FAIL"
