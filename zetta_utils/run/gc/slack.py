"""Slack channel summary, threaded per-run details, and per-owner DM for the GC.

The channel summary is a single message: a one-line header (stale count,
heartbeat threshold ISO, OK/WARN/FAIL totals) followed by a monospace
code block of one row per run. Per-run blocker details and remediation
hints go as threaded replies under that summary so the parent stays
compact and scannable.

When ``ZETTA_GC_NOTIFY_USERS=1``, the run owner is DM'd directly the
first time their run shows a given blocker class (or whenever the class
changes). Repeat blockers in the same class are not re-DM'd until the
class changes; the owner can still see the live state in the channel
summary.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from slack_sdk.errors import SlackApiError

from zetta_utils import constants
from zetta_utils.log import get_logger
from zetta_utils.run.db import RUN_DB
from zetta_utils.run.gc.common import CleanupReport, get_slack_client
from zetta_utils.run.gc.deleters import DeleteStatus
from zetta_utils.run.gc.state import RunGCState, mark_notified
from zetta_utils.run.gc.users import UserResolver
from zetta_utils.run.gc.utils import format_cluster

logger = get_logger("zetta_utils")

NOTIFY_USERS = os.environ.get("ZETTA_GC_NOTIFY_USERS", "0") == "1"

#: Per-error-class display label and one-line remediation hint, used by
#: both the threaded channel reply and the owner DM.
_CATEGORY_HINTS: dict[str, tuple[str, str]] = {
    "cluster_404": (
        "cluster gone",
        "run reference points at a deleted cluster, clear it manually",
    ),
    "cluster_auth": (
        "cluster auth",
        "GC bot needs cluster access, verify RBAC and credentials",
    ),
    "k8s_auth": (
        "k8s permission",
        "GC bot lacks RBAC, apply scripts/gcp/rbac.yml",
    ),
    "k8s_5xx": (
        "k8s 5xx",
        "transient k8s control plane error, will retry on next cycle",
    ),
    "sqs": (
        "sqs error",
        "transient AWS SQS issue, will retry on next cycle",
    ),
    "k8s_other": ("k8s other", "unclassified k8s error, check logs"),
}


def post_message(msg: str, priority: bool = True) -> None:
    """Single-message helper; preserved for the legacy idle ping path.

    When ``priority=False``, the previous non-priority message ts stored
    in ``RUN_DB["gc_last_msg"]`` is deleted so only one idle ping is
    visible at a time.
    """
    try:
        channel = os.environ["SLACK_CHANNEL"]
        response = get_slack_client().chat_postMessage(channel=channel, text=msg)
        if not priority:
            try:
                get_slack_client().chat_delete(channel=channel, ts=RUN_DB["gc_last_msg"])
            except (KeyError, SlackApiError):
                ...
            RUN_DB["gc_last_msg"] = response["ts"]
    except SlackApiError as err:
        logger.warning(err.response["error"])


def post_idle() -> None:
    """Idle cycle: post ``Nothing to do.`` and delete the previous idle ping."""
    post_message("Nothing to do.", priority=False)


def post_cycle(
    reports: list[CleanupReport],
    states_pre: dict[str, RunGCState],
    user_resolver: UserResolver,
) -> None:
    """Post the compact channel summary, per-run thread replies, and owner DMs.

    :param reports: One :class:`CleanupReport` per stale run, in order.
    :param states_pre: GC state for each run as it stood *before* the
        current cleanup pass wrote its updates. Drives DM gating
        (compares this cycle's ``error_class`` to ``last_notify_error_class``).
    :param user_resolver: Lazy ``ZETTA_USER`` -> Slack id resolver.
    """
    if not reports:
        return
    summary_ts = _post_channel_summary(reports)
    for report in reports:
        if report.status_label == "OK":
            continue
        if summary_ts is not None:
            _post_thread_reply(summary_ts, report)
        if NOTIFY_USERS:
            _maybe_dm_owner(report, states_pre.get(report.run_id, RunGCState()), user_resolver)


def _post_channel_summary(reports: list[CleanupReport]) -> str | None:
    channel = os.environ["SLACK_CHANNEL"]
    hb_threshold = time.time() - int(os.environ["EXECUTION_HEARTBEAT_LOOKBACK"])
    threshold_str = datetime.fromtimestamp(hb_threshold, timezone.utc).isoformat()

    counts = {"OK": 0, "WARN": 0, "FAIL": 0}
    for report in reports:
        counts[report.status_label] += 1

    header = (
        f"{len(reports)} run(s) with heartbeat before `{threshold_str}`. "
        f"OK {counts['OK']} · WARN {counts['WARN']} · FAIL {counts['FAIL']}."
    )

    now = time.time()
    body = "\n".join(
        _format_row(report, now) for report in sorted(reports, key=lambda r: r.run_id)
    )
    text = f"{header}\n```\n{body}\n```"

    try:
        response = get_slack_client().chat_postMessage(channel=channel, text=text)
        return response["ts"]
    except SlackApiError as err:
        logger.warning(f"Channel summary post failed: {err.response['error']}")
        return None


def _format_row(report: CleanupReport, now: float) -> str:
    status = report.status_label.ljust(4)
    display = report.run_id if len(report.run_id) <= 42 else report.run_id[:41] + "…"
    run_id_cell = _run_id_link(report.run_id, display)
    # Pad based on the visible-width display string, not the full mrkdwn
    # link, so columns line up in the rendered Slack code block.
    pad = " " * max(0, 42 - len(display))
    user = report.zetta_user[:12]
    age = _relative_age(now, report.timestamp)
    last_hb = _relative_age(now, report.heartbeat)
    project = _report_project(report) or "not available"
    c = report.counts
    resources = f"{c['total']}r d{c['deleted']}/n{c['not_found']}/f{c['failed']}"
    return (
        f"{status} {run_id_cell}{pad} {user:<12} age {age:>3} last_hb {last_hb:>3} "
        f"{resources:<18} {project}"
    )


def _report_project(report: CleanupReport) -> str:
    """GCP project id from the run's first registered cluster."""
    if not report.clusters:
        return ""
    return report.clusters[0].project or ""


_FIRESTORE_CONSOLE_BASE = (
    "https://console.cloud.google.com/firestore/databases/run-db/data/panel/run-info"
)


def _run_id_link(run_id: str, display: str | None = None) -> str:
    """Slack-mrkdwn hyperlink to the run's Firestore console page.

    The ``?project=`` query parameter is the project that owns the
    ``run-db`` Firestore database (``constants.DEFAULT_PROJECT``), not
    the run's own cluster project — the console URL needs the database
    host project to authorize the read.

    :param run_id: Full run id; used both in the URL path and as the
        default visible link text.
    :param display: Visible link text. Defaults to ``run_id``.
    """
    text = display if display is not None else run_id
    url = f"{_FIRESTORE_CONSOLE_BASE}/{run_id}?project={constants.DEFAULT_PROJECT}"
    return f"<{url}|{text}>"


def _relative_age(now: float, then: float) -> str:
    if then <= 0:
        return "?"
    delta = max(0, int(now - then))
    if delta < 60:
        return f"{delta}s"
    if delta < 3600:
        return f"{delta // 60}m"
    if delta < 86400:
        return f"{delta // 3600}h"
    return f"{delta // 86400}d"


def _post_thread_reply(summary_ts: str, report: CleanupReport) -> None:
    channel = os.environ["SLACK_CHANNEL"]
    text = _format_thread_text(report)
    try:
        get_slack_client().chat_postMessage(channel=channel, thread_ts=summary_ts, text=text)
    except SlackApiError as err:
        logger.warning(f"Thread reply for {report.run_id} failed: {err.response['error']}")


def _format_thread_text(report: CleanupReport) -> str:
    label, hint = _category_label_and_hint(report.error_class)
    project = _report_project(report)
    project_suffix = f" @ `{project}`" if project else ""
    c = report.counts
    header = [
        f"*{_run_id_link(report.run_id)}* ({report.zetta_user}{project_suffix}) — {label}",
        hint,
        f"Resources: {c['total']} total, deleted {c['deleted']}, "
        f"not_found {c['not_found']}, failed {c['failed']}.",
    ]
    body: list[str] = []
    failed = [o for o in report.outcomes if o.outcome.status == DeleteStatus.FAILED]
    if failed:
        body.append("Failed resources:")
        for outcome in failed:
            body.append(
                f"  - {outcome.resource.type}/{outcome.resource.name}\n"
                f"      {outcome.outcome.error_class}: {outcome.outcome.error}"
            )
    if report.cluster_failures:
        if body:
            body.append("")
        body.append("Cluster failures:")
        for cluster, cf in report.cluster_failures.items():
            body.append(f"  - {format_cluster(cluster)}\n      {cf.error_class}: {cf.error}")
    if body:
        header.append("```\n" + "\n".join(body) + "\n```")
    return "\n".join(header)


def _maybe_dm_owner(
    report: CleanupReport,
    state_pre: RunGCState,
    user_resolver: UserResolver,
) -> None:
    if not report.error_class:
        return
    if report.error_class == state_pre.last_notify_error_class:
        return
    slack_id = user_resolver.resolve(report.zetta_user)
    if not slack_id:
        return
    try:
        im_response = get_slack_client().conversations_open(users=slack_id)
        im_channel = im_response["channel"]["id"]
        get_slack_client().chat_postMessage(channel=im_channel, text=_format_dm_text(report))
        mark_notified(report.run_id, report.error_class)
    except SlackApiError as err:
        logger.warning(f"DM to {report.zetta_user} failed: {err.response['error']}")


def _format_dm_text(report: CleanupReport) -> str:
    label, hint = _category_label_and_hint(report.error_class)
    project = _report_project(report)
    project_line = f"GCP project: `{project}`" if project else ""
    c = report.counts
    header = [
        f"Your run {_run_id_link(report.run_id)} is blocked from cleanup.",
        f"{label}: {hint}",
    ]
    if project_line:
        header.append(project_line)
    header.append(
        f"Resources: {c['total']} total, deleted {c['deleted']}, "
        f"not_found {c['not_found']}, failed {c['failed']}."
    )
    body: list[str] = []
    failed = [o for o in report.outcomes if o.outcome.status == DeleteStatus.FAILED]
    if failed:
        body.append("Failed resources:")
        for outcome in failed:
            body.append(f"  - {outcome.resource.type}/{outcome.resource.name}")
    if report.cluster_failures:
        if body:
            body.append("")
        body.append("Cluster failures:")
        for cluster, cf in report.cluster_failures.items():
            body.append(f"  - {format_cluster(cluster)}: {cf.error_class}")
    if body:
        header.append("```\n" + "\n".join(body) + "\n```")
    return "\n".join(header)


def _category_label_and_hint(error_class: str) -> tuple[str, str]:
    return _CATEGORY_HINTS.get(error_class, ("error", "unclassified failure"))
