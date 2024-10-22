"""
GC Slack interactions.
"""

import os

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from zetta_utils.log import get_logger
from zetta_utils.run import RUN_DB

logger = get_logger("zetta_utils")
slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])


def post_message(msg: str, priority=True):
    try:
        response = slack_client.chat_postMessage(channel=os.environ["SLACK_CHANNEL"], text=msg)
        if not priority:
            try:
                slack_client.chat_delete(
                    channel=os.environ["SLACK_CHANNEL"], ts=RUN_DB["gc_last_msg"]
                )
            except (KeyError, SlackApiError):
                ...
            RUN_DB["gc_last_msg"] = response["ts"]
    except SlackApiError as err:
        logger.warning(err.response["error"])
