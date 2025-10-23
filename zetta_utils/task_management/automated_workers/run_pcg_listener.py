#!/usr/bin/env python3
"""
CLI script to run the PyChunkedGraph edit listener worker.

Usage:
    python -m zetta_utils.task_management.automated_workers.run_pcg_listener \
        --project-id zetta-proofreading \
        --subscription-name proofreading-common_PCG_EDIT_INFO \
        --project-name your_project \
        --datastack-name your_datastack \
        --server-address https://cave.your-server.com
"""

import argparse
import logging

from .pcg_edit_listener import run_pcg_edit_listener

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    parser = argparse.ArgumentParser(
        description="Run PyChunkedGraph edit listener worker"
    )
    parser.add_argument(
        "--project-id",
        required=True,
        help="GCP project ID",
    )
    parser.add_argument(
        "--subscription-name",
        required=True,
        help="Pub/Sub subscription name (e.g., proofreading-common_PCG_EDIT_INFO)",
    )
    parser.add_argument(
        "--project-name",
        required=True,
        help="Task management project name",
    )
    parser.add_argument(
        "--datastack-name",
        required=False,
        help="CAVE datastack name (auto-detected from project if not provided)",
    )
    parser.add_argument(
        "--server-address",
        required=False,
        help="CAVEclient server address (auto-detected if not provided)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Polling interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=10,
        help="Maximum messages to pull per batch (default: 10)",
    )

    args = parser.parse_args()

    run_pcg_edit_listener(
        project_id=args.project_id,
        subscription_name=args.subscription_name,
        project_name=args.project_name,
        datastack_name=args.datastack_name,
        server_address=args.server_address,
        poll_interval_sec=args.poll_interval,
        max_messages=args.max_messages,
    )


if __name__ == "__main__":
    main()
