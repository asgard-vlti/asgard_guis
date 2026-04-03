"""Simple ZMQ REP endpoint for receiving watchdog status updates.

This script listens for watchdog payloads published by MCSClient.publish_wd()
and prints each incoming wd_status dictionary before replying with ACK.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import zmq

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


def _format_payload(payload: Any) -> str:
    try:
        return json.dumps(payload, indent=2, sort_keys=True)
    except (TypeError, ValueError):
        return str(payload)


def _colorize_state(value: str) -> str:
    if value == "open":
        return f"{GREEN}{value}{RESET}"
    if value == "closed":
        return f"{RED}{value}{RESET}"
    return value


def _print_watchdog_status(wd_status: Any) -> None:
    if not isinstance(wd_status, dict):
        print(_format_payload(wd_status), flush=True)
        return

    for task_name, task_status in wd_status.items():
        print(task_name, flush=True)
        if not isinstance(task_status, dict):
            print(f"  {task_status}", flush=True)
            continue

        process = _colorize_state(str(task_status.get("process", "unknown")))
        if "zmq" in task_status:
            zmq_state = _colorize_state(str(task_status.get("zmq", "unknown")))
            print(f"  process={process}", flush=True)
            print(f"  zmq={zmq_state}", flush=True)
        else:
            status = task_status.get("status", "unknown")
            print(f"  process={process}", flush=True)
            print(f"  status={status}", flush=True)


def run_server(bind_endpoint: str) -> None:
    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(bind_endpoint)

    logging.info("Watchdog status REP endpoint bound to %s", bind_endpoint)

    while True:
        message = socket.recv_string()
        try:
            wd_status = json.loads(message)
        except json.JSONDecodeError:
            wd_status = message

        _print_watchdog_status(wd_status)
        socket.send_string("ACK")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Receive watchdog status updates from MCSClient.publish_wd()."
    )
    parser.add_argument(
        "--bind-endpoint",
        default="tcp://*:7051",
        help="ZMQ endpoint to bind the REP socket to.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run_server(args.bind_endpoint)


if __name__ == "__main__":
    main()
