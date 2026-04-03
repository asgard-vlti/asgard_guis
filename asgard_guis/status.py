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
CLEAR_SCREEN = "\033[2J\033[H"


def _format_payload(payload: Any) -> str:
    try:
        return json.dumps(payload, indent=2, sort_keys=True)
    except (TypeError, ValueError):
        return str(payload)


def _colorize_state(value: str) -> str:
    if value.lower() in ["open", "running", "true"] or "error" in value.lower():
        return f"{GREEN}{value}{RESET}"
    if value.lower() in ["closed"]:
        return f"{RED}{value}{RESET}"
    return value


def _clear_screen() -> None:
    print(CLEAR_SCREEN, end="", flush=True)


fields_of_interest = {
    "BTT1": ["cnt"],
    "BTT2": ["cnt"],
    "BTT3": ["cnt"],
    "BTT4": ["cnt"],
    "CRED1": ["cam_status", "shm_error", "fps"],
    "DM": None,  # this means print the whole DM status entry
    "HDLR": ["cnt", "locked"],
    "back_end": [],
}


def _print_watchdog_status(wd_status: Any) -> None:
    if not isinstance(wd_status, dict):
        _clear_screen()
        print(_format_payload(wd_status), flush=True)
        return

    for task_name, task_status in wd_status.items():
        lines = [task_name]
        if not isinstance(task_status, dict):
            lines.append(f"  {task_status}")
            print("\n".join(lines), flush=True)
            continue

        process = _colorize_state(str(task_status.get("process", "unknown")))
        if "zmq" in task_status:
            zmq_state = _colorize_state(str(task_status.get("zmq", "unknown")))
            lines.append(f"\tprocess:{process}")
            lines.append(f"\tzmq:{zmq_state}")

            if task_status.get("status", None) is not None:
                s = json.loads(task_status["status"])

        else:
            status = task_status.get("status", "unknown")
            lines.append(f"\tprocess:{process}")
            lines.append(f"\tstatus:{status}")

        print("\n".join(lines), flush=True)


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

        _clear_screen()

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
