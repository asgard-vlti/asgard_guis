"""Simple ZMQ REP endpoint for receiving watchdog status updates.

This script listens for watchdog payloads published by MCSClient.publish_wd()
and prints each incoming wd_status dictionary before replying with ACK.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any
import datetime

import zmq


class TextStatusInfo:
    GREEN = "\033[32m"
    RED = "\033[31m"
    RESET = "\033[0m"
    CLEAR_SCREEN = "\033[2J\033[H"
    STALE_THRESHOLD_SECONDS = 5
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

    def __init__(self):
        self.last_wd_time = None

    @staticmethod
    def _format_payload(payload: Any) -> str:
        try:
            return json.dumps(payload, indent=2, sort_keys=True)
        except (TypeError, ValueError):
            return str(payload)

    @classmethod
    def _colorize_state(cls, value: str, inverse=False) -> str:
        good_phrases = ["open", "running", "true"]
        bad_phrases = ["closed", "error", "false"]

        if any(phrase in value.lower() for phrase in good_phrases):
            color = cls.GREEN if not inverse else cls.RED
            return f"{color}{value}{cls.RESET}"
        elif any(phrase in value.lower() for phrase in bad_phrases):
            color = cls.RED if not inverse else cls.GREEN
            return f"{color}{value}{cls.RESET}"

        # if value.lower() in ["open", "running"] or "error" in value.lower():
        #     return f"{cls.GREEN}{value}{cls.RESET}"
        # if value.lower() in ["closed"]:
        #     return f"{cls.RED}{value}{cls.RESET}"
        return value

    def _clear_screen(self) -> None:
        print(self.CLEAR_SCREEN, end="", flush=True)

    def _print_watchdog_status(
        self, wd_status: Any, update_last_time: bool = True
    ) -> None:
        if not isinstance(wd_status, dict):
            self._clear_screen()
            print(self._format_payload(wd_status), flush=True)
            return

        now = datetime.datetime.now(datetime.timezone.utc)

        is_stale = (
            self.last_wd_time is not None
            and (now - self.last_wd_time).total_seconds() > self.STALE_THRESHOLD_SECONDS
        )

        elapsed_seconds = (
            (now - self.last_wd_time).total_seconds()
            if self.last_wd_time is not None
            else 0.0
        )

        if update_last_time:
            self.last_wd_time = now

        header = f"last updated {elapsed_seconds:.2f} seconds ago"
        if is_stale:
            header = f"{self.RED}{header}{self.RESET}"
        print(header, flush=True)

        for task_name, task_status in wd_status.items():
            lines = [task_name]
            if not isinstance(task_status, dict):
                lines.append(f"  {task_status}")
                print("\n".join(lines), flush=True)
                continue

            process = self._colorize_state(str(task_status.get("process", "unknown")))
            if "zmq" in task_status:
                zmq_state = self._colorize_state(str(task_status.get("zmq", "unknown")))
                lines.append(f"  process: {process}")
                lines.append(f"  zmq: {zmq_state}")

                if task_status.get("status", None) is not None:
                    try:
                        s = json.loads(task_status["status"])
                    except json.JSONDecodeError:
                        s = task_status["status"]
                    if (
                        task_name in self.fields_of_interest
                        and self.fields_of_interest[task_name] is not []
                    ):
                        lines.append("  status:")
                        fields = self.fields_of_interest[task_name]
                        if fields is None:
                            if not isinstance(s, dict):
                                lines[
                                    -1
                                ] += f" {self._colorize_state(self._format_payload(s))}"
                            else:
                                for key, value in s.items():
                                    value_str = self._colorize_state(
                                        self._format_payload(value)
                                    )
                                    lines.append(f"    {key}: {value_str}")
                        else:
                            if len(fields) == 0:
                                # rm the state entry in lines
                                lines.pop()
                            elif len(fields) == 1:
                                field = fields[0]
                                value = s.get(field, "N/A")
                                value_str = self._colorize_state(str(value))
                                lines[-1] += f" {field}: {value_str}"
                            else:
                                for field in fields:
                                    inv_color = False
                                    if "error" in field.lower():
                                        inv_color = True
                                    value = s.get(field, "N/A")
                                    lines.append(
                                        f"    {field}: {self._colorize_state(str(value), inverse=inv_color)}"
                                    )

            else:
                status = task_status.get("status", "unknown")
                lines.append(f"  process: {process}")
                lines.append(f"  status: {self._colorize_state(status)}")

            print("\n".join(lines), flush=True)

    def run_server(self, bind_endpoint: str) -> None:
        context = zmq.Context.instance()
        socket = context.socket(zmq.REP)
        socket.bind(bind_endpoint)
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        last_wd_status = None

        logging.info("Watchdog status REP endpoint bound to %s", bind_endpoint)

        while True:
            events = dict(poller.poll(timeout=800))
            if socket in events and events[socket] == zmq.POLLIN:
                message = socket.recv_string()
                try:
                    wd_status = json.loads(message)
                except json.JSONDecodeError:
                    wd_status = message

                last_wd_status = wd_status
                self._clear_screen()
                self._print_watchdog_status(wd_status, update_last_time=True)
                socket.send_string("ACK")
                continue

            if last_wd_status is not None:
                self._clear_screen()
                self._print_watchdog_status(last_wd_status, update_last_time=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Receive watchdog status updates from MCSClient.publish_wd()."
    )
    parser.add_argument(
        "--bind-endpoint",
        default="tcp://*:7051",
        help="ZMQ endpoint to bind the REP socket to.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="If the display should be a GUI instead of terminal output",
    )
    # toggle sim option
    parser.add_argument(
        "--sim",
        action="store_true",
        help="If True, simulates receiving a watchdog message every 5 seconds instead of binding a ZMQ socket.",
    )

    args = parser.parse_args()

    if args.sim:
        args.connect_endpoint = "tcp://localhost:7051"

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    sinfo = TextStatusInfo()
    sinfo.run_server(args.bind_endpoint)


if __name__ == "__main__":
    main()
