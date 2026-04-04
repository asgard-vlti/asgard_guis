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
import time
import numpy as np

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

    def _print_watchdog_status(self, wd_status: Any) -> None:
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

        logging.info("Watchdog status REP endpoint bound to %s", bind_endpoint)

        while True:
            message = socket.recv_string()
            try:
                wd_status = json.loads(message)
            except json.JSONDecodeError:
                wd_status = message

            self._clear_screen()

            self._print_watchdog_status(wd_status)
            socket.send_string("ACK")


def run_simulator(sinfo: TextStatusInfo) -> None:
    while True:
        # pick a random number between 3-7 seconds for the pause time to simulate staleness
        pause_time = np.random.uniform(3, 7)
        sinfo._clear_screen()
        payload = sim_msg()
        wd_status = json.loads(payload)
        sinfo._print_watchdog_status(wd_status)
        time.sleep(pause_time)


def sim_msg():
    cnt = int(time.time()) % 10000  # just a changing number for demonstration
    payload = {
        "BTT1": {
            "process": "open",
            "status": f'{{"cnt":{cnt},"flux":119221.0,"tx":1.61,"ty":2.173}}',
            "zmq": "open",
        },
        "BTT2": {
            "process": "open",
            "status": f'{{"cnt":{cnt},"flux":90724.9,"tx":-5.526,"ty":-10.945}}',
            "zmq": "open",
        },
        "BTT3": {
            "process": "open",
            "status": f'{{"cnt":{cnt},"flux":27516.6,"tx":35.318,"ty":-20.465}}',
            "zmq": "open",
        },
        "BTT4": {
            "process": "open",
            "status": f'{{"cnt":{cnt},"flux":108671.2,"tx":2.152,"ty":5.303}}',
            "zmq": "open",
        },
        "CRED1": {
            "process": "open",
            "status": f'{{"cnt":{cnt},"cam_status":"running","fps":2000.0,"nbreads":1,"shm_error":false,"skipped_frames":0,"tsig_len":5}}',
            "zmq": "open",
        },
        "DM": {"process": "open", "status": '"running"', "zmq": "open"},
        "Eng gui": {"process": "open", "status": "running"},
        "HDLR": {
            "process": "closed",
            "status": '{"closure_phase_K1":[0.383,-0.544,-0.701,0.217],"closure_phase_K2":[0.282,0.09,0.054,0.244],"cnt":9534,"dl_offload":[0.0,0.0,0.0,0.0],"dm_piston":[0.0,0.0,0.0,0.0],"gd_bl":[4.738,-5.107,2.674,2.303,-3.008,-5.228],"gd_phasor_imag":[823302887.6,-2105260658.1,8302004.9,-1941856682.1,8011147917.6,-2687869428.2],"gd_phasor_real":[726425994.3,-1590734035.8,-568120110.2,-12364084896.9,-275330639.3,-431822351.4],"gd_snr":[39.55,62.87,21.79,162.84,132.92,72.11],"gd_tel":[2.439,-0.437,1.363,-3.365],"itime":234.0499999999321,"locked":true,"pd_av":[-0.242,-0.13,0.062,0.173,0.22,0.08],"pd_av_filtered":[-0.242,-0.13,0.062,0.173,0.22,0.08],"pd_bl":[-0.234,-0.16,0.035,0.154,0.23,0.09],"pd_snr":[2.64,5.51,1.3,17.16,15.43,7.07],"pd_tel":[0.388,0.075,0.228,-0.692],"test_ix":0,"test_n":0,"v2_K1":[0.0052,0.015,0.0012,0.1242,0.0951,0.0187],"v2_K2":[0.0749,0.1095,0.0609,0.3073,0.1658,0.0991]}',
            "zmq": "open",
        },
        "MDS": {
            "process": "open",
            "status": "NACK: Unkown custom command\n",
            "zmq": "open",
        },
        "back_end": {
            "process": "open",
            "status": '{"reply": {"time": "2026-04-03T21:34:20", "content": "OK:"}}',
            "zmq": "open",
        },
    }

    return json.dumps(payload)


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

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.sim:
        np.random.seed(0)  # for reproducible simulation
        run_simulator(TextStatusInfo())
    else:
        sinfo = TextStatusInfo()
        sinfo.run_server(args.bind_endpoint)


if __name__ == "__main__":
    main()
