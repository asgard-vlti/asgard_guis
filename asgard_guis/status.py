"""Simple ZMQ REP endpoint for receiving watchdog status updates.

This script listens for watchdog payloads published by MCSClient.publish_wd()
and prints each incoming wd_status dictionary before replying with ACK.
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import logging
import sys
from typing import Any, cast

import zmq

try:
    from PyQt5 import QtCore, QtWidgets  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional runtime dependency
    QtCore = cast(Any, None)
    QtWidgets = cast(Any, None)


class StatusFormatter:
    GREEN = "\033[32m"
    RED = "\033[31m"
    RESET = "\033[0m"
    STALE_THRESHOLD_SECONDS = 5
    STATE_COLORS = {
        "default": "#202020",
        "green": "#1f7a1f",
        "red": "#b71c1c",
    }
    fields_of_interest = {
        "BTT1": ["cnt"],
        "BTT2": ["cnt"],
        "BTT3": ["cnt"],
        "BTT4": ["cnt"],
        "CRED1": ["cam_status", "shm_error", "fps"],
        "DM": None,
        "HDLR": ["cnt", "locked"],
        "back_end": [],
    }

    def __init__(self) -> None:
        self.last_wd_time: datetime.datetime | None = None

    @staticmethod
    def _format_payload(payload: Any) -> str:
        try:
            return json.dumps(payload, indent=2, sort_keys=True)
        except (TypeError, ValueError):
            return str(payload)

    @classmethod
    def _state_color(cls, value: str, inverse: bool = False) -> str:
        lower = value.lower()
        good_phrases = ["open", "running", "true"]
        bad_phrases = ["closed", "error", "false"]

        if any(phrase in lower for phrase in good_phrases):
            return "red" if inverse else "green"
        if any(phrase in lower for phrase in bad_phrases):
            return "green" if inverse else "red"
        return "default"

    @classmethod
    def _colorize_state(cls, value: str, inverse: bool = False) -> str:
        color_name = cls._state_color(value, inverse=inverse)
        if color_name == "green":
            return f"{cls.GREEN}{value}{cls.RESET}"
        if color_name == "red":
            return f"{cls.RED}{value}{cls.RESET}"
        return value

    @classmethod
    def _decode_status(cls, status_payload: Any) -> Any:
        if not isinstance(status_payload, str):
            return status_payload
        try:
            return json.loads(status_payload)
        except json.JSONDecodeError:
            return status_payload

    def _timing(self, update_last_time: bool) -> tuple[float, bool]:
        now = datetime.datetime.now(datetime.timezone.utc)
        elapsed_seconds = (
            (now - self.last_wd_time).total_seconds()
            if self.last_wd_time is not None
            else 0.0
        )
        is_stale = (
            self.last_wd_time is not None
            and elapsed_seconds > self.STALE_THRESHOLD_SECONDS
        )
        if update_last_time:
            self.last_wd_time = now
        return elapsed_seconds, is_stale

    def _build_task_block(self, task_name: str, task_status: Any) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []

        def add_entry(
            label: str,
            value: Any,
            color: str = "default",
            indent: int = 1,
        ) -> None:
            entries.append(
                {
                    "label": label,
                    "value": str(value),
                    "color": color,
                    "indent": indent,
                }
            )

        if not isinstance(task_status, dict):
            add_entry("", self._format_payload(task_status), "default", indent=1)
            return {
                "task_name": task_name,
                "entries": entries,
                "has_red": False,
            }

        process = str(task_status.get("process", "unknown"))
        process_color = self._state_color(process)
        add_entry("process", process, process_color, indent=1)

        if "zmq" in task_status:
            zmq_state = str(task_status.get("zmq", "unknown"))
            zmq_color = self._state_color(zmq_state)
            add_entry("zmq", zmq_state, zmq_color, indent=1)

            if (
                task_status.get("status") is not None
                and task_name in self.fields_of_interest
            ):
                fields = self.fields_of_interest[task_name]
                decoded_status = self._decode_status(task_status.get("status"))

                if fields is None:
                    if not isinstance(decoded_status, dict):
                        status_value = self._format_payload(decoded_status)
                        add_entry(
                            "status",
                            status_value,
                            self._state_color(status_value),
                            indent=1,
                        )
                    else:
                        for key, value in decoded_status.items():
                            value_str = self._format_payload(value)
                            add_entry(
                                key,
                                value_str,
                                self._state_color(value_str),
                                indent=2,
                            )
                elif len(fields) == 1:
                    field = fields[0]
                    value = (
                        decoded_status.get(field, "N/A")
                        if isinstance(decoded_status, dict)
                        else "N/A"
                    )
                    value_str = str(value)
                    add_entry(field, value_str, self._state_color(value_str), indent=1)
                elif len(fields) > 1:
                    for field in fields:
                        inverse = "error" in field.lower()
                        value = (
                            decoded_status.get(field, "N/A")
                            if isinstance(decoded_status, dict)
                            else "N/A"
                        )
                        value_str = str(value)
                        add_entry(
                            field,
                            value_str,
                            self._state_color(value_str, inverse=inverse),
                            indent=2,
                        )
        else:
            status = str(task_status.get("status", "unknown"))
            add_entry("status", status, self._state_color(status), indent=1)

        has_red = any(entry["color"] == "red" for entry in entries)
        return {
            "task_name": task_name,
            "entries": entries,
            "has_red": has_red,
        }

    def build_render_state(
        self, wd_status: Any, update_last_time: bool = True
    ) -> dict[str, Any]:
        if not isinstance(wd_status, dict):
            return {
                "is_payload_dict": False,
                "payload": self._format_payload(wd_status),
                "elapsed_seconds": 0.0,
                "is_stale": False,
                "tasks": [],
            }

        elapsed_seconds, is_stale = self._timing(update_last_time=update_last_time)
        tasks = [
            self._build_task_block(task_name, task_status)
            for task_name, task_status in wd_status.items()
        ]
        return {
            "is_payload_dict": True,
            "elapsed_seconds": elapsed_seconds,
            "is_stale": is_stale,
            "tasks": tasks,
        }


class TextStatusInfo(StatusFormatter):
    CLEAR_SCREEN = "\033[2J\033[H"

    def _clear_screen(self) -> None:
        print(self.CLEAR_SCREEN, end="", flush=True)

    def _print_watchdog_status(
        self, wd_status: Any, update_last_time: bool = True
    ) -> None:
        render_state = self.build_render_state(
            wd_status, update_last_time=update_last_time
        )

        if not render_state["is_payload_dict"]:
            self._clear_screen()
            print(render_state["payload"], flush=True)
            return

        header = f"last updated {render_state['elapsed_seconds']:.2f} seconds ago"
        if render_state["is_stale"]:
            header = f"{self.RED}{header}{self.RESET}"
        print(header, flush=True)

        for task in render_state["tasks"]:
            lines = [task["task_name"]]
            for entry in task["entries"]:
                colorized_value = self._colorize_state(
                    entry["value"],
                    inverse=False,
                )
                indent = "  " * int(entry["indent"])
                if entry["label"]:
                    lines.append(f"{indent}{entry['label']}: {colorized_value}")
                else:
                    lines.append(f"{indent}{colorized_value}")

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

                print(f"Received watchdog status update at {datetime.datetime.now()}:")

                last_wd_status = wd_status
                self._clear_screen()
                self._print_watchdog_status(wd_status, update_last_time=True)
                socket.send_string("ACK")
                continue

            if last_wd_status is not None:
                self._clear_screen()
                self._print_watchdog_status(last_wd_status, update_last_time=False)


if QtWidgets is not None:

    class ProcessStatusBox(QtWidgets.QFrame):
        def __init__(self, process_name: str) -> None:
            super().__init__()
            self._process_name = process_name
            self._setup_ui()

        def _setup_ui(self) -> None:
            self.setObjectName("processBox")
            self.setMinimumHeight(125)

            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(10, 8, 10, 8)
            layout.setSpacing(6)

            self.title = QtWidgets.QLabel(self._process_name)
            self.title.setStyleSheet("font-weight: 700; color: #111111;")
            self.details = QtWidgets.QLabel()
            self.details.setStyleSheet("color: #202020;")
            self.details.setTextFormat(QtCore.Qt.RichText)
            self.details.setWordWrap(True)
            self.details.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)

            layout.addWidget(self.title)
            layout.addWidget(self.details, 1)

            self._apply_border(red_border=False)

        def _apply_border(self, red_border: bool) -> None:
            border = "#c62828" if red_border else "#888888"
            width = 2 if red_border else 1
            self.setStyleSheet(
                "QFrame#processBox {"
                f"border: {width}px solid {border};"
                "border-radius: 6px;"
                "background-color: #f8f8f8;"
                "}"
            )

        def update_from_task(self, task_block: dict[str, Any]) -> None:
            html_lines: list[str] = []
            for entry in task_block["entries"]:
                indent = "&nbsp;" * (int(entry["indent"]) * 4)
                value_color = StatusFormatter.STATE_COLORS.get(
                    entry["color"], "#202020"
                )
                escaped_value = html.escape(str(entry["value"]))
                escaped_label = html.escape(str(entry["label"]))
                if escaped_label:
                    html_lines.append(
                        f"{indent}<span style='color:#505050'>{escaped_label}:</span> "
                        f"<span style='color:{value_color}'>{escaped_value}</span>"
                    )
                else:
                    html_lines.append(
                        f"{indent}<span style='color:{value_color}'>{escaped_value}</span>"
                    )

            self.details.setText("<br/>".join(html_lines))
            self._apply_border(red_border=bool(task_block.get("has_red")))

    class WatchdogStatusWindow(QtWidgets.QWidget):
        PROCESS_GRID = {
            "BTT1": (0, 0, 1, 1),
            "BTT2": (0, 1, 1, 1),
            "BTT3": (0, 2, 1, 1),
            "BTT4": (0, 3, 1, 1),
            "CRED1": (1, 0, 1, 2),
            "DM": (1, 2, 1, 1),
            "HDLR": (1, 3, 1, 1),
            "MDS": (2, 0, 1, 2),
            "Eng gui": (2, 2, 1, 1),
            "back_end": (2, 3, 1, 1),
        }

        def __init__(self, bind_endpoint: str) -> None:
            super().__init__()
            self.bind_endpoint = bind_endpoint
            self.formatter = StatusFormatter()
            self.last_wd_status: Any = None
            self._boxes: dict[str, ProcessStatusBox] = {}
            self._dynamic_slot_index = 0

            self._setup_ui()
            self._setup_socket()
            self._setup_timer()

        def _setup_ui(self) -> None:
            self.setWindowTitle("Watchdog Status")
            self.resize(1160, 700)

            root = QtWidgets.QVBoxLayout(self)
            root.setContentsMargins(12, 12, 12, 12)
            root.setSpacing(10)

            self.header_label = QtWidgets.QLabel("Waiting for watchdog updates...")
            self.header_label.setStyleSheet("font-weight: 700; color: #202020;")
            root.addWidget(self.header_label)

            self.grid = QtWidgets.QGridLayout()
            self.grid.setHorizontalSpacing(10)
            self.grid.setVerticalSpacing(10)
            root.addLayout(self.grid, 1)

        def _setup_socket(self) -> None:
            self.context = zmq.Context.instance()
            self.socket = self.context.socket(zmq.REP)
            self.socket.bind(self.bind_endpoint)
            self.poller = zmq.Poller()
            self.poller.register(self.socket, zmq.POLLIN)
            logging.info("Watchdog status REP endpoint bound to %s", self.bind_endpoint)

        def _setup_timer(self) -> None:
            self.timer = QtCore.QTimer(self)
            self.timer.setInterval(300)
            self.timer.timeout.connect(self._poll_once)
            self.timer.start()

        def _next_dynamic_slot(self) -> tuple[int, int, int, int]:
            row = 3 + (self._dynamic_slot_index // 4)
            col = self._dynamic_slot_index % 4
            self._dynamic_slot_index += 1
            return row, col, 1, 1

        def _get_or_create_box(self, task_name: str) -> ProcessStatusBox:
            if task_name in self._boxes:
                return self._boxes[task_name]

            box = ProcessStatusBox(task_name)
            row, col, row_span, col_span = self.PROCESS_GRID.get(
                task_name,
                self._next_dynamic_slot(),
            )
            self.grid.addWidget(box, row, col, row_span, col_span)
            self._boxes[task_name] = box
            return box

        def _render(self, wd_status: Any, update_last_time: bool) -> None:
            state = self.formatter.build_render_state(
                wd_status,
                update_last_time=update_last_time,
            )

            if not state["is_payload_dict"]:
                self.header_label.setText(html.escape(state["payload"]))
                return

            header = f"last updated {state['elapsed_seconds']:.2f} seconds ago"
            header_color = "#c62828" if state["is_stale"] else "#202020"
            self.header_label.setText(
                f"<span style='color:{header_color}'>{html.escape(header)}</span>"
            )

            seen = set()
            for task in state["tasks"]:
                task_name = task["task_name"]
                seen.add(task_name)
                box = self._get_or_create_box(task_name)
                box.update_from_task(task)
                box.show()

            for task_name, box in self._boxes.items():
                if task_name not in seen:
                    box.hide()

        def _poll_once(self) -> None:
            events = dict(self.poller.poll(timeout=0))
            if self.socket in events and events[self.socket] == zmq.POLLIN:
                message = self.socket.recv_string()
                try:
                    wd_status = json.loads(message)
                except json.JSONDecodeError:
                    wd_status = message

                print(f"Received watchdog status update at {datetime.datetime.now()}:")

                self.last_wd_status = wd_status
                self._render(wd_status, update_last_time=True)
                self.socket.send_string("ACK")
                return

            if self.last_wd_status is not None:
                self._render(self.last_wd_status, update_last_time=False)

        def keyPressEvent(self, event: Any) -> None:
            key = event.key() if hasattr(event, "key") else None
            if key == QtCore.Qt.Key_Escape:
                for widget in QtWidgets.QApplication.topLevelWidgets():
                    widget.close()
                QtCore.QCoreApplication.quit()
                event.accept()
                return
            super().keyPressEvent(event)

        def closeEvent(self, event: Any) -> None:
            self.timer.stop()
            self.poller.unregister(self.socket)
            self.socket.close()
            super().closeEvent(event)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Receive watchdog status updates from MCSClient.publish_wd()."
    )
    parser.add_argument(
        "--bind-endpoint",
        default="tcp://192.168.100.1:7051",
        help="ZMQ endpoint to bind the REP socket to.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="If the display should be a GUI instead of terminal output",
    )
    # # toggle sim option
    # parser.add_argument(
    #     "--sim",
    #     action="store_true",
    #     help="If True, simulates receiving a watchdog message every 5 seconds instead of binding a ZMQ socket.",
    # )

    args = parser.parse_args()

    # if args.sim:
    #     args.bind_endpoint = "tcp://localhost:7051"

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if args.gui:
        if QtWidgets is None or QtCore is None:
            raise ImportError("PyQt5 is required for --gui mode")

        window_cls = globals().get("WatchdogStatusWindow")
        if window_cls is None:
            raise ImportError("PyQt5 is required for --gui mode")

        app = QtWidgets.QApplication(sys.argv)
        window = cast(Any, window_cls)(args.bind_endpoint)
        getattr(window, "show")()
        sys.exit(app.exec_())

    sinfo = TextStatusInfo()
    sinfo.run_server(args.bind_endpoint)


if __name__ == "__main__":
    main()
