"""Simple ZMQ REQ client for polling watchdog status updates.

This script connects to a watchdog status endpoint and periodically sends a
request message, then renders the returned wd_status payload.
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import logging
import math
import sys
import time
from typing import Any, cast

import zmq

try:
    from PyQt5 import QtCore, QtWidgets  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional runtime dependency
    QtCore = cast(Any, None)
    QtWidgets = cast(Any, None)

# TODO: add BDS and SSF check and make into a nicer GUI that shows the layout of the instr

class StatusFormatter:
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"
    STALE_THRESHOLD_SECONDS = 10
    STATE_COLORS = {
        "default": "#d4d8e2",
        "green": "#74d99f",
        "red": "#ff7b7b",
        "yellow": "#ffd700",
    }
    fields_of_interest = {
        "BTT1": ["cnt"],
        "BTT2": ["cnt"],
        "BTT3": ["cnt"],
        "BTT4": ["cnt"],
        "BAO1": ["cnt"],
        "BAO2": ["cnt"],
        "BAO3": ["cnt"],
        "BAO4": ["cnt"],
        "CRED1": ["cam_status", "shm_error", "fps", "gain"],
        "DM": None,
        "HDLR": ["cnt", "locked"],
        "MDS": ["SDLA"],
        "back_end": [],
    }
    DISK_LABELS = {
        "cred1": "CRED1 saving",
        "ft_performance": "FT performance saving",
        "tt_performance": "TT performance saving",
    }
    TT_BEAMS = tuple(f"beam{beam}" for beam in range(1, 5))
    CRED1_STREAMS = tuple(f"baldr{beam}" for beam in range(1, 5)) + (
        "hei_k1",
        "hei_k2",
    )

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
    def _colorize_entry(cls, value: str, color: str) -> str:
        code = {"green": cls.GREEN, "red": cls.RED, "yellow": cls.YELLOW}.get(
            color
        )
        return f"{code}{value}{cls.RESET}" if code else value

    @staticmethod
    def _valid_disk_check(check: Any) -> bool:
        if not isinstance(check, dict) or check.get("state") not in ("fresh", "stale"):
            return False
        limit = check.get("limit_s")
        age = check.get("age_s")
        if not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0:
            return False
        if age is not None and (
            not isinstance(age, (int, float)) or not math.isfinite(age)
        ):
            return False
        return check["state"] != "fresh" or (age is not None and 0 <= age < limit)

    @classmethod
    def _build_disk_blocks(
        cls, disk_status: Any, disk_error: str | None
    ) -> list[dict[str, Any]]:
        blocks = []
        for source, label in cls.DISK_LABELS.items():
            group = disk_status.get(source) if isinstance(disk_status, dict) else None
            if disk_error or not isinstance(group, dict):
                state = "red"
                summary = "cannot verify"
                failures = [{"label": "check", "value": disk_error or "no reply"}]
            else:
                state = group.get("state")
                checks = group.get("checks")
                if state not in ("green", "yellow", "red") or not isinstance(
                    checks, dict
                ) or not checks or any(
                    not cls._valid_disk_check(check) for check in checks.values()
                ) or (
                    source == "tt_performance" and set(checks) != set(cls.TT_BEAMS)
                ) or (
                    source == "cred1" and set(checks) != set(cls.CRED1_STREAMS)
                ):
                    state = "red"
                    summary = "cannot verify"
                    failures = [{"label": "check", "value": "invalid reply"}]
                else:
                    fresh = sum(
                        isinstance(check, dict) and check.get("state") == "fresh"
                        for check in checks.values()
                    )
                    expected = (
                        "green" if fresh == len(checks) else "yellow" if fresh else "red"
                    )
                    if state != expected:
                        state = "red"
                        summary = "cannot verify"
                        failures = [{"label": "check", "value": "invalid summary"}]
                    else:
                        summary = f"{fresh}/{len(checks)} streams saving"
                        failures = []
                        if source == "tt_performance":
                            names = cls.TT_BEAMS
                        elif source == "cred1":
                            names = cls.CRED1_STREAMS
                        else:
                            names = checks
                        for name in names:
                            check = checks.get(name)
                            if check.get("state") == "fresh":
                                if source == "cred1":
                                    failures.append(
                                        {"label": name, "value": "saving", "color": "green"}
                                    )
                                elif source == "tt_performance":
                                    failures.append(
                                        {
                                            "label": name,
                                            "value": f"saving; last write {check['age_s']:.1f}s ago",
                                            "short_value": f"{check['age_s']:.1f}s ago",
                                            "color": "green",
                                        }
                                    )
                                continue
                            if source == "cred1":
                                detail = str(check.get("detail") or "")
                                value = "stale"
                                if detail and detail != "no recent write":
                                    value += f"; {detail}"
                                failures.append(
                                    {
                                        "label": name,
                                        "value": value,
                                        "short_value": "stale",
                                        "color": "red",
                                    }
                                )
                                continue
                            age = check.get("age_s")
                            limit = check.get("limit_s")
                            if isinstance(age, (int, float)) and math.isfinite(age):
                                age_text = (
                                    f"last write {age:.1f}s ago"
                                    if age >= 0
                                    else f"write timestamp {-age:.1f}s in future"
                                )
                            else:
                                age_text = "no write observed"
                            limit_text = (
                                f" (limit {limit:.1f}s)"
                                if isinstance(limit, (int, float)) and math.isfinite(limit)
                                else ""
                            )
                            detail = str(check.get("detail") or "")
                            if source == "tt_performance":
                                if age is None:
                                    value = f"stale; {detail or 'no write observed'}"
                                    short_value = detail or "no write"
                                else:
                                    value = f"stale; {age_text}{limit_text}"
                                    if detail and detail != "no recent write":
                                        value += f"; {detail}"
                                    short_value = (
                                        f"stale {age:.1f}s / {limit:.1f}s"
                                        if age >= 0
                                        else "future write time"
                                    )
                            else:
                                value = f"{age_text}{limit_text}; {detail}".rstrip("; ")
                            failures.append(
                                {
                                    "label": str(name),
                                    "value": value,
                                    **(
                                        {"short_value": short_value}
                                        if source == "tt_performance"
                                        else {}
                                    ),
                                    "color": "red",
                                }
                            )
            if source == "tt_performance" and summary == "cannot verify":
                summary = f"cannot verify: {failures[0]['value']}"
                failures = [
                    {"label": name, "value": "unknown", "color": "red"}
                    for name in cls.TT_BEAMS
                ]
            if source == "cred1" and summary == "cannot verify":
                summary = f"?/{len(cls.CRED1_STREAMS)} streams saving"
                failures = [
                    {"label": name, "value": "unknown", "color": "red"}
                    for name in cls.CRED1_STREAMS
                ]
            entries = [
                {"label": "disk", "value": summary, "color": state, "indent": 1}
            ]
            entries.extend(
                {"color": "red", **failure, "indent": 2} for failure in failures
            )
            blocks.append(
                {
                    "task_name": label,
                    "entries": entries,
                    "has_red": state == "red",
                    "has_yellow": state == "yellow",
                }
            )
        return blocks

    @classmethod
    def _decode_status(cls, status_payload: Any) -> Any:
        if not isinstance(status_payload, str):
            return status_payload
        try:
            json_payload = json.loads(status_payload)
            # This is a hack to incorporate Jesse's new status format, 
            # which wraps the actual status in a "data" field while
            # an error status is in a "status_code" field.
            if "data" in json_payload:
                return json_payload["data"]
            return json_payload
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

            if task_status.get("status") is not None and task_name in self.fields_of_interest:
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
                    color = (
                        "yellow"
                        if task_name == "MDS" and field == "SDLA" and self._is_sdla_error(value_str)
                        else self._state_color(value_str)
                    )
                    add_entry(field, value_str, color, indent=1)
                elif len(fields) > 1:
                    for field in fields:
                        inverse = "error" in field.lower()
                        value = (
                            decoded_status.get(field, "N/A")
                            if isinstance(decoded_status, dict)
                            else "N/A"
                        )
                        value_str = str(value)
                        if field not in ["locked"]:
                            c = self._state_color(value_str, inverse=inverse)
                        else:
                            c = "default"

                        add_entry(
                            field,
                            value_str,
                            c,
                            indent=2,
                        )
        else:
            status = str(task_status.get("status", "unknown"))
            add_entry("status", status, self._state_color(status), indent=1)

        has_red = any(entry["color"] == "red" for entry in entries)
        has_yellow = (
            task_name == "MDS"
            and any(
                entry["label"] == "SDLA"
                and (
                    self._is_zero(entry["value"])
                    or self._is_sdla_error(entry["value"])
                )
                for entry in entries
            )
        )
        return {
            "task_name": task_name,
            "entries": entries,
            "has_red": has_red,
            "has_yellow": has_yellow,
        }

    @staticmethod
    def _is_zero(value: Any) -> bool:
        try:
            return float(value) == 0.0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _is_sdla_error(value: Any) -> bool:
        return str(value) == "Error (Standby?)"

    def build_render_state(
        self,
        wd_status: Any,
        update_last_time: bool = True,
        disk_status: Any = None,
        disk_error: str | None = None,
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
        tasks.extend(self._build_disk_blocks(disk_status, disk_error))
        return {
            "is_payload_dict": True,
            "elapsed_seconds": elapsed_seconds,
            "is_stale": is_stale,
            "tasks": tasks,
        }


def add_sdla_to_mds_status(wd_status: Any, mds_endpoint: str) -> Any:
    if not isinstance(wd_status, dict) or not isinstance(wd_status.get("MDS"), dict):
        return wd_status

    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 1000)
    socket.setsockopt(zmq.SNDTIMEO, 1000)
    socket.connect(mds_endpoint)
    try:
        socket.send_string("read SDLA")
        sdla_response = socket.recv_string().strip()
    except zmq.ZMQError as error:
        logging.warning("Could not read SDLA from MDS: %s", error)
        return wd_status
    finally:
        socket.close()

    mds_status = wd_status["MDS"]
    decoded_status = StatusFormatter._decode_status(mds_status.get("status"))
    if not isinstance(decoded_status, dict):
        decoded_status = {"status": decoded_status}
    try:
        sdla_value = float(sdla_response)
        sdla = f"{sdla_value:.1f}" if math.isfinite(sdla_value) else "Error (Standby?)"
    except ValueError:
        sdla = "Error (Standby?)"
    decoded_status["SDLA"] = sdla
    mds_status["status"] = json.dumps(decoded_status)
    return wd_status


class DiskStatusClient:
    INTERVAL_SECONDS = 1.0
    TIMEOUT_SECONDS = 10.0

    def __init__(self, endpoint: str) -> None:
        self.context = zmq.Context.instance()
        self.endpoint = endpoint
        self.poller = zmq.Poller()
        self.socket = self._new_socket()
        self.poller.register(self.socket, zmq.POLLIN)
        self.last_request_at: float | None = None
        self.last_reply_at: float | None = None
        self.awaiting_reply = False
        self.payload: Any = None
        self.error: str | None = "waiting for disk status"

    def _new_socket(self) -> zmq.Socket[Any]:
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.endpoint)
        return socket

    def _reconnect(self) -> None:
        self.poller.unregister(self.socket)
        self.socket.close(linger=0)
        self.socket = self._new_socket()
        self.poller.register(self.socket, zmq.POLLIN)
        self.awaiting_reply = False

    def tick(self) -> None:
        now = time.monotonic()
        if (
            self.awaiting_reply
            and self.last_request_at is not None
            and now - self.last_request_at >= self.TIMEOUT_SECONDS
        ):
            self.error = "disk status reply overdue"
            self._reconnect()

        if not self.awaiting_reply and (
            self.last_request_at is None
            or now - self.last_request_at >= self.INTERVAL_SECONDS
        ):
            try:
                self.socket.send_string("disk_status", flags=zmq.NOBLOCK)
                self.awaiting_reply = True
                self.last_request_at = now
            except zmq.ZMQError:
                self.error = "cannot request disk status"
                self.last_request_at = now
                self._reconnect()

        events = dict(self.poller.poll(timeout=0))
        if events.get(self.socket, 0) & zmq.POLLIN:
            try:
                self.payload = json.loads(self.socket.recv_string(flags=zmq.NOBLOCK))
                self.error = None
            except (ValueError, zmq.ZMQError):
                self.payload = None
                self.error = "invalid disk status reply"
            self.last_reply_at = now
            self.awaiting_reply = False

    def result(self) -> tuple[Any, str | None]:
        if self.last_reply_at is None:
            return None, self.error
        if time.monotonic() - self.last_reply_at >= self.TIMEOUT_SECONDS:
            return None, "disk status reply overdue"
        return self.payload, self.error

    def close(self) -> None:
        self.poller.unregister(self.socket)
        self.socket.close(linger=0)


class TextStatusInfo(StatusFormatter):
    CLEAR_SCREEN = "\033[2J\033[H"

    def _clear_screen(self) -> None:
        print(self.CLEAR_SCREEN, end="", flush=True)

    def _print_watchdog_status(
        self,
        wd_status: Any,
        update_last_time: bool = True,
        disk_status: Any = None,
        disk_error: str | None = None,
    ) -> None:
        render_state = self.build_render_state(
            wd_status,
            update_last_time=update_last_time,
            disk_status=disk_status,
            disk_error=disk_error,
        )

        if not render_state["is_payload_dict"]:
            self._clear_screen()
            print(render_state["payload"], flush=True)
            return

        header = (
            f"last updated {render_state['elapsed_seconds']:.2f} seconds ago"
            if self.last_wd_time is not None
            else "waiting for watchdog status"
        )
        if render_state["is_stale"]:
            header = f"{self.RED}{header}{self.RESET}"
        print(header, flush=True)

        for task in render_state["tasks"]:
            lines = [task["task_name"]]
            for entry in task["entries"]:
                colorized_value = self._colorize_entry(
                    entry["value"], entry["color"]
                )
                indent = "  " * int(entry["indent"])
                if entry["label"]:
                    lines.append(f"{indent}{entry['label']}: {colorized_value}")
                else:
                    lines.append(f"{indent}{colorized_value}")

            print("\n".join(lines), flush=True)

    def run_server(
        self,
        connect_endpoint: str,
        mds_endpoint: str,
        request_interval_s: float = 5.0,
    ) -> None:
        reply_timeout_s = max(1.0, request_interval_s)
        context = zmq.Context.instance()

        def _new_socket() -> zmq.Socket[Any]:
            req_socket = context.socket(zmq.REQ)
            req_socket.connect(connect_endpoint)
            return req_socket

        socket = _new_socket()
        disk_client = DiskStatusClient(connect_endpoint)
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        last_wd_status: Any = {}
        awaiting_reply = False
        last_request_time: datetime.datetime | None = None

        logging.info("Watchdog status REQ endpoint connected to %s", connect_endpoint)

        while True:
            disk_client.tick()
            disk_status, disk_error = disk_client.result()
            now = datetime.datetime.now(datetime.timezone.utc)

            if (
                awaiting_reply
                and last_request_time is not None
                and (now - last_request_time).total_seconds() > reply_timeout_s
            ):
                logging.warning(
                    "No watchdog reply for %.1fs; reconnecting REQ socket",
                    reply_timeout_s,
                )
                poller.unregister(socket)
                socket.close(linger=0)
                socket = _new_socket()
                poller.register(socket, zmq.POLLIN)
                awaiting_reply = False

            should_request = not awaiting_reply and (
                last_request_time is None
                or (now - last_request_time).total_seconds() >= request_interval_s
            )

            if should_request:
                socket.send_string("status")
                last_request_time = now
                awaiting_reply = True

            events = dict(poller.poll(timeout=300))
            if socket in events and events[socket] == zmq.POLLIN:
                message = socket.recv_string()
                awaiting_reply = False
                try:
                    wd_status = json.loads(message)
                except json.JSONDecodeError:
                    wd_status = message

                valid_status = isinstance(wd_status, dict)
                if valid_status:
                    last_wd_status = add_sdla_to_mds_status(wd_status, mds_endpoint)
                else:
                    logging.warning("Invalid watchdog status reply")
                self._clear_screen()
                self._print_watchdog_status(
                    last_wd_status,
                    update_last_time=valid_status,
                    disk_status=disk_status,
                    disk_error=disk_error,
                )
                continue

            if last_wd_status is not None:
                self._clear_screen()
                self._print_watchdog_status(
                    last_wd_status,
                    update_last_time=False,
                    disk_status=disk_status,
                    disk_error=disk_error,
                )


if QtWidgets is not None:

    class ProcessStatusBox(QtWidgets.QFrame):
        def __init__(self, process_name: str) -> None:
            super().__init__()
            self._process_name = process_name
            self._opacity_effect: Any = None
            self._previous_cnt: int | None = None
            self._watchdog_yellow = False
            self._setup_ui()

        def _setup_ui(self) -> None:
            self.setObjectName("processBox")
            self.setMinimumHeight(75)
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(10, 8, 10, 8)
            layout.setSpacing(6)

            self.title = QtWidgets.QLabel(self._process_name)
            self.title.setStyleSheet("font-weight: 700; color: #f2f4f8;")
            self.details = QtWidgets.QLabel()
            self.details.setStyleSheet("color: #d4d8e2;")
            self.details.setTextFormat(QtCore.Qt.RichText)
            self.details.setWordWrap(True)
            self.details.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
            self.details.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)

            layout.addWidget(self.title)
            layout.addWidget(self.details, 1)
            self.saving_label = QtWidgets.QLabel()
            self.saving_label.setTextFormat(QtCore.Qt.PlainText)
            self.saving_label.setSizePolicy(
                QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
            )
            self.saving_label.hide()
            layout.addWidget(self.saving_label)

            self._apply_border(red_border=False)
            self._opacity_effect = QtWidgets.QGraphicsOpacityEffect(self.details)
            self.details.setGraphicsEffect(self._opacity_effect)
            self.set_dimmed(False)

        def _apply_border(
            self, red_border: bool = False, yellow_border: bool = False
        ) -> None:
            if red_border:
                border = "#ff7b7b"
                width = 2
            elif yellow_border:
                border = "#ffd700"
                width = 2
            else:
                border = "#4f5b73"
                width = 1
            self.setStyleSheet(
                "QFrame#processBox {"
                f"border: {width}px solid {border};"
                "border-radius: 6px;"
                "background-color: #141925;"
                "}"
            )

        def set_dimmed(self, is_dimmed: bool) -> None:
            if self._opacity_effect is None:
                return
            self._opacity_effect.setOpacity(0.4 if is_dimmed else 1.0)

        def update_from_task(
            self,
            task_block: dict[str, Any],
            evaluate_progress: bool = True,
            saving_status: dict[str, Any] | None = None,
        ) -> None:
            html_lines: list[str] = []
            current_cnt: int | None = None

            for entry in task_block["entries"]:
                indent_level = max(int(entry["indent"]) - 1, 0)
                indent = "&nbsp;" * (indent_level * 2)
                value_color = StatusFormatter.STATE_COLORS.get(
                    entry["color"], "#d4d8e2"
                )
                escaped_value = html.escape(str(entry.get("short_value", entry["value"])))
                escaped_label = html.escape(str(entry["label"]))

                # Track cnt value if present
                if escaped_label == "cnt":
                    try:
                        current_cnt = int(escaped_value)
                    except (ValueError, TypeError):
                        current_cnt = None

                if escaped_label:
                    html_lines.append(
                        f"{indent}<span style='color:#97a2ba'>{escaped_label}:</span> "
                        f"<span style='color:{value_color}'>{escaped_value}</span>"
                    )
                else:
                    html_lines.append(
                        f"{indent}<span style='color:{value_color}'>{escaped_value}</span>"
                    )

            self.details.setText("<br/>".join(html_lines))
            if saving_status is not None:
                self.saving_label.setText(saving_status["text"])
                self.saving_label.setToolTip(saving_status["tooltip"])
                color = "yellow" if saving_status["warning"] else "green"
                self.saving_label.setStyleSheet(
                    f"color: {StatusFormatter.STATE_COLORS[color]};"
                )
                self.saving_label.show()
            else:
                self.saving_label.hide()

            has_red = bool(task_block.get("has_red"))
            if evaluate_progress:
                # Determine border color only on fresh polled data.
                # Red takes precedence, then yellow if cnt unchanged.
                cnt_unchanged = (
                    current_cnt is not None
                    and self._previous_cnt is not None
                    and current_cnt == self._previous_cnt
                )
                self._watchdog_yellow = bool(task_block.get("has_yellow")) or cnt_unchanged
                # Update previous cnt for next comparison
                if current_cnt is not None:
                    self._previous_cnt = current_cnt
            self._apply_border(
                red_border=has_red,
                yellow_border=self._watchdog_yellow
                or bool(saving_status and saving_status["warning"]),
            )

    class WatchdogStatusWindow(QtWidgets.QWidget):
        GRID_COLUMNS = 12
        REGULAR_SPAN = 3
        SAVING_TARGETS = {
            "CRED1": ("cred1",),
            "Heim Telem": ("ft_performance", "ft_settings"),
            "Baldr TT Telem": ("tt_performance", "tt_settings"),
        }
        SAVING_STREAMS = {
            "cred1": StatusFormatter.CRED1_STREAMS,
            "ft_performance": ("FT performance",),
            "ft_settings": ("ft_settings",),
            "tt_performance": StatusFormatter.TT_BEAMS,
            "tt_settings": StatusFormatter.TT_BEAMS,
        }

        def __init__(
            self,
            connect_endpoint: str,
            mds_endpoint: str,
            request_interval_s: float = 5.0,
        ) -> None:
            super().__init__()
            self.connect_endpoint = connect_endpoint
            self.mds_endpoint = mds_endpoint
            self.request_interval_s = request_interval_s
            self.reply_timeout_s = max(1.0, request_interval_s)
            self.formatter = StatusFormatter()
            self.last_wd_status: Any = None
            self._boxes: dict[str, ProcessStatusBox] = {}
            self._awaiting_reply = False
            self._last_request_time: datetime.datetime | None = None

            self._setup_ui()
            self._setup_socket()
            self._setup_timer()

        def _setup_ui(self) -> None:
            self.setWindowTitle("Watchdog Status")
            self.resize(950, 650)
            self.setStyleSheet("QWidget { background-color: #141925; color: #dbe1ee; }")

            root = QtWidgets.QVBoxLayout(self)
            root.setContentsMargins(12, 12, 12, 12)
            root.setSpacing(10)

            self.header_label = QtWidgets.QLabel("Waiting for watchdog updates...")
            self.header_label.setStyleSheet("font-weight: 700; color: #dbe1ee;")
            root.addWidget(self.header_label)

            self.grid = QtWidgets.QGridLayout()
            self.grid.setHorizontalSpacing(10)
            self.grid.setVerticalSpacing(10)
            for column in range(self.GRID_COLUMNS):
                self.grid.setColumnStretch(column, 1)
            root.addLayout(self.grid, 1)

        def _setup_socket(self) -> None:
            self.context = zmq.Context.instance()
            self.disk_client = DiskStatusClient(self.connect_endpoint)

            def _new_socket() -> zmq.Socket[Any]:
                req_socket = self.context.socket(zmq.REQ)
                req_socket.connect(self.connect_endpoint)
                return req_socket

            self._new_socket = _new_socket
            self.socket = self._new_socket()
            self.poller = zmq.Poller()
            self.poller.register(self.socket, zmq.POLLIN)
            logging.info(
                "Watchdog status REQ endpoint connected to %s", self.connect_endpoint
            )

        def _setup_timer(self) -> None:
            self.timer = QtCore.QTimer(self)
            self.timer.setInterval(300)
            self.timer.timeout.connect(self._poll_once)
            self.timer.start()

        @staticmethod
        def _numeric_suffix(name: str) -> tuple[int, str]:
            suffix = ""
            for ch in reversed(name):
                if ch.isdigit():
                    suffix = ch + suffix
                else:
                    break
            if suffix:
                return int(suffix), name
            return sys.maxsize, name

        @classmethod
        def _saving_status(
            cls,
            sources: tuple[str, ...],
            disk_status: Any,
            disk_error: str | None,
        ) -> dict[str, Any]:
            total = sum(len(cls.SAVING_STREAMS[source]) for source in sources)
            fresh_total = 0
            unknown = False
            lines = []

            for source in sources:
                expected_names = cls.SAVING_STREAMS[source]
                group = disk_status.get(source) if isinstance(disk_status, dict) else None
                checks = group.get("checks") if isinstance(group, dict) else None
                valid = (
                    disk_error is None
                    and isinstance(checks, dict)
                    and (
                        len(checks) == 1
                        if source == "ft_performance"
                        else set(checks) == set(expected_names)
                    )
                    and all(
                        StatusFormatter._valid_disk_check(check)
                        for check in checks.values()
                    )
                )
                if valid:
                    fresh = sum(check["state"] == "fresh" for check in checks.values())
                    expected_state = (
                        "green" if fresh == len(checks) else "yellow" if fresh else "red"
                    )
                    valid = group.get("state") == expected_state

                if len(sources) > 1:
                    lines.append(
                        "Telemetry:" if source.endswith("performance") else "Settings:"
                    )
                if valid:
                    fresh_total += fresh
                    names = tuple(checks) if source == "ft_performance" else expected_names
                    for name in names:
                        check = checks[name]
                        line = f"{name}: {'saved' if check['state'] == 'fresh' else 'not saving'}"
                        age = check.get("age_s")
                        if isinstance(age, (int, float)) and math.isfinite(age):
                            if age >= 0:
                                line += f"; last write {age:.1f}s ago"
                            else:
                                line += f"; write timestamp {-age:.1f}s in future"
                        detail = check.get("detail")
                        if detail:
                            line += f"; {detail}"
                        lines.append(line)
                else:
                    unknown = True
                    reason = disk_error or (
                        "no reply" if group is None else "invalid disk status"
                    )
                    lines.append(f"Cannot verify saving: {reason}")
                    lines.extend(f"{name}: unknown" for name in expected_names)

            count = "?" if unknown else str(fresh_total)
            warning = unknown or fresh_total != total

            tooltip_text = "\n".join(lines)
            return {
                "text": f"disk: {count}/{total} streams saved",
                "tooltip": f"<pre>{html.escape(tooltip_text)}</pre>",
                "warning": warning,
            }

        def _layout_positions(
            self, task_names: list[str]
        ) -> dict[str, tuple[int, int, int, int]]:
            btt_tasks = [name for name in task_names if name.upper().startswith("BTT")]
            bao_tasks = [name for name in task_names if name.upper().startswith("BAO")]
            standard_rows = (
                (("CRED1", 4), ("DM", 3), ("MDS", 3), ("Eng gui", 2)),
                (
                    ("HDLR", 3),
                    ("Heim Telem", 3),
                    ("Baldr TT Telem", 3),
                    ("back_end", 3),
                ),
            )
            standard_names = {
                name for row in standard_rows for name, _ in row
            }
            extra_tasks = [
                name for name in task_names
                if name not in btt_tasks
                and name not in bao_tasks
                and name not in standard_names
                and name not in StatusFormatter.DISK_LABELS.values()
            ]

            btt_tasks.sort(key=self._numeric_suffix)
            bao_tasks.sort(key=self._numeric_suffix)

            positions: dict[str, tuple[int, int, int, int]] = {}
            regular_columns = self.GRID_COLUMNS // self.REGULAR_SPAN

            for idx, name in enumerate(btt_tasks):
                row = idx // regular_columns
                col = (idx % regular_columns) * self.REGULAR_SPAN
                positions[name] = (row, col, 1, self.REGULAR_SPAN)

            btt_rows = math.ceil(len(btt_tasks) / regular_columns) if btt_tasks else 0
            for idx, name in enumerate(bao_tasks):
                row = btt_rows + (idx // regular_columns)
                col = (idx % regular_columns) * self.REGULAR_SPAN
                positions[name] = (row, col, 1, self.REGULAR_SPAN)

            bao_rows = math.ceil(len(bao_tasks) / regular_columns) if bao_tasks else 0
            standard_start_row = btt_rows + bao_rows
            for row_offset, row_slots in enumerate(standard_rows):
                col = 0
                for name, span in row_slots:
                    if name in task_names:
                        positions[name] = (standard_start_row + row_offset, col, 1, span)
                    col += span

            extra_start_row = standard_start_row + len(standard_rows)
            for idx, name in enumerate(extra_tasks):
                row = extra_start_row + idx // regular_columns
                col = (idx % regular_columns) * self.REGULAR_SPAN
                positions[name] = (row, col, 1, self.REGULAR_SPAN)

            return positions

        def _get_or_create_box(self, task_name: str) -> ProcessStatusBox:
            if task_name in self._boxes:
                return self._boxes[task_name]

            box = ProcessStatusBox(task_name)
            self._boxes[task_name] = box
            return box

        def _render(
            self, wd_status: Any, update_last_time: bool, evaluate_progress: bool = True
        ) -> None:
            disk_status, disk_error = self.disk_client.result()
            state = self.formatter.build_render_state(
                wd_status,
                update_last_time=update_last_time,
                disk_status=disk_status,
                disk_error=disk_error,
            )

            if not state["is_payload_dict"]:
                self.header_label.setText(html.escape(state["payload"]))
                return

            header = (
                f"last updated {state['elapsed_seconds']:.2f} seconds ago"
                if self.formatter.last_wd_time is not None
                else "waiting for watchdog status"
            )
            header_color = "#ff7b7b" if state["is_stale"] else "#dbe1ee"
            self.header_label.setText(
                f"<span style='color:{header_color}'>{html.escape(header)}</span>"
            )

            tasks = [
                task for task in state["tasks"]
                if task["task_name"] not in StatusFormatter.DISK_LABELS.values()
            ]
            saving_statuses = {
                target: self._saving_status(
                    sources,
                    disk_status,
                    disk_error,
                )
                for target, sources in self.SAVING_TARGETS.items()
            }
            existing_names = {task["task_name"] for task in tasks}
            for target in saving_statuses:
                if target not in existing_names:
                    tasks.append(
                        {
                            "task_name": target,
                            "entries": [
                                {
                                    "label": "status",
                                    "value": "watchdog status unavailable",
                                    "color": "default",
                                    "indent": 1,
                                }
                            ],
                            "has_red": False,
                        }
                    )
            task_names = [str(task["task_name"]) for task in tasks]
            positions = self._layout_positions(task_names)

            seen = set()
            for task in tasks:
                task_name = task["task_name"]
                seen.add(task_name)
                box = self._get_or_create_box(task_name)
                box.update_from_task(
                    task,
                    evaluate_progress=evaluate_progress,
                    saving_status=saving_statuses.get(task_name),
                )
                self.grid.removeWidget(box)
                row, col, row_span, col_span = positions[task_name]
                self.grid.addWidget(box, row, col, row_span, col_span)
                box.show()

            for task_name, box in self._boxes.items():
                if task_name not in seen:
                    box.hide()

            is_stale = bool(state["is_stale"])
            for box in self._boxes.values():
                box.set_dimmed(is_stale)

        def _poll_once(self) -> None:
            self.disk_client.tick()
            now = datetime.datetime.now(datetime.timezone.utc)
            if (
                self._awaiting_reply
                and self._last_request_time is not None
                and (now - self._last_request_time).total_seconds()
                > self.reply_timeout_s
            ):
                logging.warning(
                    "No watchdog reply for %.1fs; reconnecting REQ socket",
                    self.reply_timeout_s,
                )
                self.poller.unregister(self.socket)
                self.socket.close(linger=0)
                self.socket = self._new_socket()
                self.poller.register(self.socket, zmq.POLLIN)
                self._awaiting_reply = False

            should_request = not self._awaiting_reply and (
                self._last_request_time is None
                or (now - self._last_request_time).total_seconds()
                >= self.request_interval_s
            )
            if should_request:
                self.socket.send_string("status")
                self._last_request_time = now
                self._awaiting_reply = True

            events = dict(self.poller.poll(timeout=0))
            if self.socket in events and events[self.socket] == zmq.POLLIN:
                message = self.socket.recv_string()
                self._awaiting_reply = False
                try:
                    wd_status = json.loads(message)
                except json.JSONDecodeError:
                    wd_status = message

                # print(f"Received watchdog status update at {datetime.datetime.now()}:")

                valid_status = isinstance(wd_status, dict)
                if valid_status:
                    self.last_wd_status = add_sdla_to_mds_status(
                        wd_status, self.mds_endpoint
                    )
                else:
                    logging.warning("Invalid watchdog status reply")
                self._render(
                    self.last_wd_status or {},
                    update_last_time=valid_status,
                    evaluate_progress=valid_status,
                )
                return

            if self.last_wd_status is not None:
                self._render(self.last_wd_status, update_last_time=False, evaluate_progress=False)
            else:
                self._render({}, update_last_time=False, evaluate_progress=False)

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
            self.disk_client.close()
            self.poller.unregister(self.socket)
            self.socket.close()
            super().closeEvent(event)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poll watchdog status updates from a ZMQ REQ endpoint."
    )
    parser.add_argument(
        "--endpoint",
        "--bind-endpoint",
        dest="endpoint",
        default="tcp://mimir:7019",
        help="ZMQ endpoint to connect the REQ socket to.",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=5.0,
        help="Seconds between status requests (default: 5.0).",
    )
    parser.add_argument(
        "--mds-endpoint",
        default="tcp://mimir:5555",
        help="MDS endpoint used to query the SDLA position.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=True,
        help="If the display should be a GUI instead of terminal output",
    )
    parser.add_argument(
        "--no-gui",
        dest="gui",
        action="store_false",
        help="Display watchdog status in the terminal.",
    )

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

        app = QtWidgets.QApplication([])
        window = cast(Any, window_cls)(
            args.endpoint, args.mds_endpoint, args.request_interval
        )
        getattr(window, "show")()
        sys.exit(app.exec_())

    sinfo = TextStatusInfo()
    sinfo.run_server(args.endpoint, args.mds_endpoint, args.request_interval)


if __name__ == "__main__":
    main()
