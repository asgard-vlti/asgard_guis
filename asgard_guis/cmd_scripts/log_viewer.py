import html
import os
import re
import signal
import subprocess
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

DEFAULT_LOG_ROOT = "/home/asg/logs"
TAIL_LINES = 100
REFRESH_MS = 1000

SERVER_DIRS = [
    "cam_server",
    "DM_server",
    "heimdallr",
]

ANSI_PATTERN = re.compile(r"\x1b\[([0-9;]*)m")
ANSI_STRIP_PATTERN = re.compile(r"\x1b\[[0-9;]*m")

FG_COLORS = {
    30: "#000000",
    31: "#cd0000",
    32: "#00cd00",
    33: "#cdcd00",
    34: "#0000ee",
    35: "#cd00cd",
    36: "#00cdcd",
    37: "#e5e5e5",
    90: "#7f7f7f",
    91: "#ff0000",
    92: "#00ff00",
    93: "#ffff00",
    94: "#5c5cff",
    95: "#ff00ff",
    96: "#00ffff",
    97: "#ffffff",
}

BG_COLORS = {
    40: "#000000",
    41: "#cd0000",
    42: "#00cd00",
    43: "#cdcd00",
    44: "#0000ee",
    45: "#cd00cd",
    46: "#00cdcd",
    47: "#e5e5e5",
    100: "#7f7f7f",
    101: "#ff0000",
    102: "#00ff00",
    103: "#ffff00",
    104: "#5c5cff",
    105: "#ff00ff",
    106: "#00ffff",
    107: "#ffffff",
}


class HistoryLineEdit(QtWidgets.QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = []
        self.history_index = -1

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            parent = self.parent()
            if hasattr(parent, "apply_filter"):
                parent.apply_filter()
                return

        if event.key() == QtCore.Qt.Key_Up:
            if self.history:
                if self.history_index == -1:
                    self.history_index = len(self.history) - 1
                elif self.history_index > 0:
                    self.history_index -= 1
                self.setText(self.history[self.history_index])
                self.setCursorPosition(len(self.text()))
        elif event.key() == QtCore.Qt.Key_Down:
            if self.history:
                if self.history_index != -1 and self.history_index < len(self.history) - 1:
                    self.history_index += 1
                    self.setText(self.history[self.history_index])
                    self.setCursorPosition(len(self.text()))
                else:
                    self.history_index = -1
                    self.clear()
        else:
            super().keyPressEvent(event)

    def add_history(self, value):
        if value and (not self.history or self.history[-1] != value):
            self.history.append(value)
        self.history_index = -1


def strip_ansi(text):
    return ANSI_STRIP_PATTERN.sub("", text)


def _apply_ansi_code(state, code):
    if code == 0:
        state["bold"] = False
        state["underline"] = False
        state["fg"] = None
        state["bg"] = None
    elif code == 1:
        state["bold"] = True
    elif code == 22:
        state["bold"] = False
    elif code == 4:
        state["underline"] = True
    elif code == 24:
        state["underline"] = False
    elif code == 39:
        state["fg"] = None
    elif code == 49:
        state["bg"] = None
    elif code in FG_COLORS:
        state["fg"] = FG_COLORS[code]
    elif code in BG_COLORS:
        state["bg"] = BG_COLORS[code]


def _style_for_state(state):
    styles = []
    if state["fg"]:
        styles.append(f"color: {state['fg']}")
    if state["bg"]:
        styles.append(f"background-color: {state['bg']}")
    if state["bold"]:
        styles.append("font-weight: bold")
    if state["underline"]:
        styles.append("text-decoration: underline")
    return "; ".join(styles)


def ansi_to_html(text):
    state = {"bold": False, "underline": False, "fg": None, "bg": None}
    output = []
    last = 0

    for match in ANSI_PATTERN.finditer(text):
        if match.start() > last:
            chunk = html.escape(text[last:match.start()])
            style = _style_for_state(state)
            if style:
                output.append(f'<span style="{style}">{chunk}</span>')
            else:
                output.append(chunk)

        code_text = match.group(1)
        if code_text == "":
            codes = [0]
        else:
            codes = []
            for item in code_text.split(";"):
                if item == "":
                    continue
                try:
                    codes.append(int(item))
                except ValueError:
                    continue
            if not codes:
                codes = [0]

        for code in codes:
            _apply_ansi_code(state, code)

        last = match.end()

    if last < len(text):
        chunk = html.escape(text[last:])
        style = _style_for_state(state)
        if style:
            output.append(f'<span style="{style}">{chunk}</span>')
        else:
            output.append(chunk)

    return "".join(output)


def read_last_lines(path, line_count=100):
    if line_count <= 0:
        return []

    with open(path, "rb") as file_obj:
        file_obj.seek(0, os.SEEK_END)
        file_size = file_obj.tell()
        if file_size == 0:
            return []

        block_size = 8192
        blocks = []
        bytes_to_read = 0

        while file_size > 0 and bytes_to_read < file_size:
            read_size = min(block_size, file_size - bytes_to_read)
            bytes_to_read += read_size
            file_obj.seek(file_size - bytes_to_read)
            blocks.append(file_obj.read(read_size))
            if b"\n".join(reversed(blocks)).count(b"\n") >= line_count + 1:
                break

    data = b"".join(reversed(blocks))
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return lines[-line_count:]


class LogTab(QtWidgets.QWidget):
    def __init__(
        self,
        tab_name,
        relative_log_dir,
        log_root,
        parent=None,
        with_baldr_selector=False,
    ):
        super().__init__(parent)
        self.tab_name = tab_name
        self.relative_log_dir = relative_log_dir
        self.log_root = log_root
        self.with_baldr_selector = with_baldr_selector

        self.filter_text = ""
        self.last_rendered_key = None
        self.initial_scroll_done = False

        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()

        self.filter_input = HistoryLineEdit(self)
        self.filter_input.setPlaceholderText("Filter log text (case-insensitive)")
        self.apply_filter_button = QtWidgets.QPushButton("Apply Filter", self)
        self.clear_filter_button = QtWidgets.QPushButton("Clear", self)
        self.kill_button = QtWidgets.QPushButton("Kill", self)
        self.restart_button = QtWidgets.QPushButton("Restart", self)

        controls.addWidget(QtWidgets.QLabel("Filter:", self))
        controls.addWidget(self.filter_input)
        controls.addWidget(self.apply_filter_button)
        controls.addWidget(self.clear_filter_button)
        controls.addWidget(self.kill_button)
        controls.addWidget(self.restart_button)

        self.instance_dropdown = None
        if self.with_baldr_selector:
            self.instance_dropdown = QtWidgets.QComboBox(self)
            self.instance_dropdown.addItems(["1", "2", "3", "4"])
            controls.addWidget(QtWidgets.QLabel("Instance:", self))
            controls.addWidget(self.instance_dropdown)

        self.text_area = QtWidgets.QTextEdit(self)
        self.text_area.setReadOnly(True)
        self.text_area.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)

        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.text_area.setFont(font)

        layout.addLayout(controls)
        layout.addWidget(self.text_area)

        self.apply_filter_button.clicked.connect(self.apply_filter)
        self.clear_filter_button.clicked.connect(self.clear_filter)
        self.kill_button.clicked.connect(self.kill_server)
        self.restart_button.clicked.connect(self.restart_server)
        self.filter_input.textChanged.connect(self.on_filter_text_changed)

        if self.instance_dropdown is not None:
            self.instance_dropdown.currentIndexChanged.connect(self.refresh_log)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(REFRESH_MS)
        self.timer.timeout.connect(self.refresh_log)
        self.timer.start()

        self.refresh_log()
        QtCore.QTimer.singleShot(0, self.scroll_to_bottom)

    def on_filter_text_changed(self, _):
        self.refresh_log()

    def apply_filter(self):
        self.filter_text = self.filter_input.text().strip()
        self.filter_input.add_history(self.filter_text)
        self.refresh_log(force=True)

    def clear_filter(self):
        self.filter_input.clear()
        self.filter_text = ""
        self.refresh_log(force=True)

    def log_path(self):
        if self.with_baldr_selector and self.instance_dropdown is not None:
            instance = self.instance_dropdown.currentText()
            return os.path.join(self.log_root, self.relative_log_dir, instance, "log.txt")
        return os.path.join(self.log_root, self.relative_log_dir, "log.txt")

    def server_key(self):
        if self.with_baldr_selector and self.instance_dropdown is not None:
            return f"{self.relative_log_dir}.{self.instance_dropdown.currentText()}"
        return self.relative_log_dir

    def restart_command(self):
        if self.with_baldr_selector and self.instance_dropdown is not None:
            return f"run_{self.relative_log_dir} {self.instance_dropdown.currentText()}"
        return f"run_{self.relative_log_dir}"

    def append_action_line(self, message):
        timestamp = QtCore.QDateTime.currentDateTimeUtc().toString("yyyy-MM-ddTHH:mm:ss'Z'")
        control_line = f"{timestamp} [CONTROL] {message}"
        path = self.log_path()

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as log_file:
                log_file.write(control_line + "\n")
        except OSError:
            # Keep UI behavior resilient even if appending to the log file fails.
            pass

    def scroll_to_bottom(self):
        vbar = self.text_area.verticalScrollBar()
        vbar.setValue(vbar.maximum())

    def confirm_action(self, title, message):
        result = QtWidgets.QMessageBox.question(
            self,
            title,
            message,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        return result == QtWidgets.QMessageBox.Yes

    def kill_server(self):
        if not self.confirm_action(
            "Confirm Kill",
            f"Kill {self.server_key()} now?",
        ):
            self.append_action_line(f"Kill cancelled for {self.server_key()}.")
            self.refresh_log(force=True)
            return

        lock_path = f"/tmp/asg.{self.server_key()}.lock"
        pid = None

        try:
            with open(lock_path, "r", encoding="utf-8") as lock_file:
                content = lock_file.read().strip()
        except OSError:
            content = ""

        if content:
            try:
                pid = int(content)
            except ValueError:
                pid = None

        if pid is None:
            self.append_action_line(f"No PID found for {self.server_key()}.")
            self.refresh_log(force=True)
            return

        try:
            os.kill(pid, signal.SIGKILL)
            self.append_action_line(f"Killed {self.server_key()} (PID {pid}).")
        except ProcessLookupError:
            self.append_action_line(
                f"No PID found for {self.server_key()} (stale lock PID {pid})."
            )
        except PermissionError as exc:
            self.append_action_line(
                f"Failed to kill {self.server_key()} (PID {pid}): {exc}"
            )
        except OSError as exc:
            self.append_action_line(
                f"Failed to kill {self.server_key()} (PID {pid}): {exc}"
            )

        self.refresh_log(force=True)

    def restart_server(self):
        cmd = self.restart_command()

        try:
            subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self.append_action_line(f"Restarted {self.server_key()} with '{cmd}'.")
        except OSError as exc:
            self.append_action_line(
                f"Failed to restart {self.server_key()} with '{cmd}': {exc}"
            )

        self.refresh_log(force=True)

    def refresh_log(self, force=False):
        # Keep the view anchored at the bottom only when user is already at the bottom.
        vbar = self.text_area.verticalScrollBar()
        was_at_bottom = vbar.value() >= max(0, vbar.maximum() - 2)

        path = self.log_path()
        filter_text = self.filter_input.text().strip()

        if not os.path.exists(path):
            if force or self.last_rendered_key != ("missing", path):
                self.text_area.setPlainText(f"Log file not found:\n{path}")
                self.last_rendered_key = ("missing", path)
            return

        try:
            lines = read_last_lines(path, TAIL_LINES)
        except OSError as exc:
            key = ("error", path, str(exc))
            if force or self.last_rendered_key != key:
                self.text_area.setPlainText(f"Failed reading log file:\n{path}\n\n{exc}")
                self.last_rendered_key = key
            return

        if filter_text:
            needle = filter_text.lower()
            lines = [line for line in lines if needle in strip_ansi(line).lower()]

        joined_text = "\n".join(lines)
        render_key = (path, filter_text, hash(joined_text))
        if not force and render_key == self.last_rendered_key:
            return

        html_text = ansi_to_html(joined_text)
        wrapped = (
            "<div style=\"font-family: monospace; white-space: pre;\">"
            f"{html_text}"
            "</div>"
        )
        self.text_area.setHtml(wrapped)
        self.last_rendered_key = render_key

        if was_at_bottom:
            vbar = self.text_area.verticalScrollBar()
            vbar.setValue(vbar.maximum())

        if not self.initial_scroll_done:
            self.scroll_to_bottom()
            self.initial_scroll_done = True


class UniversalLogClient(QtWidgets.QMainWindow):
    def __init__(self, log_root, parent=None):
        super().__init__(parent)
        self.log_root = log_root
        self.setWindowTitle(f"Asgard log viewer: {self.log_root}")
        self.resize(1000, 700)

        self.tabs = QtWidgets.QTabWidget(self)
        self.setCentralWidget(self.tabs)

        for server_name in SERVER_DIRS:
            tab = LogTab(server_name, server_name, self.log_root)
            self.tabs.addTab(tab, server_name)

        baldr_tab = LogTab("baldr_tt", "baldr_tt", self.log_root, with_baldr_selector=True)
        self.tabs.addTab(baldr_tab, "baldr_tt")

        self.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress:
            key = event.key()
            modifiers = event.modifiers()

            if key == QtCore.Qt.Key_Escape:
                self.close()
                return True

            if modifiers & QtCore.Qt.ControlModifier:
                if QtCore.Qt.Key_1 <= key <= QtCore.Qt.Key_9:
                    tab_idx = key - QtCore.Qt.Key_1
                    if tab_idx < self.tabs.count():
                        self.tabs.setCurrentIndex(tab_idx)
                        return True

        return super().eventFilter(obj, event)


def main():
    if len(sys.argv) == 1:
        log_root = DEFAULT_LOG_ROOT
    elif len(sys.argv) == 2:
        log_root = sys.argv[1]
    else:
        print("Usage: python log_viewer.py <log_root>")
        sys.exit(1)

    app = QtWidgets.QApplication(sys.argv)
    client = UniversalLogClient(log_root)
    client.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
