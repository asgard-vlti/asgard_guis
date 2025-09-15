import sys
import zmq
import json

from PyQt5 import QtWidgets, QtGui, QtCore


sockets = [
    ("cam_server", 6667),
    ("DM_server", 6666),
    ("heimdallr", 6660),
    ("baldr1", 6662),
    ("baldr2", 6663),
    ("baldr3", 6664),
    ("baldr4", 6665),
]


class HistoryLineEdit(QtWidgets.QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = []
        self.history_index = -1

    def event(self, event):
        # Intercept Tab key at the event level to prevent focus change
        if event.type() == QtCore.QEvent.KeyPress and event.key() == QtCore.Qt.Key_Tab:
            parent = self.parent()
            if hasattr(parent, "command_dropdown"):
                current_text = self.text()
                dropdown = parent.command_dropdown
                matches = []
                for i in range(dropdown.count()):
                    cmd = dropdown.itemText(i)
                    if cmd.startswith(current_text):
                        matches.append(cmd)
                if matches:
                    if len(matches) == 1:
                        self.setText(matches[0])
                    else:
                        prefix = matches[0]
                        for m in matches[1:]:
                            i = 0
                            while i < len(prefix) and i < len(m) and prefix[i] == m[i]:
                                i += 1
                            prefix = prefix[:i]
                        self.setText(prefix)
                    self.setCursorPosition(len(self.text()))
                    if len(matches) > 1:
                        dropdown.showPopup()
                # Do not propagate Tab further (prevents focus change)
                return True
        return super().event(event)

    def keyPressEvent(self, event):
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
                if (
                    self.history_index != -1
                    and self.history_index < len(self.history) - 1
                ):
                    self.history_index += 1
                    self.setText(self.history[self.history_index])
                    self.setCursorPosition(len(self.text()))
                else:
                    self.history_index = -1
                    self.clear()
        else:
            super().keyPressEvent(event)

    def add_history(self, cmd):
        if cmd and (not self.history or self.history[-1] != cmd):
            self.history.append(cmd)
        self.history_index = -1


class ServerTab(QtWidgets.QWidget):
    def __init__(self, server_name, zmq_socket, parent=None):
        super().__init__(parent)
        self.server_name = server_name
        self.zmq_socket = zmq_socket
        self.parent_window = parent  # Not used, but could be for future

        layout = QtWidgets.QVBoxLayout(self)
        self.command_dropdown = QtWidgets.QComboBox(self)
        self.command_dropdown.setEditable(False)
        self.command_dropdown.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.input_line = HistoryLineEdit(self)
        self.send_button = QtWidgets.QPushButton("Send", self)
        self.text_area = QtWidgets.QTextEdit(self)
        self.text_area.setReadOnly(True)

        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.command_dropdown)
        hlayout.addWidget(self.input_line)
        hlayout.addWidget(self.send_button)

        layout.addLayout(hlayout)
        layout.addWidget(self.text_area)

        self.send_button.clicked.connect(self.send_command)
        self.input_line.returnPressed.connect(self.send_command)
        self.command_dropdown.activated[str].connect(self.set_input_line)

    def set_input_line(self, cmd):
        self.input_line.setText(cmd)
        # Fetch and display arguments for the selected command
        self.display_command_arguments(cmd)

    def display_command_arguments(self, cmd):
        # Send 'arguments "<cmd>"' to the server and display the result as a table
        try:
            self.zmq_socket.send_string(f'arguments "{cmd}"')
            reply = self.zmq_socket.recv_string()
        except zmq.error.Again:
            self.append_colored(
                "[Error] ZMQ request timed out while fetching arguments."
            )
            return
        except zmq.error.ZMQError as e:
            if "Operation cannot be accomplished in current state" in str(e):
                self.append_colored(
                    "[Warning] Socket out of state, attempting to reconnect for arguments..."
                )
                self.reconnect_socket()
                try:
                    self.zmq_socket.send_string(f'arguments "{cmd}"')
                    reply = self.zmq_socket.recv_string()
                except Exception:
                    self.append_colored(
                        "[Error] Could not fetch arguments after reconnect."
                    )
                    return
            else:
                self.append_colored(f"[Error] {e}")
                return
        except Exception as e:
            self.append_colored(f"[Error] {e}")
            return

        # Try to parse the reply and display as a table
        try:
            args = json.loads(reply)
            if args is None:
                self.text_area.append("No arguments required.")
            elif isinstance(args, list) and args and isinstance(args[0], dict):
                # Display as a table
                table = "<table border='1' cellspacing='0' cellpadding='2'><tr><th>Name</th><th>Type</th></tr>"
                for arg in args:
                    name = arg.get("name", "")
                    typ = arg.get("type", "")
                    table += f"<tr><td>{name}</td><td>{typ}</td></tr>"
                table += "</table>"
                self.text_area.append(table)
            else:
                self.append_colored(str(args))
        except Exception:
            # Fallback: just show the reply
            self.append_colored(reply.replace("\\n", "\n"))

    def reconnect_socket(self):
        # Recreate the socket and reconnect
        try:
            context = self.zmq_socket.context
            # Get the endpoint from the socket's last connection
            last_endpoint = self.zmq_socket.getsockopt(zmq.LAST_ENDPOINT).decode()
            self.zmq_socket.close(linger=0)
            new_socket = context.socket(zmq.REQ)
            new_socket.setsockopt(zmq.SNDTIMEO, 2000)
            new_socket.setsockopt(zmq.RCVTIMEO, 2000)
            new_socket.connect(last_endpoint)
            self.zmq_socket = new_socket
        except Exception as e:
            self.text_area.append(f"[Error] Could not reconnect socket: {e}")

    def send_command(self):
        cmd = self.input_line.text().strip()
        if not cmd:
            return
        # Check if the command exists in the dropdown
        command_set = set(
            self.command_dropdown.itemText(i)
            for i in range(self.command_dropdown.count())
        )
        cmd_name = cmd.split(" ", 1)[0] if " " in cmd else cmd
        if cmd_name not in command_set and cmd_name not in ["exit"]:
            self.append_colored(f"[Error] Command '{cmd_name}' not recognized.")
            self.input_line.clear()
            self.text_area.moveCursor(QtGui.QTextCursor.End)
            return
        self.input_line.add_history(cmd)
        self.text_area.append(f"> {cmd}")
        try:
            self.zmq_socket.send_string(cmd)
            reply = self.zmq_socket.recv_string()
        except zmq.error.Again:
            reply = "[Error] ZMQ request timed out."
        except zmq.error.ZMQError as e:
            if "Operation cannot be accomplished in current state" in str(e):
                self.append_colored(
                    "[Warning] Socket out of state, attempting to reconnect..."
                )
                self.reconnect_socket()
                try:
                    self.zmq_socket.send_string(cmd)
                    reply = self.zmq_socket.recv_string()
                except Exception as e2:
                    reply = f"[Error] After reconnect: {e2}"
            else:
                reply = f"[Error] {e}"
        except Exception as e:
            reply = f"[Error] {e}"
        # Try to parse as JSON and pretty-print, else just show with \n expanded
        try:
            resp = json.loads(reply)
            if isinstance(resp, dict):
                pretty = json.dumps(resp, indent=4)
                self.text_area.append(pretty)
            else:
                self.append_colored(str(resp).replace("\\n", "\n"))
        except json.JSONDecodeError:
            self.append_colored(reply.replace("\\n", "\n"))
        self.input_line.clear()
        # Scroll to the bottom after new output
        self.text_area.moveCursor(QtGui.QTextCursor.End)

    def append_colored(self, text):
        # If 'error' (case-insensitive) is in the text, show in red, else normal
        import html

        if "error" in text.lower():
            html_text = f'<span style="color:red;">{html.escape(text)}</span>'
            self.text_area.moveCursor(QtGui.QTextCursor.End)
            self.text_area.insertHtml(html_text + "<br>")
        elif "warning" in text.lower():
            html_text = f'<span style="color:orange;">{html.escape(text)}</span>'
            self.text_area.moveCursor(QtGui.QTextCursor.End)
            self.text_area.insertHtml(html_text + "<br>")
        else:
            self.text_area.append(text)

    def populate_commands(self):
        # Send "command_names" and populate the dropdown
        self.command_dropdown.clear()
        try:
            self.zmq_socket.send_string("command_names")
            reply = self.zmq_socket.recv_string()
        except zmq.error.Again:
            self.text_area.append(
                "[Error] ZMQ request timed out while fetching commands."
            )
            return
        except zmq.error.ZMQError as e:
            if "Operation cannot be accomplished in current state" in str(e):
                self.text_area.append(
                    "[Warning] Socket out of state, attempting to reconnect for commands..."
                )
                self.reconnect_socket()
                try:
                    self.zmq_socket.send_string("command_names")
                    reply = self.zmq_socket.recv_string()
                except Exception:
                    return
            else:
                return
        except Exception:
            return
        try:
            commands = json.loads(reply)
            if isinstance(commands, list):
                commands = sorted(commands, key=str.lower)
                self.command_dropdown.addItems(commands)
        except Exception:
            try:
                commands = eval(reply)
                if isinstance(commands, list):
                    commands = sorted([str(c) for c in commands], key=str.lower)
                    self.command_dropdown.addItems(commands)
            except Exception:
                pass


class UniversalClient(QtWidgets.QMainWindow):
    def __init__(self, ip_addr, servers, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Asgard DCS text interface: {ip_addr}")
        self.resize(700, 500)
        self.tabs = QtWidgets.QTabWidget(self)
        self.setCentralWidget(self.tabs)
        self.context = zmq.Context()
        self.sockets = []
        self.tab_widgets = []

        for name, port in servers:
            socket = self.context.socket(zmq.REQ)
            socket.connect(f"tcp://{ip_addr}:{port}")
            # Set ZMQ send/recv timeouts (milliseconds)
            socket.setsockopt(zmq.SNDTIMEO, 2000)
            socket.setsockopt(zmq.RCVTIMEO, 2000)
            self.sockets.append(socket)
            tab = ServerTab(name, socket)
            self.tabs.addTab(tab, name)
            self.tab_widgets.append(tab)

        self.tabs.currentChanged.connect(self.on_tab_changed)
        # Populate commands for all tabs once at startup
        for tab in self.tab_widgets:
            tab.populate_commands()

        # Install event filter for Ctrl+Number tab switching
        self.installEventFilter(self)

        # Move window to bottom left corner of available screen
        screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        win_width = self.width()
        win_height = self.height()
        win_x = screen_geometry.left()
        win_y = screen_geometry.bottom() - win_height
        self.move(win_x, win_y)

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress:
            key = event.key()
            modifiers = event.modifiers()
            # Qt.Key_1 is 0x31, Qt.Key_9 is 0x39
            if modifiers & QtCore.Qt.ControlModifier:
                if QtCore.Qt.Key_1 <= key <= QtCore.Qt.Key_9:
                    tab_idx = key - QtCore.Qt.Key_1
                    if tab_idx < self.tabs.count():
                        self.tabs.setCurrentIndex(tab_idx)
                        # Set focus to the input line of the selected tab
                        tab = self.tab_widgets[tab_idx]
                        if hasattr(tab, "input_line"):
                            tab.input_line.setFocus(QtCore.Qt.TabFocusReason)
                        return True
        return super().eventFilter(obj, event)

    def on_tab_changed(self, idx):
        # No longer repopulate commands or clear history on tab change
        pass


def main():
    if len(sys.argv) != 2:
        print("Usage: python universal_client.py <ip_address>")
        sys.exit(1)
    ip_addr = sys.argv[1]
    servers = sockets
    app = QtWidgets.QApplication(sys.argv)
    client = UniversalClient(ip_addr, servers)
    client.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
