import sys
import zmq
import json

from PyQt5 import QtWidgets, QtGui, QtCore
import html

sockets = [
    ("cam_server", [6667]),
    ("DM_server", [6666]),
    ("heimdallr", [6660]),
    ("MDS", [5555]),
    ("baldr", [6662, 6663, 6664, 6665]),
    ("baldr_tt", [6671, 6672, 6673, 6674]),
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
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            parent = self.parent()
            if hasattr(parent, "send_command") and hasattr(parent, "send_all_commands"):
                if event.modifiers() & QtCore.Qt.ShiftModifier:
                    parent.send_all_commands()
                else:
                    parent.send_command()
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
    def __init__(self, server_name, zmq_sockets, parent=None, rcv_timeout=1500):
        super().__init__(parent)
        self.server_name = server_name
        self.zmq_sockets = zmq_sockets
        self.active_socket_index = 0
        self.zmq_socket = zmq_sockets[0]  # Use the first socket for commands
        self.rcv_timeout = rcv_timeout
        self.parent_window = parent  # Not used, but could be for future

        layout = QtWidgets.QVBoxLayout(self)
        self.command_dropdown = QtWidgets.QComboBox(self)
        self.command_dropdown.setEditable(False)
        self.command_dropdown.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.input_line = HistoryLineEdit(self)
        self.send_button = QtWidgets.QPushButton("Send", self)
        self.send_all_button = QtWidgets.QPushButton("Send All", self)
        self.send_button.setFixedWidth(int(self.send_button.sizeHint().width() * 0.8))
        self.send_all_button.setFixedWidth(
            int(self.send_all_button.sizeHint().width() * 0.9)
        )
        self.socket_dropdown = QtWidgets.QComboBox(self)
        self.socket_dropdown.setEditable(False)
        self.socket_dropdown.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        if len(self.zmq_sockets) == 1:
            self.send_all_button.setEnabled(False)
            self.socket_dropdown.addItem("")
            self.socket_dropdown.setEnabled(False)
        else:
            self.socket_dropdown.addItems(
                [str(i) for i in range(1, len(self.zmq_sockets) + 1)]
            )
        self.text_area = QtWidgets.QTextEdit(self)
        self.text_area.setReadOnly(True)

        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.command_dropdown)
        hlayout.addWidget(self.input_line)
        hlayout.addWidget(self.send_button)
        hlayout.addWidget(self.send_all_button)
        hlayout.addWidget(self.socket_dropdown)

        layout.addLayout(hlayout)
        layout.addWidget(self.text_area)

        self.send_button.clicked.connect(self.send_command)
        self.send_all_button.clicked.connect(self.send_all_commands)
        self.input_line.returnPressed.connect(self.send_command)
        self.command_dropdown.activated[str].connect(self.set_input_line)
        self.socket_dropdown.currentIndexChanged.connect(self.set_active_socket)

    def set_active_socket(self, idx):
        if 0 <= idx < len(self.zmq_sockets):
            self.active_socket_index = idx
            self.zmq_socket = self.zmq_sockets[idx]

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
            new_socket.setsockopt(zmq.SNDTIMEO, 1500)
            new_socket.setsockopt(zmq.RCVTIMEO, self.rcv_timeout)
            new_socket.connect(last_endpoint)
            self.zmq_socket = new_socket
            if 0 <= self.active_socket_index < len(self.zmq_sockets):
                self.zmq_sockets[self.active_socket_index] = new_socket
        except Exception as e:
            self.text_area.append(f"[Error] Could not reconnect socket: {e}")

    def reconnect_socket_by_index(self, idx):
        try:
            old_socket = self.zmq_sockets[idx]
            context = old_socket.context
            last_endpoint = old_socket.getsockopt(zmq.LAST_ENDPOINT).decode()
            old_socket.close(linger=0)
            new_socket = context.socket(zmq.REQ)
            new_socket.setsockopt(zmq.SNDTIMEO, 1500)
            new_socket.setsockopt(zmq.RCVTIMEO, self.rcv_timeout)
            new_socket.connect(last_endpoint)
            self.zmq_sockets[idx] = new_socket
            if idx == self.active_socket_index:
                self.zmq_socket = new_socket
            return new_socket
        except Exception as e:
            self.append_colored(f"[Error] Could not reconnect socket {idx + 1}: {e}")
            return None

    def send_and_receive(self, socket_obj, cmd):
        socket_obj.send_string(cmd)
        return socket_obj.recv_string(), False

    def send_and_receive_with_reconnect(self, idx, cmd):
        socket_obj = self.zmq_sockets[idx]
        try:
            reply, error_occurred = self.send_and_receive(socket_obj, cmd)
        except zmq.error.Again:
            reply = "[Error] ZMQ request timed out."
            error_occurred = True
        except zmq.error.ZMQError as e:
            if "Operation cannot be accomplished in current state" in str(e):
                self.append_colored(
                    f"[Warning] Socket {idx + 1} out of state, attempting to reconnect..."
                )
                socket_obj = self.reconnect_socket_by_index(idx)
                if socket_obj is None:
                    return "[Error] Could not reconnect socket.", True
                try:
                    reply, error_occurred = self.send_and_receive(socket_obj, cmd)
                except Exception as e2:
                    reply = f"[Error] After reconnect: {e2}"
                    error_occurred = True
            else:
                reply = f"[Error] {e}"
                error_occurred = True
        except Exception as e:
            reply = f"[Error] {e}"
            error_occurred = True
        return reply, error_occurred

    def append_response(self, reply):
        try:
            resp = json.loads(reply)
            if isinstance(resp, dict):
                pretty = json.dumps(resp, indent=4)
                self.text_area.append(pretty)
            else:
                self.append_colored(str(resp).replace("\\n", "\n"))
        except json.JSONDecodeError:
            self.append_colored(reply.replace("\\n", "\n"))

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
            self.text_area.moveCursor(QtGui.QTextCursor.End)
            return
        self.input_line.add_history(cmd)
        if len(self.zmq_sockets) <= 1:
            self.text_area.append(f"> {cmd}")
        else:
            self.text_area.append(f"> [{self.active_socket_index + 1}] {cmd}")
        reply, error_occurred = self.send_and_receive_with_reconnect(
            self.active_socket_index, cmd
        )
        self.append_response(reply)
        if not error_occurred:
            self.input_line.clear()
        # Scroll to the bottom after new output
        self.text_area.moveCursor(QtGui.QTextCursor.End)

    def send_all_commands(self):
        if len(self.zmq_sockets) <= 1:
            return
        cmd = self.input_line.text().strip()
        if not cmd:
            return
        command_set = set(
            self.command_dropdown.itemText(i)
            for i in range(self.command_dropdown.count())
        )
        cmd_name = cmd.split(" ", 1)[0] if " " in cmd else cmd
        if cmd_name not in command_set and cmd_name not in ["exit"]:
            self.append_colored(f"[Error] Command '{cmd_name}' not recognized.")
            self.text_area.moveCursor(QtGui.QTextCursor.End)
            return

        self.input_line.add_history(cmd)
        self.text_area.append(f"> [All] {cmd}")
        any_error = False
        for idx in range(len(self.zmq_sockets)):
            reply, error_occurred = self.send_and_receive_with_reconnect(idx, cmd)
            self.text_area.append(f"Beam {idx + 1} reply:")
            self.append_response(reply)
            if error_occurred:
                any_error = True

        if not any_error:
            self.input_line.clear()
        self.text_area.moveCursor(QtGui.QTextCursor.End)

    def append_colored(self, text):
        # If 'error' (case-insensitive) is in the text, show in red, else normal

        # Always insert a new line before colored messages for clarity
        if "error" in text.lower():
            self.text_area.append("")
            html_text = f'<span style="color:red;">{html.escape(text)}</span>'
            self.text_area.moveCursor(QtGui.QTextCursor.End)
            self.text_area.insertHtml(html_text + "<br>")
        elif "warning" in text.lower():
            self.text_area.append("")
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
            print(f"Received command_names reply: {reply}")
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
        except Exception as e:
            print(f"Unexpected error occurred: {e}")
            return
        try:
            commands = json.loads(reply)
            if isinstance(commands, list):
                commands = sorted(commands, key=str.lower)
                self.command_dropdown.addItems(commands)
                print(f"Populated commands: {commands}")
        except Exception as e:
            print(
                f"Failed to parse command_names reply {self.server_name} as JSON: {e}"
            )
            try:
                commands = eval(reply)
                if isinstance(commands, list):
                    commands = sorted([str(c) for c in commands], key=str.lower)
                    self.command_dropdown.addItems(commands)
            except Exception as e:
                print(f"Failed to evaluate command_names reply {self.server_name}: {e}")
                pass


class UniversalClient(QtWidgets.QMainWindow):
    def __init__(self, ip_addr, servers, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Asgard DCS text interface: {ip_addr}")
        self.resize(700, 500)
        self.tabs = QtWidgets.QTabWidget(self)
        self.setCentralWidget(self.tabs)
        self.context = zmq.Context()
        self.tab_widgets = []

        for name, ports in servers:
            sockets = []
            for port in ports:
                socket = self.context.socket(zmq.REQ)
                socket.connect(f"tcp://{ip_addr}:{port}")
                sockets.append(socket)
                # Set ZMQ send/recv timeouts (milliseconds)
                socket.setsockopt(zmq.SNDTIMEO, 1500)
                if name=="cam_server":
                    socket.setsockopt(zmq.RCVTIMEO, 4000)
                else:
                    socket.setsockopt(zmq.RCVTIMEO, 1500)
            if (name == "cam_server"):
                tab = ServerTab(name, sockets, rcv_timeout=4000)
                print("Using a longer cam_server timeout")
            else:
                tab = ServerTab(name, sockets)
            self.tabs.addTab(tab, name)
            self.tab_widgets.append(tab)

        self.tabs.currentChanged.connect(self.on_tab_changed)
        # Populate commands for the first tab only.
        self.tab_widgets[0].populate_commands()

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
            # Handle Escape key to close the application
            if key == QtCore.Qt.Key_Escape:
                self.close()
                return True
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
        # No longer clear history on tab change but do populate commands
        self.tab_widgets[idx].populate_commands()


def main():
    if len(sys.argv) == 1:
        ip_addr = "mimir"
    elif len(sys.argv) == 2:
        ip_addr = sys.argv[1]
    else:
        print("Usage: python universal_client.py <ip_address>")
        sys.exit(1)
    servers = sockets
    app = QtWidgets.QApplication(sys.argv)
    client = UniversalClient(ip_addr, servers)
    client.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
