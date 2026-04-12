import argparse
import subprocess
import sys

import zmq
from PyQt5 import QtCore, QtWidgets


class BLFPoller(QtCore.QThread):
	"""Background thread that periodically polls MDS port 5555 for BLF1-4."""

	status_updated = QtCore.pyqtSignal(str)

	def __init__(self, host, port=5555, interval_ms=5000):
		super().__init__()
		self._host = host
		self._port = port
		self._interval_ms = interval_ms
		self._stop = False

	def run(self):
		context = zmq.Context()
		try:
			while not self._stop:
				self._do_poll(context)
				elapsed = 0
				while elapsed < self._interval_ms and not self._stop:
					self.msleep(100)
					elapsed += 100
		finally:
			context.term()

	def _do_poll(self, context):
		socket = context.socket(zmq.REQ)
		socket.setsockopt(zmq.SNDTIMEO, 500)
		socket.setsockopt(zmq.RCVTIMEO, 500)
		socket.connect(f"tcp://{self._host}:{self._port}")

		values = []
		for i in range(1, 5):
			try:
				socket.send_string(f"read BLF{i}")
				reply = socket.recv_string().strip()
				values.append(int(reply))
			except (zmq.error.Again, ValueError, Exception):
				values.append(None)
				# Lazy pirate: fresh socket for next request
				socket.setsockopt(zmq.LINGER, 0)
				socket.close()
				socket = context.socket(zmq.REQ)
				socket.setsockopt(zmq.SNDTIMEO, 500)
				socket.setsockopt(zmq.RCVTIMEO, 500)
				socket.connect(f"tcp://{self._host}:{self._port}")

		socket.setsockopt(zmq.LINGER, 0)
		socket.close()

		if any(v is None for v in values):
			status = "Mixed/Error"
		elif all(v == 0 for v in values):
			status = "Standard"
		elif all(v == 1 for v in values):
			status = "Faint"
		else:
			status = "Mixed/Error"

		self.status_updated.emit(status)

	def stop(self):
		self._stop = True


class ShortcutsGUI(QtWidgets.QWidget):
	def __init__(self, host="mimir", debug=False):
		super().__init__()
		self.host = host
		self.debug = debug
		self.context = zmq.Context()
		self.heim_socket = self._build_socket(6660)
		self.baldr_sockets = []

		self.setWindowTitle("ASGARD Shortcuts")
		self.resize(900, 620)
		self._init_ui()
		self._refresh_baldr_sockets()

		self._poller = BLFPoller(host=self.host)
		self._poller.status_updated.connect(self._on_blf_status)
		self._poller.start()

	def _init_ui(self):
		root = QtWidgets.QVBoxLayout(self)

		heim_group = QtWidgets.QGroupBox("Heimdallr")
		heim_layout = QtWidgets.QVBoxLayout(heim_group)

		scripts_layout = QtWidgets.QHBoxLayout()
		self.align_btn = QtWidgets.QPushButton("Heim. Internal Align")
		self.align_btn.clicked.connect(
			lambda: self._run_script("h-autoalign", ["-a", "ia"])
		)
		self.tilts_btn = QtWidgets.QPushButton("Heim. Tilts")
		self.tilts_btn.clicked.connect(lambda: self._run_script("h-tilts", []))
		scripts_layout.addWidget(self.align_btn)
		scripts_layout.addWidget(self.tilts_btn)
		heim_layout.addLayout(scripts_layout)

		jump_layout = QtWidgets.QGridLayout()
		jump_layout.addWidget(QtWidgets.QLabel("Jump size:"), 0, 0)
		self.jump_spin = QtWidgets.QSpinBox()
		self.jump_spin.setMinimum(1)
		self.jump_spin.setMaximum(10000)
		self.jump_spin.setValue(30)
		jump_layout.addWidget(self.jump_spin, 0, 1)

		for i in range(4):
			jump_layout.addWidget(QtWidgets.QLabel(f"DL{i + 1}"), i + 1, 0)
			left_btn = QtWidgets.QPushButton("Left")
			left_btn.clicked.connect(
				lambda _checked=False, idx=i: self._send_dlr(idx, sign=1)
			)
			right_btn = QtWidgets.QPushButton("Right")
			right_btn.clicked.connect(
				lambda _checked=False, idx=i: self._send_dlr(idx, sign=-1)
			)
			jump_layout.addWidget(left_btn, i + 1, 1)
			jump_layout.addWidget(right_btn, i + 1, 2)

		heim_layout.addLayout(jump_layout)

		fringe_layout = QtWidgets.QHBoxLayout()
		self.fringe_btn = QtWidgets.QPushButton("Fringe Scan")
		self.fringe_btn.clicked.connect(self._fringe_scan)
		fringe_layout.addWidget(self.fringe_btn)

		fringe_layout.addWidget(QtWidgets.QLabel("gd_threshold:"))
		self.gd_spin = QtWidgets.QDoubleSpinBox()
		self.gd_spin.setDecimals(3)
		self.gd_spin.setMinimum(0.0)
		self.gd_spin.setMaximum(1_000_000.0)
		self.gd_spin.setValue(100.0)
		fringe_layout.addWidget(self.gd_spin)

		self.gd_btn = QtWidgets.QPushButton("Set gd_threshold")
		self.gd_btn.clicked.connect(self._set_gd_threshold)
		fringe_layout.addWidget(self.gd_btn)
		heim_layout.addLayout(fringe_layout)

		root.addWidget(heim_group)

		baldr_group = QtWidgets.QGroupBox("Baldr 1-4")
		baldr_layout = QtWidgets.QGridLayout(baldr_group)

		baldr_layout.addWidget(QtWidgets.QLabel("BLF state:"), 0, 0)
		self.blf_status_label = QtWidgets.QLabel("—")
		baldr_layout.addWidget(self.blf_status_label, 0, 1)
		baldr_layout.addWidget(QtWidgets.QLabel("Baldr mode:"), 0, 2)
		self.mode_combo = QtWidgets.QComboBox()
		self.mode_combo.addItems(["standard (6662-6665)", "faint (6671-6674)"])
		self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
		baldr_layout.addWidget(self.mode_combo, 0, 3)

		baldr_layout.addWidget(QtWidgets.QLabel("ttg:"), 1, 0)
		self.ttg_edit = QtWidgets.QLineEdit()
		self.ttg_edit.setPlaceholderText("Enter ttg value")
		baldr_layout.addWidget(self.ttg_edit, 1, 1)
		self.ttg_btn = QtWidgets.QPushButton("Send ttg")
		self.ttg_btn.clicked.connect(
			lambda: self._send_baldr_all(f"ttg {self.ttg_edit.text().strip()}")
		)
		baldr_layout.addWidget(self.ttg_btn, 1, 2)

		baldr_layout.addWidget(QtWidgets.QLabel("hog:"), 2, 0)
		self.hog_edit = QtWidgets.QLineEdit()
		self.hog_edit.setPlaceholderText("Enter hog value")
		baldr_layout.addWidget(self.hog_edit, 2, 1)
		self.hog_btn = QtWidgets.QPushButton("Send hog")
		self.hog_btn.clicked.connect(
			lambda: self._send_baldr_all(f"hog {self.hog_edit.text().strip()}")
		)
		baldr_layout.addWidget(self.hog_btn, 2, 2)

		self.close_tt_btn = QtWidgets.QPushButton("Close TT")
		self.close_tt_btn.clicked.connect(lambda: self._send_baldr_all("ttg close"))
		baldr_layout.addWidget(self.close_tt_btn, 3, 1)

		self.close_hh_btn = QtWidgets.QPushButton("Close HH")
		self.close_hh_btn.clicked.connect(lambda: self._send_baldr_all("hog close"))
		baldr_layout.addWidget(self.close_hh_btn, 3, 2)

		root.addWidget(baldr_group)

		self.log = QtWidgets.QTextEdit()
		self.log.setReadOnly(True)
		root.addWidget(self.log)

	def _build_socket(self, port):
		socket = self.context.socket(zmq.REQ)
		socket.setsockopt(zmq.SNDTIMEO, 1500)
		socket.setsockopt(zmq.RCVTIMEO, 2000)
		socket.connect(f"tcp://{self.host}:{port}")
		return socket

	def _on_mode_changed(self, index):
		script_arg = "standard" if index == 0 else "faint"
		self._run_script("b-mode", [script_arg])
		self._refresh_baldr_sockets()

	def _on_blf_status(self, status):
		self.blf_status_label.setText(status)

	def _refresh_baldr_sockets(self):
		for socket in self.baldr_sockets:
			socket.setsockopt(zmq.LINGER, 0)
			socket.close()

		self.baldr_sockets = []
		base = 6662 if self.mode_combo.currentIndex() == 0 else 6671
		ports = [base + i for i in range(4)]
		for port in ports:
			self.baldr_sockets.append(self._build_socket(port))

		self._append_log(
			f"[INFO] Connected Baldr sockets on {self.host}: {', '.join(str(p) for p in ports)}"
		)

	def _replace_socket(self, old_socket):
		"""Lazy Pirate: close *old_socket* and return a fresh REQ socket
		connected to the same endpoint. Also updates self.heim_socket /
		self.baldr_sockets in-place so future callers get the new socket."""
		try:
			endpoint = old_socket.getsockopt(zmq.LAST_ENDPOINT).decode()
		except Exception:
			endpoint = None

		old_socket.setsockopt(zmq.LINGER, 0)
		old_socket.close()

		new_socket = self.context.socket(zmq.REQ)
		new_socket.setsockopt(zmq.SNDTIMEO, 1500)
		new_socket.setsockopt(zmq.RCVTIMEO, 2000)
		if endpoint:
			new_socket.connect(endpoint)

		if self.heim_socket is old_socket:
			self.heim_socket = new_socket
		for i, s in enumerate(self.baldr_sockets):
			if s is old_socket:
				self.baldr_sockets[i] = new_socket
		return new_socket

	def _send_cmd(self, socket, cmd):
		if self.debug:
			self._append_log(f"[DEBUG] {cmd}")
			return "debug"

		try:
			socket.send_string(cmd)
			reply = socket.recv_string()
			self._append_log(f"[OK] {cmd} -> {reply}")
			return reply
		except zmq.error.Again:
			self._append_log(f"[TIMEOUT] {cmd} — reconnecting and retrying…")
			socket = self._replace_socket(socket)
			try:
				socket.send_string(cmd)
				reply = socket.recv_string()
				self._append_log(f"[OK] {cmd} -> {reply}")
				return reply
			except zmq.error.Again:
				self._append_log(f"[TIMEOUT] {cmd} — retry also timed out")
		except Exception as exc:
			self._append_log(f"[ERROR] {cmd} -> {exc}")
		return None

	def _send_dlr(self, index, sign):
		jump = self.jump_spin.value()
		values = [0, 0, 0, 0]
		values[index] = sign * jump
		cmd = f"dlr {','.join(str(v) for v in values)}"
		self._send_cmd(self.heim_socket, cmd)

	def _fringe_scan(self):
		cmds = ["set_gd_threshold 100", "search 1.0, 50", "dls 0,0,0,0"]
		for cmd in cmds:
			self._send_cmd(self.heim_socket, cmd)

	def _set_gd_threshold(self):
		value = self.gd_spin.value()
		self._send_cmd(self.heim_socket, f"set_gd_threshold {value}")

	def _send_baldr_all(self, cmd):
		if not cmd or cmd.endswith(" "):
			self._append_log("[WARN] Command is empty.")
			return

		for i, socket in enumerate(self.baldr_sockets, start=1):
			if self.debug:
				self._append_log(f"[DEBUG] Baldr{i}: {cmd}")
				continue

			try:
				socket.send_string(cmd)
				reply = socket.recv_string()
				self._append_log(f"[OK] Baldr{i} {cmd} -> {reply}")
			except zmq.error.Again:
				self._append_log(f"[TIMEOUT] Baldr{i} {cmd}")
			except Exception as exc:
				self._append_log(f"[ERROR] Baldr{i} {cmd} -> {exc}")

	def _run_script(self, script, args):
		cmd = [script, *args]
		if self.debug:
			self._append_log(f"[DEBUG] run: {' '.join(cmd)}")
			return

		try:
			subprocess.Popen(cmd)
			self._append_log(f"[OK] launched: {' '.join(cmd)}")
		except Exception as exc:
			self._append_log(f"[ERROR] failed to launch {' '.join(cmd)} -> {exc}")

	def _append_log(self, text):
		self.log.append(text)

	def closeEvent(self, event):
		self._poller.stop()
		self._poller.wait(2000)
		self.heim_socket.setsockopt(zmq.LINGER, 0)
		self.heim_socket.close()
		for socket in self.baldr_sockets:
			socket.setsockopt(zmq.LINGER, 0)
			socket.close()
		self.context.term()
		super().closeEvent(event)


def main():
	parser = argparse.ArgumentParser(description="ASGARD shortcuts GUI")
	parser.add_argument("--host", default="mimir", help="ZMQ host")
	parser.add_argument(
		"--debug",
		action="store_true",
		help="Log commands without sending or launching scripts",
	)
	args = parser.parse_args()

	app = QtWidgets.QApplication(sys.argv)
	win = ShortcutsGUI(host=args.host, debug=args.debug)
	win.show()
	sys.exit(app.exec_())


if __name__ == "__main__":
	main()
