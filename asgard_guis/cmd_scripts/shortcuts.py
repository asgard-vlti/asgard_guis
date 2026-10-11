import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

import zmq
from PyQt5 import QtCore, QtWidgets


RECON_SCRIPT = "/home/asg/.conda/envs/asgard/bin/baldrtt-recon"
RECON_HOST = "mimir"
BALDR_TT_PORTS = {beam: 6670 + beam for beam in range(1, 5)}
HEIM_SERVO_MODES = {0: "Fight", 1: "Lacour", 2: "Stop", 3: "Simple", 4: "Off", 5: "DDSPC"}
HEIM_OFFLOAD_MODES = {0: "Nested", 1: "GD", 2: "Off", 3: "Manual", 4: "Mod"}
DDSPC_HISTORY_PRESETS = ((20, 2), (30, 3), (40, 3), (50, 3))


def _is_recon_command(argv):
	if not argv:
		return False
	if Path(argv[0]).name.startswith("python"):
		if len(argv) > 2 and argv[1] == "-m":
			return argv[2] == "dcs.baldr_tt.save_recon"
		if len(argv) < 2:
			return False
		script = argv[1]
	else:
		script = argv[0]
	return Path(script).name == "baldrtt-recon" or script.endswith(
		"/baldr_tt/save_recon.py"
	)


def _is_recon_pid(pid, proc_root="/proc"):
	try:
		with open(f"{proc_root}/{pid}/cmdline", "rb") as cmdline:
			argv = [part.decode("utf-8", "replace") for part in cmdline.read().split(b"\0") if part]
	except OSError:
		return False
	return _is_recon_command(argv)


def _recon_pids(proc_root="/proc"):
	try:
		entries = os.listdir(proc_root)
	except OSError:
		return []
	return sorted(
		int(entry)
		for entry in entries
		if entry.isdigit() and int(entry) != os.getpid() and _is_recon_pid(entry, proc_root)
	)


def _stop_recon_processes(log_message, grace_seconds=4):
	pids = _recon_pids()
	if not pids:
		log_message("[INFO] No baldrtt-recon jobs found.")
		return True

	for pid in pids:
		if not _is_recon_pid(pid):
			continue
		try:
			os.kill(pid, signal.SIGTERM)
			log_message(f"[INFO] Sent SIGTERM to baldrtt-recon PID {pid}.")
		except ProcessLookupError:
			pass
		except OSError as exc:
			log_message(f"[ERROR] Cannot stop baldrtt-recon PID {pid}: {exc}")

	deadline = time.monotonic() + grace_seconds
	remaining = [pid for pid in pids if _is_recon_pid(pid)]
	while remaining and time.monotonic() < deadline:
		time.sleep(max(0.0, min(0.1, deadline - time.monotonic())))
		remaining = [pid for pid in remaining if _is_recon_pid(pid)]

	for pid in remaining:
		if not _is_recon_pid(pid):
			continue
		try:
			os.kill(pid, signal.SIGKILL)
			log_message(f"[WARN] Force-stopped baldrtt-recon PID {pid}.")
		except ProcessLookupError:
			pass
		except OSError as exc:
			log_message(f"[ERROR] Cannot force-stop baldrtt-recon PID {pid}: {exc}")

	force_deadline = time.monotonic() + 0.5
	still_running = [pid for pid in remaining if _is_recon_pid(pid)]
	while still_running and time.monotonic() < force_deadline:
		time.sleep(0.05)
		still_running = [pid for pid in still_running if _is_recon_pid(pid)]
	if still_running:
		log_message(f"[ERROR] Recon jobs still running: {still_running}")
	return not still_running


def _clear_baldr_beam(host, beam):
	context = zmq.Context.instance()
	socket = context.socket(zmq.REQ)
	socket.setsockopt(zmq.SNDTIMEO, 500)
	socket.setsockopt(zmq.RCVTIMEO, 2000)
	socket.setsockopt(zmq.LINGER, 0)
	socket.connect(f"tcp://{host}:{BALDR_TT_PORTS[beam]}")
	try:
		socket.send_string("stop_clear_dm")
		reply = socket.recv_json()
		if reply is not True:
			raise RuntimeError(
				f"server did not acknowledge stop_clear_dm (reply: {reply!r}); "
				"check that the updated /usr/local/bin/baldr_tt is running"
			)
	finally:
		socket.close()


class StopReconWorker(QtCore.QThread):
	log_message = QtCore.pyqtSignal(str)
	stop_finished = QtCore.pyqtSignal(bool)

	def __init__(self, host, debug=False):
		super().__init__()
		self._host = host
		self._debug = debug

	def run(self):
		if self._debug:
			self.log_message.emit("[DEBUG] Would stop all local baldrtt-recon jobs and clear Baldr TT beams 1-4.")
			self.stop_finished.emit(True)
			return

		try:
			processes_stopped = _stop_recon_processes(self.log_message.emit)
			beams_cleared = True
			with ThreadPoolExecutor(max_workers=4) as executor:
				futures = {
					executor.submit(_clear_baldr_beam, self._host, beam): beam
					for beam in BALDR_TT_PORTS
				}
				for future in as_completed(futures):
					beam = futures[future]
					try:
						future.result()
						self.log_message.emit(f"[OK] Baldr{beam} servo stopped and DM channels cleared.")
					except Exception as exc:
						beams_cleared = False
						self.log_message.emit(f"[ERROR] Baldr{beam} reset failed: {exc}")
			self.stop_finished.emit(processes_stopped and beams_cleared)
		except Exception as exc:
			self.log_message.emit(f"[ERROR] Recon stop failed: {exc}")
			self.stop_finished.emit(False)


class BLFPoller(QtCore.QThread):
	"""Background thread that periodically polls MDS port 5555 for BLF1-4."""

	status_updated = QtCore.pyqtSignal(str)

	def __init__(self, host, port=5555, interval_ms=15000):
		super().__init__()
		self._host = host
		self._port = port
		self._interval_ms = interval_ms
		self._stop = False
		self._blf_values = [None] * 4
		self._next_blf_index = 0

	def run(self):
		context = zmq.Context()
		try:
			while not self._stop:
				self._do_poll(context)
				elapsed = 0
				while elapsed < self._interval_ms and not self._stop:
					self.msleep(200)
					elapsed += 200
		finally:
			context.term()

	def _do_poll(self, context):
		index = self._next_blf_index
		value = self._read_blf(context, index)
		value_changed = (
			self._blf_values[index] is not None and value != self._blf_values[index]
		)
		self._blf_values[index] = value
		self._next_blf_index = (index + 1) % 4

		if value_changed:
			for other_index in range(4):
				if other_index != index:
					self._blf_values[other_index] = self._read_blf(context, other_index)

		values = self._blf_values
		if any(value is None for value in values):
			status = "Mixed/Error"
		elif all(value == 0 for value in values):
			status = "Standard"
		elif all(value == 1 for value in values):
			status = "Faint"
		else:
			status = "Mixed/Error"

		self.status_updated.emit(status)

	def _read_blf(self, context, index):
		socket = context.socket(zmq.REQ)
		socket.setsockopt(zmq.SNDTIMEO, 500)
		socket.setsockopt(zmq.RCVTIMEO, 500)
		socket.connect(f"tcp://{self._host}:{self._port}")

		try:
			socket.send_string(f"read BLF{index + 1}")
			return int(float(socket.recv_string().strip()))
		except (zmq.error.Again, ValueError, Exception):
			return None
		finally:
			socket.setsockopt(zmq.LINGER, 0)
			socket.close()

	def stop(self):
		self._stop = True


class HeimdallrLoopWorker(QtCore.QThread):
	"""Own Heimdallr status and control requests outside the GUI thread."""

	state_updated = QtCore.pyqtSignal(object, object)
	state_failed = QtCore.pyqtSignal(str)
	command_finished = QtCore.pyqtSignal(str, bool, str)
	log_message = QtCore.pyqtSignal(str)

	def __init__(self, host, interval_seconds=1.0):
		super().__init__()
		self._host = host
		self._interval_seconds = interval_seconds
		self._commands = queue.Queue()
		self._stop_requested = threading.Event()

	def enqueue(self, commands, label):
		self._commands.put((list(commands), label))

	def stop(self):
		self._stop_requested.set()
		self._commands.put(None)

	def _request(self, context, command, timeout_ms):
		socket = context.socket(zmq.REQ)
		socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
		socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
		socket.setsockopt(zmq.LINGER, 0)
		socket.connect(f"tcp://{self._host}:6660")
		try:
			socket.send_string(command)
			reply = socket.recv_string()
			try:
				value = json.loads(reply)
			except json.JSONDecodeError:
				value = reply
			if isinstance(value, dict) and "error" in value:
				raise RuntimeError(str(value["error"]))
			if isinstance(value, str) and value.upper().startswith("ERROR"):
				raise RuntimeError(value)
			return value
		finally:
			socket.close()

	def _poll_once(self, context):
		settings = self._request(context, "settings", 500)
		ddspc = self._request(context, 'ddspc "get"', 500)
		if not isinstance(settings, dict) or not all(
			key in settings for key in ("servo_mode", "offload_mode", "delay_line_type", "kp")
		):
			raise ValueError("settings reply is missing loop state")
		if not isinstance(ddspc, dict) or not isinstance(ddspc.get("configured"), dict):
			raise ValueError("DDSPC reply is missing configured state")
		self.state_updated.emit(settings, ddspc)
		return settings, ddspc

	def _refresh_state(self, context):
		try:
			return self._poll_once(context)
		except Exception as exc:
			self.state_failed.emit(str(exc))
			return None

	@staticmethod
	def _readback_errors(commands, settings, ddspc):
		expected = {}
		for command in commands:
			if command == 'servo "off"':
				expected["servo_mode"] = 4
			elif command == 'servo "on"':
				expected.update(servo_mode=1, offload_mode=0)
			elif command == 'servo "ddspc"':
				expected.update(servo_mode=5, offload_mode=0)
			elif command == 'offload "off"':
				expected["offload_mode"] = 2
			elif command == 'offload "gd"':
				expected.update(servo_mode=4, offload_mode=1)
			elif command == 'offload "mod"':
				expected.update(servo_mode=4, offload_mode=4)
			elif command.startswith("gain "):
				expected["kp"] = float(command[5:])
			elif command.startswith('ddspc "set-reg-cutoff" '):
				expected["reg_cutoff"] = float(command.split()[-1])
			elif command.startswith('ddspc "set-history" '):
				expected["history"] = tuple(json.loads(command.split(" ", 2)[2]))

		errors = []
		for key in ("servo_mode", "offload_mode"):
			if key in expected and settings.get(key) != expected[key]:
				errors.append(f"{key} readback is {settings.get(key)!r}, expected {expected[key]}")
		if "kp" in expected:
			try:
				actual = float(settings["kp"])
				if not math.isfinite(actual) or not math.isclose(actual, expected["kp"], rel_tol=1e-9, abs_tol=1e-12):
					errors.append(f"Kp readback is {actual!r}, expected {expected['kp']}")
			except (KeyError, TypeError, ValueError):
				errors.append("Kp readback is unavailable")
		configured = ddspc["configured"]
		if "reg_cutoff" in expected:
			try:
				actual = float(configured["reg_cutoff"])
				if not math.isfinite(actual) or not math.isclose(actual, expected["reg_cutoff"], rel_tol=1e-9, abs_tol=1e-12):
					errors.append(f"cutoff readback is {actual!r}, expected {expected['reg_cutoff']}")
			except (KeyError, TypeError, ValueError):
				errors.append("cutoff readback is unavailable")
		if "history" in expected:
			actual = (configured.get("history"), configured.get("future"))
			if actual != expected["history"]:
				errors.append(f"history readback is {actual!r}, expected {expected['history']}")
		return errors

	def _execute(self, context, commands, label):
		errors = []
		for command in commands:
			try:
				self._request(context, command, 1500)
				self.log_message.emit(f"[ACK] {command}")
			except Exception as exc:
				errors.append(f"{command}: {exc}")
				self.log_message.emit(f"[ERROR] {command} -> {exc}")
		state = self._refresh_state(context)
		if state is None:
			errors.append("resulting state could not be verified")
		else:
			errors.extend(self._readback_errors(commands, *state))
		if not errors:
			self.log_message.emit(f"[OK] {label} verified by readback")
		elif state is not None:
			self.log_message.emit(f"[ERROR] {label} -> {'; '.join(errors)}")
		self.command_finished.emit(label, not errors, "; ".join(errors))

	def run(self):
		context = zmq.Context()
		next_poll = time.monotonic()
		try:
			while not self._stop_requested.is_set():
				try:
					item = self._commands.get_nowait()
				except queue.Empty:
					if time.monotonic() >= next_poll:
						poll_started = time.monotonic()
						self._refresh_state(context)
						next_poll = poll_started + self._interval_seconds
						while next_poll <= time.monotonic():
							next_poll += self._interval_seconds
						continue
					try:
						item = self._commands.get(timeout=min(0.2, max(0, next_poll - time.monotonic())))
					except queue.Empty:
						continue
				if item is None or self._stop_requested.is_set():
					break
				commands, label = item
				self._execute(context, commands, label)
				next_poll = time.monotonic() + self._interval_seconds
		finally:
			context.term()


class ReconWorker(QtCore.QThread):
	"""Run the long Baldr TT reconstructor process without blocking the GUI."""

	log_message = QtCore.pyqtSignal(str)
	recon_finished = QtCore.pyqtSignal(bool)

	def __init__(self, debug=False):
		super().__init__()
		self._debug = debug
		self._stop_requested = False
		self._launch_lock = threading.Lock()

	def request_stop(self):
		with self._launch_lock:
			self._stop_requested = True

	def run(self):
		if self._debug:
			self.log_message.emit("[DEBUG] run: baldrtt-recon")
			self.recon_finished.emit(True)
			return

		try:
			script = RECON_SCRIPT
			with self._launch_lock:
				if self._stop_requested:
					process = None
				else:
					self.log_message.emit(f"[INFO] Running: {script}")
					process = subprocess.Popen([script])
			if process is None:
				self.log_message.emit("[INFO] baldrtt-recon start cancelled.")
				self.recon_finished.emit(False)
				return
			returncode = process.wait()
			if self._stop_requested and returncode == 143:
				self.log_message.emit("[INFO] baldrtt-recon stopped after SIGTERM; checking DM resets.")
				self.recon_finished.emit(False)
				return
			if returncode != 0:
				self.log_message.emit(f"[ERROR] {script} exited with status {returncode}.")
				self.recon_finished.emit(False)
				return

			self.recon_finished.emit(True)
		except Exception as exc:
			self.log_message.emit(f"[ERROR] {script} failed: {exc}")
			self.recon_finished.emit(False)


class ShortcutsGUI(QtWidgets.QWidget):
	def __init__(self, host="mimir", debug=False):
		super().__init__()
		self.host = host
		self.debug = debug
		self.context = zmq.Context()
		self.heim_socket = self._build_socket(6660)
		self.mds_socket = self._build_socket(5555)
		self.cam_server_socket = self._build_socket(6667, host="mimir")
		self.baldr_sockets = []
		self.current_baldr_mode = "Standard"
		self._recon_generation = 0
		self._recon_stop_requested = False
		self._heim_settings = None
		self._heim_ddspc = None
		self._heim_state_fresh = False
		self._heim_state_time = 0.0
		self._heim_action_pending = False
		self._kp_dirty = False
		self._cutoff_dirty = False
		self._history_dirty = False
		self._pending_kp = None
		self._pending_cutoff = None
		self._pending_history = None

		self.setWindowTitle("ASGARD Shortcuts")
		self._apply_dark_theme()
		self._init_ui()
		self.adjustSize()
		self.resize(self.minimumSizeHint())
		self.setMinimumSize(self.minimumSizeHint())
		self._refresh_baldr_sockets()

		self._poller = BLFPoller(host=self.host)
		self._poller.status_updated.connect(self._on_blf_status)
		self._poller.start()
		self._heim_worker = HeimdallrLoopWorker(self.host)
		self._heim_worker.state_updated.connect(self._on_heim_state)
		self._heim_worker.state_failed.connect(self._on_heim_state_failed)
		self._heim_worker.command_finished.connect(self._on_heim_command_finished)
		self._heim_worker.log_message.connect(self._append_log)
		self._heim_stale_timer = QtCore.QTimer(self)
		self._heim_stale_timer.setInterval(500)
		self._heim_stale_timer.timeout.connect(self._check_heim_stale)
		if not self.debug:
			self._heim_worker.start()
			self._heim_stale_timer.start()

	def _init_ui(self):
		root = QtWidgets.QVBoxLayout(self)
		root.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)

		heim_group = QtWidgets.QGroupBox("Heimdallr")
		heim_layout = QtWidgets.QGridLayout(heim_group)
		self.heim_layout = heim_layout
		self.align_btn = QtWidgets.QPushButton("Internal Align")
		self.align_btn.clicked.connect(
			lambda: self._run_script("h-autoalign", ["-a", "ia"])
		)
		self.tilts_btn = QtWidgets.QPushButton("Tilts")
		self.tilts_btn.clicked.connect(lambda: self._run_script("h-tilts", []))
		self.sky_mode_btn = QtWidgets.QPushButton("Sky Mode")
		self.sky_mode_btn.clicked.connect(lambda: self._run_script("s-skymode", []))
		self.lab_mode_btn = QtWidgets.QPushButton("Lab Mode")
		self.lab_mode_btn.clicked.connect(lambda: self._run_script("s-labmode", []))
		self.make_dark_btn = QtWidgets.QPushButton("Make Dark")
		self.make_dark_btn.clicked.connect(self._make_dark)
		for column, button in enumerate(
			(self.align_btn, self.tilts_btn, self.sky_mode_btn, self.lab_mode_btn)
		):
			heim_layout.addWidget(button, 0, column)
		heim_layout.addWidget(self.make_dark_btn, 1, 0)

		self.fringe_btn = QtWidgets.QPushButton("Fringe Scan")
		self.fringe_btn.clicked.connect(self._fringe_scan)
		heim_layout.addWidget(self.fringe_btn, 1, 1)
		self.heim_status_label = QtWidgets.QLabel("Debug: no live state" if self.debug else "Waiting for loop state…")
		self.heim_status_label.setWordWrap(True)
		self.heim_status_label.setSizePolicy(
			QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred
		)
		heim_layout.addWidget(self.heim_status_label, 1, 2, 1, 2)

		heim_layout.addWidget(QtWidgets.QLabel("Jump size:"), 2, 0)
		self.jump_spin = QtWidgets.QSpinBox()
		self.jump_spin.setMinimum(1)
		self.jump_spin.setMaximum(10000)
		self.jump_spin.setValue(30)
		heim_layout.addWidget(self.jump_spin, 2, 1)
		heim_layout.addWidget(QtWidgets.QLabel("gd_threshold:"), 2, 2)
		self.gd_spin = QtWidgets.QDoubleSpinBox()
		self.gd_spin.setDecimals(1)
		self.gd_spin.setMinimum(0.0)
		self.gd_spin.setMaximum(1_000.0)
		self.gd_spin.setValue(100.0)
		heim_layout.addWidget(self.gd_spin, 2, 3)
		gd_line_edit = self.gd_spin.lineEdit()
		if gd_line_edit is not None:
			gd_line_edit.returnPressed.connect(self._set_gd_threshold)

		jump_strip = QtWidgets.QWidget()
		jump_strip_layout = QtWidgets.QHBoxLayout(jump_strip)
		jump_strip_layout.setContentsMargins(0, 0, 0, 0)
		jump_strip_layout.setSpacing(0)
		self.jump_buttons = {}
		for i in range(4):
			jump_cell = QtWidgets.QWidget()
			jump_cell.setSizePolicy(
				QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
			)
			jump_row = QtWidgets.QHBoxLayout(jump_cell)
			jump_row.setContentsMargins(0, 0, 0, 0)
			jump_row.setSpacing(3)
			jump_row.addStretch()
			jump_row.addWidget(QtWidgets.QLabel(f"DL{i + 1}"))
			left_btn = QtWidgets.QPushButton("◀")
			left_btn.setToolTip(f"Jump DL{i + 1} left")
			left_btn.setFixedWidth(32)
			left_btn.clicked.connect(
				lambda _checked=False, idx=i: self._send_dlr(idx, sign=1)
			)
			right_btn = QtWidgets.QPushButton("▶")
			right_btn.setToolTip(f"Jump DL{i + 1} right")
			right_btn.setFixedWidth(32)
			right_btn.clicked.connect(
				lambda _checked=False, idx=i: self._send_dlr(idx, sign=-1)
			)
			jump_row.addWidget(left_btn)
			jump_row.addWidget(right_btn)
			jump_row.addStretch()
			jump_strip_layout.addWidget(jump_cell, 1)
			self.jump_buttons[i + 1] = (left_btn, right_btn)
		heim_layout.addWidget(jump_strip, 3, 0, 1, 4)

		heim_layout.addWidget(QtWidgets.QLabel("Servo:"), 4, 0)
		self.servo_status_label = QtWidgets.QLabel("Unknown")
		heim_layout.addWidget(self.servo_status_label, 4, 1, 1, 3)
		self.servo_buttons = {}
		for column, (mode, title, command, hint) in enumerate(
			(("off", "Off", 'servo "off"', "Stops the servo; keeps the current offload mode."),
			 ("on", "Lacour", 'servo "on"', "Starts Lacour and selects Nested offload."),
			 ("ddspc", "DDSPC", 'servo "ddspc"', "Starts DDSPC and selects Nested offload."))
		):
			button = QtWidgets.QPushButton(title)
			button.setCheckable(True)
			button.setToolTip(hint)
			button.clicked.connect(
				lambda _checked=False, cmd=command, name=title: self._queue_heim_action([cmd], f"Servo {name}")
			)
			heim_layout.addWidget(button, 5, column)
			self.servo_buttons[mode] = button
		self.all_off_btn = QtWidgets.QPushButton("All Off")
		self.all_off_btn.setToolTip("Stops the servo, then turns off offloading.")
		self.all_off_btn.clicked.connect(
			lambda: self._queue_heim_action(['servo "off"', 'offload "off"'], "All Off")
		)
		heim_layout.addWidget(self.all_off_btn, 5, 3)

		heim_layout.addWidget(QtWidgets.QLabel("Offload:"), 6, 0)
		self.offload_status_label = QtWidgets.QLabel("Unknown")
		heim_layout.addWidget(self.offload_status_label, 6, 1)
		self.delay_line_status_label = QtWidgets.QLabel("Delay line: Unknown")
		heim_layout.addWidget(self.delay_line_status_label, 6, 2, 1, 2)
		self.offload_buttons = {}
		for column, (mode, title) in enumerate((("off", "Off"), ("gd", "GD"), ("mod", "Mod"))):
			button = QtWidgets.QPushButton(title)
			button.setCheckable(True)
			if mode in ("gd", "mod"):
				button.setToolTip("Stops the servo and selects this offload mode.")
			else:
				button.setToolTip("Turns off offloading; keeps the current servo mode.")
			button.clicked.connect(
				lambda _checked=False, value=mode: self._queue_heim_action([f'offload "{value}"'], f"Offload {value.upper()}")
			)
			heim_layout.addWidget(button, 7, column)
			self.offload_buttons[mode] = button

		heim_layout.addWidget(QtWidgets.QLabel("Kp (phase delay):"), 8, 0)
		self.kp_edit = QtWidgets.QLineEdit()
		self.kp_edit.setPlaceholderText("Kp")
		self.kp_edit.textChanged.connect(lambda _text: setattr(self, "_kp_dirty", True))
		self.kp_edit.returnPressed.connect(self._apply_kp)
		heim_layout.addWidget(self.kp_edit, 8, 1)
		self.apply_kp_btn = QtWidgets.QPushButton("Apply Kp")
		self.apply_kp_btn.clicked.connect(self._apply_kp)
		heim_layout.addWidget(self.apply_kp_btn, 8, 2)
		self.kp_status_label = QtWidgets.QLabel("Live: —")
		heim_layout.addWidget(self.kp_status_label, 8, 3)

		heim_layout.addWidget(QtWidgets.QLabel("DDSPC cutoff:"), 9, 0)
		self.cutoff_edit = QtWidgets.QLineEdit()
		self.cutoff_edit.setPlaceholderText("Next entry")
		self.cutoff_edit.textChanged.connect(lambda _text: setattr(self, "_cutoff_dirty", True))
		self.cutoff_edit.returnPressed.connect(self._stage_cutoff)
		heim_layout.addWidget(self.cutoff_edit, 9, 1)
		self.stage_cutoff_btn = QtWidgets.QPushButton("Stage cutoff")
		self.stage_cutoff_btn.clicked.connect(self._stage_cutoff)
		heim_layout.addWidget(self.stage_cutoff_btn, 9, 2)
		self.cutoff_status_label = QtWidgets.QLabel("conf: — | cur: —")
		heim_layout.addWidget(self.cutoff_status_label, 9, 3)

		heim_layout.addWidget(QtWidgets.QLabel("DDSPC history:"), 10, 0)
		self.history_combo = QtWidgets.QComboBox()
		for history, future in DDSPC_HISTORY_PRESETS:
			self.history_combo.addItem(f"{history}/{future}", (history, future))
		self.history_combo.currentIndexChanged.connect(
			lambda _index: setattr(self, "_history_dirty", True)
		)
		heim_layout.addWidget(self.history_combo, 10, 1)
		self.set_history_btn = QtWidgets.QPushButton("Set History")
		self.set_history_btn.clicked.connect(self._set_history)
		heim_layout.addWidget(self.set_history_btn, 10, 2)
		self.history_status_label = QtWidgets.QLabel("conf: — | cur: —")
		heim_layout.addWidget(self.history_status_label, 10, 3)
		heim_layout.addWidget(QtWidgets.QLabel("DDSPC edits are staged for the next entry."), 11, 0, 1, 4)
		self._refresh_heim_action_availability()

		root.addWidget(heim_group)

		baldr_group = QtWidgets.QGroupBox("Baldr 1-4")
		baldr_layout = QtWidgets.QGridLayout(baldr_group)

		baldr_layout.addWidget(QtWidgets.QLabel("Current Baldr mode:"), 0, 0)
		self.baldr_status_label = QtWidgets.QLabel("—")
		baldr_layout.addWidget(self.baldr_status_label, 0, 1)
		baldr_layout.addWidget(QtWidgets.QLabel("Set Baldr mode:"), 0, 2)
		self.mode_combo = QtWidgets.QComboBox()
		self.mode_combo.addItems(["Set mode...", "Standard", "Faint"])
		self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
		baldr_layout.addWidget(self.mode_combo, 0, 3)

		baldr_layout.addWidget(QtWidgets.QLabel("ttg:"), 1, 0)
		self.ttg_edit = QtWidgets.QLineEdit()
		self.ttg_edit.setPlaceholderText("Value")
		baldr_layout.addWidget(self.ttg_edit, 1, 1)
		self.ttg_btn = QtWidgets.QPushButton("Send ttg")
		self.ttg_btn.clicked.connect(
			lambda: self._send_baldr_all(f"ttg {self.ttg_edit.text().strip()}")
		)
		baldr_layout.addWidget(self.ttg_btn, 1, 2)

		baldr_layout.addWidget(QtWidgets.QLabel("hog:"), 2, 0)
		self.hog_edit = QtWidgets.QLineEdit()
		self.hog_edit.setPlaceholderText("Value")
		baldr_layout.addWidget(self.hog_edit, 2, 1)
		self.hog_btn = QtWidgets.QPushButton("Send hog")
		self.hog_btn.clicked.connect(
			lambda: self._send_baldr_all(f"hog {self.hog_edit.text().strip()}")
		)
		baldr_layout.addWidget(self.hog_btn, 2, 2)
		self.recon_btn = QtWidgets.QPushButton("Recon")
		self.recon_btn.clicked.connect(self._run_recon)
		baldr_layout.addWidget(self.recon_btn, 3, 0)
		self.stop_recon_btn = QtWidgets.QPushButton("Stop Recon (All)")
		self.stop_recon_btn.clicked.connect(self._stop_recon)
		baldr_layout.addWidget(self.stop_recon_btn, 5, 0, 1, 4)
		self.recon_status_label = QtWidgets.QLabel("")
		baldr_layout.addWidget(self.recon_status_label, 4, 0, 1, 4)

		self.close_tt_btn = QtWidgets.QPushButton("Close TT")
		self.close_tt_btn.clicked.connect(lambda: self._send_baldr_all('servo "tt"'))
		baldr_layout.addWidget(self.close_tt_btn, 3, 1)

		self.close_hh_btn = QtWidgets.QPushButton("Close HO")
		self.close_hh_btn.clicked.connect(lambda: self._send_baldr_all('servo "ho"'))
		baldr_layout.addWidget(self.close_hh_btn, 3, 2)

		self.open_loops_btn = QtWidgets.QPushButton("Open Loops")
		self.open_loops_btn.clicked.connect(
			lambda: self._send_baldr_all('servo "off"')
		)
		baldr_layout.addWidget(self.open_loops_btn, 3, 3)

		root.addWidget(baldr_group)

		self.log = QtWidgets.QTextEdit()
		self.log.setReadOnly(True)
		self.log.setMinimumHeight(120)
		root.addWidget(self.log)

	def _apply_dark_theme(self):
		self.setStyleSheet(
			"""
			QWidget {
				background-color: #1e1f22;
				color: #e6e6e6;
			}
			QGroupBox {
				border: 1px solid #3a3d41;
				border-radius: 6px;
				margin-top: 10px;
				padding-top: 8px;
				font-weight: 600;
			}
			QGroupBox::title {
				subcontrol-origin: margin;
				left: 10px;
				padding: 0 4px;
			}
			QPushButton {
				background-color: #2d3138;
				border: 1px solid #4a4f57;
				border-radius: 5px;
				padding: 5px 10px;
			}
			QPushButton:hover {
				background-color: #383d45;
			}
			QPushButton:pressed {
				background-color: #24282e;
			}
			QPushButton:checked {
				background-color: #225740;
				border: 2px solid #78d9a2;
				color: #ffffff;
			}
			QPushButton:disabled {
				color: #858b94;
			}
			QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {
				background-color: #272a30;
				border: 1px solid #4a4f57;
				border-radius: 4px;
				padding: 4px;
			}
			QComboBox QAbstractItemView {
				background-color: #272a30;
				color: #e6e6e6;
				selection-background-color: #3f6db3;
			}
			"""
		)

	def _build_socket(self, port, host=None):
		socket = self.context.socket(zmq.REQ)
		socket.setsockopt(zmq.SNDTIMEO, 1500)
		socket.setsockopt(zmq.RCVTIMEO, 2000)
		target_host = host if host is not None else self.host
		socket.connect(f"tcp://{target_host}:{port}")
		return socket

	@staticmethod
	def _finite_float(value):
		try:
			number = float(value)
		except (TypeError, ValueError):
			return None
		return number if math.isfinite(number) else None

	@classmethod
	def _display_number(cls, value):
		number = cls._finite_float(value)
		return f"{number:g}" if number is not None else "—"

	@classmethod
	def _display_cutoff(cls, value):
		number = cls._finite_float(value)
		if number is None:
			return "—"
		mantissa, exponent = f"{number:.0e}".split("e")
		return f"{mantissa}e{int(exponent)}"

	def _is_heim_state_fresh(self):
		return self._heim_state_fresh and time.monotonic() - self._heim_state_time <= 2.5

	def _refresh_heim_action_availability(self):
		fresh = self._is_heim_state_fresh()
		busy = self._heim_action_pending
		for button in self.servo_buttons.values():
			button.setEnabled(not busy)
		self.all_off_btn.setEnabled(not busy)
		self.offload_buttons["off"].setEnabled(not busy)
		delay_line = (self._heim_settings or {}).get("delay_line_type")
		for mode in ("gd", "mod"):
			self.offload_buttons[mode].setEnabled(
				not busy and (self.debug or fresh) and (not fresh or delay_line not in (None, "off"))
			)
		self.apply_kp_btn.setEnabled(not busy)
		self.stage_cutoff_btn.setEnabled(not busy and fresh)
		servo_off = (self._heim_settings or {}).get("servo_mode") == 4
		self.set_history_btn.setEnabled(not busy and fresh and servo_off)

	def _render_heim_state(self):
		settings = self._heim_settings or {}
		ddspc = self._heim_ddspc or {}
		configured = ddspc.get("configured") or {}
		active = ddspc.get("active") or {}
		fresh = self._is_heim_state_fresh()
		suffix = "" if fresh or not settings else " (stale)"
		servo_mode = settings.get("servo_mode")
		offload_mode = settings.get("offload_mode")
		servo_text = HEIM_SERVO_MODES.get(servo_mode, f"Other ({servo_mode})" if servo_mode is not None else "Unknown")
		offload_text = HEIM_OFFLOAD_MODES.get(offload_mode, f"Other ({offload_mode})" if offload_mode is not None else "Unknown")
		if offload_mode == 0 and servo_mode == 4:
			offload_text = "Nested (inactive: servo off)"
		self.servo_status_label.setText(servo_text + suffix)
		self.offload_status_label.setText(offload_text + suffix)
		self.delay_line_status_label.setText(f"Delay line: {settings.get('delay_line_type', 'Unknown')}{suffix}")
		for mode, code in (("off", 4), ("on", 1), ("ddspc", 5)):
			self.servo_buttons[mode].setChecked(fresh and servo_mode == code)
		for mode, code in (("off", 2), ("gd", 1), ("mod", 4)):
			self.offload_buttons[mode].setChecked(fresh and offload_mode == code)
		self.kp_status_label.setText(f"Live: {self._display_number(settings.get('kp'))}{suffix}")
		self.cutoff_status_label.setText(
			f"conf: {self._display_cutoff(configured.get('reg_cutoff'))} | "
			f"cur: {self._display_cutoff(active.get('reg_cutoff'))}{suffix}"
		)
		def history_text(profile):
			if profile.get("history") is None or profile.get("future") is None:
				return "—"
			return f"{profile['history']}/{profile['future']}"
		self.history_status_label.setText(
			f"conf: {history_text(configured)} | cur: {history_text(active)}{suffix}"
		)
		self._refresh_heim_action_availability()

	def _sync_heim_inputs(self):
		settings = self._heim_settings or {}
		configured = (self._heim_ddspc or {}).get("configured") or {}
		kp = self._finite_float(settings.get("kp"))
		if self._pending_kp is not None and kp is not None:
			value, submitted = self._pending_kp
			if math.isclose(kp, value, rel_tol=1e-9, abs_tol=1e-12):
				if self.kp_edit.text() == submitted:
					self._kp_dirty = False
				self._pending_kp = None
		if not self._kp_dirty and kp is not None and not self.kp_edit.hasFocus():
			text = f"{kp:g}"
			if self.kp_edit.text() != text:
				self.kp_edit.blockSignals(True)
				self.kp_edit.setText(text)
				self.kp_edit.blockSignals(False)

		cutoff = self._finite_float(configured.get("reg_cutoff"))
		if self._pending_cutoff is not None and cutoff is not None:
			value, submitted = self._pending_cutoff
			if math.isclose(cutoff, value, rel_tol=1e-9, abs_tol=1e-12):
				if self.cutoff_edit.text() == submitted:
					self._cutoff_dirty = False
				self._pending_cutoff = None
		if not self._cutoff_dirty and cutoff is not None and not self.cutoff_edit.hasFocus():
			text = f"{cutoff:g}"
			if self.cutoff_edit.text() != text:
				self.cutoff_edit.blockSignals(True)
				self.cutoff_edit.setText(text)
				self.cutoff_edit.blockSignals(False)

		configured_pair = (configured.get("history"), configured.get("future"))
		if self._pending_history is not None and configured_pair == self._pending_history:
			if self.history_combo.currentData() == configured_pair:
				self._history_dirty = False
			self._pending_history = None
		if not self._history_dirty:
			index = next(
				(i for i in range(self.history_combo.count()) if self.history_combo.itemData(i) == configured_pair),
				-1,
			)
			if index >= 0:
				self.history_combo.blockSignals(True)
				self.history_combo.setCurrentIndex(index)
				self.history_combo.blockSignals(False)

	def _on_heim_state(self, settings, ddspc):
		self._heim_settings = settings
		self._heim_ddspc = ddspc
		self._heim_state_fresh = True
		self._heim_state_time = time.monotonic()
		self._sync_heim_inputs()
		self._render_heim_state()
		if not self._heim_action_pending:
			self.heim_status_label.setText("Live")

	def _on_heim_state_failed(self, message):
		self._heim_state_fresh = False
		self._render_heim_state()
		self.heim_status_label.setText(f"Stale: {message}")

	def _check_heim_stale(self):
		if self._heim_state_fresh and not self._is_heim_state_fresh():
			self._on_heim_state_failed("no recent update")

	def _on_heim_command_finished(self, label, success, detail):
		self._heim_action_pending = False
		if not success:
			if label == "Apply Kp":
				self._pending_kp = None
			elif label == "Stage cutoff":
				self._pending_cutoff = None
			elif label == "Set History":
				self._pending_history = None
		if success:
			self.heim_status_label.setText(f"Live: {label} applied")
		else:
			stale = " (state stale)" if not self._is_heim_state_fresh() else ""
			self.heim_status_label.setText(f"Error: {label}: {detail}{stale}")
		self._refresh_heim_action_availability()

	def _queue_heim_action(self, commands, label):
		if self._heim_action_pending:
			return
		if self.debug:
			self._render_heim_state()
			for command in commands:
				self._append_log(f"[DEBUG] {command}")
			self.heim_status_label.setText(f"Debug: {label} not sent")
			return
		self._heim_action_pending = True
		self.heim_status_label.setText(f"Pending: {label}")
		self._render_heim_state()
		self._heim_worker.enqueue(commands, label)

	def _heim_validation_error(self, message):
		self.heim_status_label.setText(f"Error: {message}")
		self._append_log(f"[WARN] {message}")

	def _apply_kp(self):
		if self._heim_action_pending:
			return
		value = self._finite_float(self.kp_edit.text().strip())
		if value is None:
			self._heim_validation_error("Kp must be a finite number")
			return
		self._kp_dirty = True
		self._pending_kp = (value, self.kp_edit.text())
		self._queue_heim_action([f"gain {value!r}"], "Apply Kp")

	def _stage_cutoff(self):
		if self._heim_action_pending:
			return
		value = self._finite_float(self.cutoff_edit.text().strip())
		if value is None or value <= 0:
			self._heim_validation_error("DDSPC cutoff must be a positive finite number")
			return
		configured = (self._heim_ddspc or {}).get("configured") or {}
		reg_start = self._finite_float(configured.get("reg_start"))
		if not self._is_heim_state_fresh() or reg_start is None:
			self._heim_validation_error("Live DDSPC settings are required to stage cutoff")
			return
		if value > reg_start:
			self._heim_validation_error(f"DDSPC cutoff must be no greater than reg_start ({reg_start:g})")
			return
		self._cutoff_dirty = True
		self._pending_cutoff = (value, self.cutoff_edit.text())
		self._queue_heim_action([f'ddspc "set-reg-cutoff" {value!r}'], "Stage cutoff")

	def _set_history(self):
		if self._heim_action_pending:
			return
		if not self._is_heim_state_fresh() or (self._heim_settings or {}).get("servo_mode") != 4:
			self._heim_validation_error("Set History requires a fresh Servo Off state")
			return
		pair = self.history_combo.currentData()
		if pair not in DDSPC_HISTORY_PRESETS:
			self._heim_validation_error("Unsupported DDSPC history preset")
			return
		history, future = pair
		self._pending_history = pair
		self._queue_heim_action([f'ddspc "set-history" [{history},{future}]'], "Set History")

	def _on_mode_changed(self, index):
		if index == 0:
			return

		script_arg = "standard" if index == 1 else "faint"
		self._run_script("b-mode", [script_arg])
		self.mode_combo.blockSignals(True)
		self.mode_combo.setCurrentIndex(0)
		self.mode_combo.blockSignals(False)

	def _on_blf_status(self, status):
		self.baldr_status_label.setText(status)
		if status in {"Standard", "Faint"} and status != self.current_baldr_mode:
			self.current_baldr_mode = status
			self._refresh_baldr_sockets()

	def _refresh_baldr_sockets(self):
		for socket in self.baldr_sockets:
			socket.setsockopt(zmq.LINGER, 0)
			socket.close()

		self.baldr_sockets = []
		base = 6662 if self.current_baldr_mode == "Standard" else 6671
		ports = [base + i for i in range(4)]
		for port in ports:
			self.baldr_sockets.append(self._build_socket(port))

		self._append_log(
			f"[INFO] Connected Baldr sockets for {self.current_baldr_mode} mode on {self.host}: {', '.join(str(p) for p in ports)}"
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
		if self.mds_socket is old_socket:
			self.mds_socket = new_socket
		if self.cam_server_socket is old_socket:
			self.cam_server_socket = new_socket
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
		cmds = ["set_gd_threshold 100", "search 0.7, 50", "dls 0,0,0,0"]
		for cmd in cmds:
			self._send_cmd(self.heim_socket, cmd)

	def _set_gd_threshold(self):
		value = self.gd_spin.value()
		self._send_cmd(self.heim_socket, f"set_gd_threshold {value}")

	def _make_dark(self):
		prev_SBB_state = self._send_cmd(self.mds_socket, "read SBB")
		self._send_cmd(self.mds_socket, "off SBB")
		self._append_log(f"[INFO] Light source off. Sending SSFs to up.")
		prev_SSF_states = []
		num_flippers=0
		for i in range(1, 5):
			prev_SSF_states += [self._send_cmd(self.mds_socket, f"read SSF{i}")]
			if (prev_SSF_states[i - 1] != "IN"):
				mds_cmd = f"moveabs SSF{i} 1.0"
				self._send_cmd(self.mds_socket, f"{mds_cmd}")
			else:
				num_flippers += 1
		#If no flippers, we need to wait for the light source
		if num_flippers==0:
			time.sleep(2)
		#In case of a slow camera mode, a small delay
		time.sleep(0.5)
  		#Now make the dark.
		self._append_log(f"[INFO] Making Dark.")
		self._send_cmd(self.cam_server_socket, "make_dark")
		# Restore SBB state
		if int(prev_SBB_state) == 1:
			self._send_cmd(self.mds_socket, "on SBB")
		# Restore SSF states
		self._append_log(f"[INFO] Restoring previous SSF states.")
		for i in range(1, 5):
			if prev_SSF_states[i - 1].startswith("OUT"):
				mds_cmd = f"moveabs SSF{i} 0.0"
				self._send_cmd(self.mds_socket, f"{mds_cmd}")
		self._append_log(f"[INFO] Done making dark!")

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

	def _run_recon(self):
		if self._recon_stop_requested:
			return
		if self.current_baldr_mode != "Faint":
			self._append_log("[WARN] Recon is only available in Faint mode.")
			return

		if hasattr(self, "recon_worker") and self.recon_worker.isRunning():
			return

		self._recon_generation += 1
		self.recon_status_label.setText("waiting for reconstructor construction")
		self.recon_btn.setEnabled(False)
		for command in ['servo "off"', "zero_tt", 'servo "tt"', "auto_coupling"]:
			self._send_baldr_all(command)
			time.sleep(0.5)

		self.recon_worker = ReconWorker(self.debug)
		self.recon_worker.log_message.connect(self._append_log)
		generation = self._recon_generation
		self.recon_worker.recon_finished.connect(
			lambda success: self._on_recon_finished(success, generation)
		)
		self.recon_worker.start()

	def _on_recon_finished(self, success, generation):
		if self._recon_stop_requested or generation != self._recon_generation:
			return
		if not success:
			self.recon_status_label.setText("Recon failed")
			self.recon_btn.setEnabled(True)
			return

		QtCore.QTimer.singleShot(500, lambda: self._send_recon_command(generation))

	def _send_recon_command(self, generation):
		if self._recon_stop_requested or generation != self._recon_generation:
			return
		self._send_baldr_all("recon")
		self.recon_status_label.setText("Recon complete")
		self.recon_btn.setEnabled(True)

	def _stop_recon(self):
		if hasattr(self, "stop_recon_worker") and self.stop_recon_worker.isRunning():
			return
		self._recon_generation += 1
		self._recon_stop_requested = True
		if hasattr(self, "recon_worker"):
			self.recon_worker.request_stop()
		self.recon_btn.setEnabled(False)
		self.stop_recon_btn.setEnabled(False)
		self.recon_status_label.setText("Stopping recon and clearing DM channels...")
		self.stop_recon_worker = StopReconWorker(host=RECON_HOST, debug=self.debug)
		self.stop_recon_worker.log_message.connect(self._append_log)
		self.stop_recon_worker.stop_finished.connect(self._on_stop_recon_finished)
		self.stop_recon_worker.start()

	def _on_stop_recon_finished(self, success):
		status = "Debug: no action" if self.debug else ("Recon stopped" if success else "Recon stop incomplete")
		self.recon_status_label.setText(status)
		self.stop_recon_btn.setEnabled(True)
		self._recon_stop_requested = False
		self.recon_btn.setEnabled(success)

	def _run_script(self, script, args):
		cmd = ["/home/asg/.conda/envs/asgard/bin/" + script, *args]
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
		if hasattr(self, "stop_recon_worker") and self.stop_recon_worker.isRunning():
			if not self.stop_recon_worker.wait(10000):
				event.ignore()
				return
		self._heim_stale_timer.stop()
		if self._heim_worker.isRunning():
			self._heim_worker.stop()
			if not self._heim_worker.wait(5000):
				event.ignore()
				return
		self._poller.stop()
		self._poller.wait(2000)
		self.heim_socket.setsockopt(zmq.LINGER, 0)
		self.heim_socket.close()
		self.mds_socket.setsockopt(zmq.LINGER, 0)
		self.mds_socket.close()
		self.cam_server_socket.setsockopt(zmq.LINGER, 0)
		self.cam_server_socket.close()
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

	app = QtWidgets.QApplication([])
	win = ShortcutsGUI(host=args.host, debug=args.debug)
	win.show()
	sys.exit(app.exec_())


if __name__ == "__main__":
	main()
