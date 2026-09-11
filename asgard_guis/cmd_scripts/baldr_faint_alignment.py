import subprocess
import sys

import zmq
from PyQt5 import QtCore, QtWidgets


class BaldrFaintAlignmentGUI(QtWidgets.QWidget):
	def __init__(self, host="mimir"):
		super().__init__()
		self.host = host
		self.move_delta = 1.0
		self.context = zmq.Context()
		self.mds_socket = self._build_socket()
		self.setWindowTitle("Baldr Faint Alignment")
		self._apply_dark_theme()
		self._init_ui()
		self.adjustSize()
		self.setMinimumSize(self.minimumSizeHint())

	def _init_ui(self):
		root_layout = QtWidgets.QVBoxLayout(self)
		header_layout = QtWidgets.QHBoxLayout()

		header_layout.addWidget(QtWidgets.QLabel("Beam:"))
		self.beam_combo = QtWidgets.QComboBox()
		self.beam_combo.addItems(["1", "2", "3", "4"])
		self.beam_combo.currentTextChanged.connect(self._update_align_button)
		header_layout.addWidget(self.beam_combo)

		header_layout.addWidget(QtWidgets.QLabel("Spherical?"))
		self.spherical_checkbox = QtWidgets.QCheckBox()
		self.spherical_checkbox.toggled.connect(self._set_spherical)
		header_layout.addWidget(self.spherical_checkbox)

		self.move_delta_label = QtWidgets.QLabel()
		self.move_delta_label.setFixedWidth(
			self.move_delta_label.fontMetrics().horizontalAdvance("Move Delta: 0.0009765625")
		)
		self._update_move_delta_label()
		header_layout.addWidget(self.move_delta_label)
		header_layout.addStretch()
		root_layout.addLayout(header_layout)

		layout = QtWidgets.QGridLayout()

		self._add_direction_controls(
			layout,
			"Field Lens",
			0,
			0,
			self._move_field_lens,
		)
		self._add_direction_controls(layout, "Pupil", 0, 3, self._move_pupil)
		self.more_btn = QtWidgets.QPushButton("More")
		self.more_btn.clicked.connect(lambda: self._change_move_delta(2))
		layout.addWidget(self.more_btn, 1, 6)
		self.less_btn = QtWidgets.QPushButton("Less")
		self.less_btn.clicked.connect(lambda: self._change_move_delta(0.5))
		layout.addWidget(self.less_btn, 2, 6)

		self.align_btn = QtWidgets.QPushButton()
		self.align_btn.clicked.connect(self._align)
		self._update_align_button(self.beam_combo.currentText())
		layout.addWidget(self.align_btn, 5, 0, 1, 3)

		self.save_btn = QtWidgets.QPushButton("Save")
		self.save_btn.clicked.connect(lambda: self._run_script("b-savemode", ["FAINT"]))
		layout.addWidget(self.save_btn, 5, 3, 1, 3)
		root_layout.addLayout(layout)

	def _add_direction_controls(self, layout, label, row, column, handler=None):
		layout.addWidget(QtWidgets.QLabel(label), row, column, 1, 3, QtCore.Qt.AlignCenter)

		up_button = self._arrow_button(QtCore.Qt.UpArrow)
		left_button = self._arrow_button(QtCore.Qt.LeftArrow)
		right_button = self._arrow_button(QtCore.Qt.RightArrow)
		down_button = self._arrow_button(QtCore.Qt.DownArrow)
		if handler is not None:
			up_button.clicked.connect(lambda: handler("up"))
			left_button.clicked.connect(lambda: handler("left"))
			right_button.clicked.connect(lambda: handler("right"))
			down_button.clicked.connect(lambda: handler("down"))

		layout.addWidget(up_button, row + 1, column + 1)
		layout.addWidget(left_button, row + 2, column)
		layout.addWidget(right_button, row + 2, column + 2)
		layout.addWidget(down_button, row + 3, column + 1)

	def _arrow_button(self, arrow_type):
		button = QtWidgets.QToolButton()
		button.setArrowType(arrow_type)
		button.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
		button.setFixedSize(36, 30)
		return button

	def _update_align_button(self, beam):
		label = "Align CRed box" if beam == "1" else "Align BOTX"
		self.align_btn.setText(label)

	def _align(self):
		beam = self.beam_combo.currentText()
		script = "b-movebox" if beam == "1" else "b-faint-tweak-botx"
		self._run_script(script, [beam])

	def _change_move_delta(self, factor):
		self.move_delta *= factor
		self._update_move_delta_label()

	def _update_move_delta_label(self):
		self.move_delta_label.setText(f"Move Delta: {self.move_delta:g}")

	def _set_spherical(self, enabled):
		if enabled:
			beam = self.beam_combo.currentText()
			self._run_script("dm-zernike", [beam, "10", "0.1"])

	def _run_script(self, script, args):
		command = ["/home/asg/.conda/envs/asgard/bin/" + script, *args]
		try:
			subprocess.Popen(command)
		except OSError as exc:
			print(f"[ERROR] failed to launch {' '.join(command)} -> {exc}")

	def _build_socket(self):
		socket = self.context.socket(zmq.REQ)
		socket.setsockopt(zmq.SNDTIMEO, 1500)
		socket.setsockopt(zmq.RCVTIMEO, 2000)
		socket.connect(f"tcp://{self.host}:5555")
		return socket

	def _move_field_lens(self, direction):
		beam = int(self.beam_combo.currentText())
		motor_axis, sign = {
			"up": ("BMY", -1),
			"down": ("BMY", 1),
			"left": ("BMX", -1),
			"right": ("BMX", 1),
		}[direction]
		if beam == 4 and motor_axis == "BMX":
			sign *= -1
        #Format the motion to have at least 1 decimal point.
		command = f"moverel {motor_axis}{beam} {sign * 10.0 * self.move_delta:.1f}"
		self._send_mds_command(command)

	def _move_pupil(self, direction):
		beam = int(self.beam_combo.currentText())
		pupil_moves = {
			1: {"right": (("BTP1", 0.08),), "up": (("BTT1", 0.08),)},
			2: {
				"right": (("BTP2", 0.07), ("BOTT2", 0.005)),
				"up": (("BTT2", 0.11), ("BOTP2", -0.005)),
			},
			3: {
				"right": (("BTP3", 0.04), ("BOTT3", 0.005)),
				"up": (("BTT3", 0.06), ("BOTP3", -0.005)),
			},
			4: {
				"right": (("BTP4", 0.015), ("BOTT4", 0.005)),
				"up": (("BTT4", 0.022), ("BOTP4", -0.005)),
			},
		}
		base_direction = "right" if direction in {"right", "left"} else "up"
		sign = -1 if direction in {"left", "down"} else 1
		for motor, distance in pupil_moves[beam][base_direction]:
			self._send_mds_command(
				f"moverel {motor} {sign * distance * self.move_delta:g}"
			)

	def _send_mds_command(self, command):
		try:
			self.mds_socket.send_string(command)
			self.mds_socket.recv_string()
		except zmq.error.Again:
			print(f"[TIMEOUT] {command}")
		except Exception as exc:
			print(f"[ERROR] {command} -> {exc}")

	def closeEvent(self, event):
		self.mds_socket.setsockopt(zmq.LINGER, 0)
		self.mds_socket.close()
		self.context.term()
		super().closeEvent(event)

	def _apply_dark_theme(self):
		self.setStyleSheet(
			"""
			QWidget {
				background-color: #1e1f22;
				color: #e6e6e6;
			}
			QPushButton, QToolButton {
				background-color: #2d3138;
				border: 1px solid #4a4f57;
				border-radius: 5px;
				padding: 5px 10px;
			}
			QPushButton:hover, QToolButton:hover {
				background-color: #383d45;
			}
			QPushButton:pressed, QToolButton:pressed {
				background-color: #24282e;
			}
			QComboBox {
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
			QCheckBox::indicator {
				width: 16px;
				height: 16px;
				border: 1px solid #8c919a;
				background-color: #272a30;
			}
			QCheckBox::indicator:checked {
				background-color: #3f6db3;
			}
			"""
		)


def main():
	app = QtWidgets.QApplication(sys.argv)
	window = BaldrFaintAlignmentGUI()
	window.show()
	sys.exit(app.exec_())


if __name__ == "__main__":
	main()