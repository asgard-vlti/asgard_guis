import subprocess
import sys

import zmq
from PyQt5 import QtCore, QtWidgets


class BaldrPupilMaskAlignmentGUI(QtWidgets.QWidget):
	def __init__(self, host="mimir"):
		super().__init__()
		self.host = host
		self.move_delta = 1.0
		self.context = zmq.Context()
		self.mds_socket = self._build_socket()
		self.cam_server_socket = self._build_socket(6667)
		self.setWindowTitle("BALDR Pupil/Mask Alignment")
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
		self.beam_combo.currentTextChanged.connect(self._change_beam)
		header_layout.addWidget(self.beam_combo)

		self.align_pupil_radio = QtWidgets.QRadioButton("Align Pupil")
		self.align_mask_radio = QtWidgets.QRadioButton("Align Mask")
		self.align_mask_radio.setChecked(True)
		self.align_pupil_radio.toggled.connect(self._change_alignment_mode)
		header_layout.addWidget(self.align_pupil_radio)
		header_layout.addWidget(self.align_mask_radio)
		header_layout.addWidget(QtWidgets.QLabel("Mask:"))
		self.mask_combo = QtWidgets.QComboBox()
		self.mask_combo.addItems(["H2", "H3", "H4", "H5", "J1", "J2", "J3", "J4"])
		self.mask_combo.setCurrentText("H3")
		self.mask_combo.currentTextChanged.connect(self._move_to_mask)
		header_layout.addWidget(self.mask_combo)

		self.move_delta_label = QtWidgets.QLabel()
		self.move_delta_label.setFixedWidth(
			self.move_delta_label.fontMetrics().horizontalAdvance("Move Delta: 0.0009765625")
		)
		self._update_move_delta_label()
		header_layout.addStretch()
		root_layout.addLayout(header_layout)

		layout = QtWidgets.QGridLayout()
		layout.setAlignment(QtCore.Qt.AlignLeft)
		layout.setHorizontalSpacing(4)
		for column in range(6):
			layout.setColumnMinimumWidth(column, 36)
			layout.setColumnStretch(column, 0)
		self.left_control = self._add_direction_controls(
			layout, "Image/Cold Stop", 0, 0, self._move_image_cold_stop
		)
		self.right_control = self._add_direction_controls(
			layout, "Pupil", 0, 3, self._move_pupil
		)
		layout.addWidget(self.move_delta_label, 1, 6)

		self.more_btn = QtWidgets.QPushButton("More")
		self.more_btn.setFixedWidth(65)
		self.more_btn.clicked.connect(lambda: self._change_move_delta(2))
		layout.addWidget(self.more_btn, 2, 6)
		self.less_btn = QtWidgets.QPushButton("Less")
		self.less_btn.setFixedWidth(65)
		self.less_btn.clicked.connect(lambda: self._change_move_delta(0.5))
		layout.addWidget(self.less_btn, 3, 6)

		self.save_btn = QtWidgets.QPushButton("Save all")
		self.save_btn.clicked.connect(lambda: self._send_mds_command("fpm_write -1"))
		layout.addWidget(self.save_btn, 5, 0, 1, 3)
		self.update_beam_btn = QtWidgets.QPushButton("Update Beam")
		self.update_beam_btn.clicked.connect(self._update_beam)
		layout.addWidget(self.update_beam_btn, 5, 3, 1, 3)
		self.all_masks_checkbox = QtWidgets.QCheckBox("All masks?")
		layout.addWidget(self.all_masks_checkbox, 5, 6)
		root_layout.addLayout(layout)
		self._update_controls()

	def _add_direction_controls(self, layout, label, row, column, handler):
		label_widget = QtWidgets.QLabel(label)
		layout.addWidget(label_widget, row, column, 1, 3, QtCore.Qt.AlignCenter)

		buttons = []
		for arrow_type, direction in (
			(QtCore.Qt.UpArrow, "up"),
			(QtCore.Qt.LeftArrow, "left"),
			(QtCore.Qt.RightArrow, "right"),
			(QtCore.Qt.DownArrow, "down"),
		):
			button = self._arrow_button(arrow_type)
			button.clicked.connect(lambda checked=False, direction=direction: handler(direction))
			buttons.append(button)

		layout.addWidget(buttons[0], row + 1, column + 1)
		layout.addWidget(buttons[1], row + 2, column)
		layout.addWidget(buttons[2], row + 2, column + 2)
		layout.addWidget(buttons[3], row + 3, column + 1)
		return label_widget, buttons

	def _arrow_button(self, arrow_type):
		button = QtWidgets.QToolButton()
		button.setArrowType(arrow_type)
		button.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
		button.setFixedSize(36, 30)
		return button

	def _update_controls(self):
		aligning_pupil = self.align_pupil_radio.isChecked()
		self.left_control[0].setText("Image/Cold Stop" if aligning_pupil else "Phase Mask")
		right_label = (
			"CRed box"
			if aligning_pupil and self.beam_combo.currentText() == "1"
			else "Pupil" if aligning_pupil else ""
		)
		self.right_control[0].setText(right_label)
		for button in self.right_control[1]:
			button.setEnabled(aligning_pupil)
		self.mask_combo.setEnabled(not aligning_pupil)

	def _change_alignment_mode(self, aligning_pupil):
		beam = self.beam_combo.currentText()
		self._send_mds_command(
			f"moverel BMY{beam} {500.0 if aligning_pupil else -500.0:.1f}"
		)
		self._update_controls()

	def _change_beam(self):
		if self.align_pupil_radio.isChecked():
			beam = self.beam_combo.currentText()
			self._send_mds_command(f"moverel BMY{beam} -500.0")
		self._update_controls()

	def _change_move_delta(self, factor):
		self.move_delta *= factor
		self._update_move_delta_label()

	def _update_move_delta_label(self):
		self.move_delta_label.setText(f"Move Delta: {self.move_delta:g}")

	def _build_socket(self, port=5555):
		socket = self.context.socket(zmq.REQ)
		socket.setsockopt(zmq.SNDTIMEO, 1500)
		socket.setsockopt(zmq.RCVTIMEO, 2000)
		socket.connect(f"tcp://{self.host}:{port}")
		return socket

	def _move_image_cold_stop(self, direction):
		beam = int(self.beam_combo.currentText())
		x, y = self._movement_vector(direction)
		x *= self.move_delta
		y *= self.move_delta
		if beam == 1:
			motor = "BTP1" if x else "BTT1"
			distance = (x or y) * 0.06
			self._send_mds_command(f"moverel {motor} {distance:g}")
			return
		self._send_mds_command(
			f"mv_img baldr {beam} {x * 0.2:.1f} {y * 0.2:.1f}"
		)

	def _move_pupil(self, direction):
		beam = int(self.beam_combo.currentText())
		x, y = self._movement_vector(direction)
		x *= self.move_delta
		y *= self.move_delta
		if beam == 1:
			self._send_camera_command(
				f"move_roi {x * 2:g} {y * 2:g}"
			)
			return
		self._send_mds_command(
			f"mv_pup baldr {beam} {y * 2:.1f} {x * 2:.1f}"
		)

	def _movement_vector(self, direction):
		return {
			"up": (0.0, 1.0),
			"down": (0.0, -1.0),
			"left": (-1.0, 0.0),
			"right": (1.0, 0.0),
		}[direction]

	def _update_beam(self):
		beam = self.beam_combo.currentText()
		update_scope = "all" if self.all_masks_checkbox.isChecked() else "one"
		self._send_mds_command(
			f"fpm_update {beam} {self.mask_combo.currentText()} {update_scope}"
		)

	def _move_to_mask(self, mask):
		for beam in range(1, 5):
			self._send_mds_command(f"fpm_movetomask {beam} {mask}")

	def _send_mds_command(self, command):
		self._send_command(self.mds_socket, command)

	def _send_camera_command(self, command):
		self._send_command(self.cam_server_socket, command)

	def _send_command(self, socket, command):
		try:
			socket.send_string(command)
			socket.recv_string()
		except zmq.error.Again:
			print(f"[TIMEOUT] {command}")
		except Exception as exc:
			print(f"[ERROR] {command} -> {exc}")

	def closeEvent(self, event):
		self.mds_socket.setsockopt(zmq.LINGER, 0)
		self.mds_socket.close()
		self.cam_server_socket.setsockopt(zmq.LINGER, 0)
		self.cam_server_socket.close()
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
			QToolButton:disabled {
				background-color: #25272b;
				border-color: #363941;
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
			QRadioButton::indicator {
				width: 16px;
				height: 16px;
				border: 1px solid #8c919a;
				background-color: #272a30;
			}
			QRadioButton::indicator:checked {
				background-color: #3f6db3;
			}
			QCheckBox::indicator {
				width: 16px;
				height: 16px;
				border: 2px solid #e6e6e6;
				border-radius: 2px;
				background-color: #272a30;
			}
			QCheckBox::indicator:checked {
				background-color: #3f6db3;
			}
			QCheckBox::indicator:disabled {
				border-color: #4a4f57;
				background-color: #25272b;
			}
			"""
		)


def main():
	app = QtWidgets.QApplication(sys.argv)
	window = BaldrPupilMaskAlignmentGUI()
	window.show()
	sys.exit(app.exec_())


if __name__ == "__main__":
	main()