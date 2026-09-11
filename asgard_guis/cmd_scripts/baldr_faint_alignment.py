import sys

from PyQt5 import QtCore, QtWidgets


class BaldrFaintAlignmentGUI(QtWidgets.QWidget):
	def __init__(self):
		super().__init__()
		self.move_delta = 1.0
		self.setWindowTitle("Baldr Faint Alignment")
		self._apply_dark_theme()
		self._init_ui()
		self.adjustSize()
		self.setMinimumSize(self.minimumSizeHint())

	def _init_ui(self):
		layout = QtWidgets.QGridLayout(self)

		layout.addWidget(QtWidgets.QLabel("Beam:"), 0, 0)
		self.beam_combo = QtWidgets.QComboBox()
		self.beam_combo.addItems(["1", "2", "3", "4"])
		self.beam_combo.currentTextChanged.connect(self._update_align_button)
		layout.addWidget(self.beam_combo, 0, 1)

		layout.addWidget(QtWidgets.QLabel("Spherical?"), 0, 2)
		self.spherical_checkbox = QtWidgets.QCheckBox()
		layout.addWidget(self.spherical_checkbox, 0, 3)

		self.move_delta_label = QtWidgets.QLabel()
		self._update_move_delta_label()
		layout.addWidget(self.move_delta_label, 0, 4, 1, 2)

		self._add_direction_controls(layout, "Field Lens", 1, 0)
		self._add_direction_controls(layout, "Pupil", 1, 3)
		self.more_btn = QtWidgets.QPushButton("More")
		self.more_btn.clicked.connect(lambda: self._change_move_delta(2))
		layout.addWidget(self.more_btn, 2, 6)
		self.less_btn = QtWidgets.QPushButton("Less")
		self.less_btn.clicked.connect(lambda: self._change_move_delta(0.5))
		layout.addWidget(self.less_btn, 3, 6)

		self.align_btn = QtWidgets.QPushButton()
		self._update_align_button(self.beam_combo.currentText())
		layout.addWidget(self.align_btn, 6, 0, 1, 3)

		self.save_btn = QtWidgets.QPushButton("Save")
		layout.addWidget(self.save_btn, 6, 3, 1, 3)

	def _add_direction_controls(self, layout, label, row, column):
		layout.addWidget(QtWidgets.QLabel(label), row, column, 1, 3, QtCore.Qt.AlignCenter)

		up_button = self._arrow_button(QtCore.Qt.UpArrow)
		left_button = self._arrow_button(QtCore.Qt.LeftArrow)
		right_button = self._arrow_button(QtCore.Qt.RightArrow)
		down_button = self._arrow_button(QtCore.Qt.DownArrow)

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

	def _change_move_delta(self, factor):
		self.move_delta *= factor
		self._update_move_delta_label()

	def _update_move_delta_label(self):
		self.move_delta_label.setText(f"Move Delta: {self.move_delta:g}")

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
	app = QtWidgets.QApplication([])
	window = BaldrFaintAlignmentGUI()
	window.show()
	sys.exit(app.exec_())


if __name__ == "__main__":
	main()