import sys

from PyQt5 import QtCore, QtWidgets


class BaldrFaintAlignmentGUI(QtWidgets.QWidget):
	def __init__(self):
		super().__init__()
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

		self.spherical_checkbox = QtWidgets.QCheckBox("Spherical?")
		layout.addWidget(self.spherical_checkbox, 0, 2, 1, 2)

		self._add_direction_controls(layout, "Field Lens", 1, 0)
		self._add_direction_controls(layout, "Pupil", 1, 3)

		self.align_btn = QtWidgets.QPushButton()
		self._update_align_button(self.beam_combo.currentText())
		layout.addWidget(self.align_btn, 4, 0, 1, 3)

		self.save_btn = QtWidgets.QPushButton("Save")
		layout.addWidget(self.save_btn, 4, 3, 1, 3)

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
			"""
		)


def main():
	app = QtWidgets.QApplication([])
	window = BaldrFaintAlignmentGUI()
	window.show()
	sys.exit(app.exec_())


if __name__ == "__main__":
	main()