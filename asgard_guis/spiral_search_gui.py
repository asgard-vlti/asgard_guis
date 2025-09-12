import sys
import argparse
from PyQt5 import QtWidgets, QtCore
from asgard_guis.spiral_search import SpiralSearchIntegrator


class SpiralSearchGUI(QtWidgets.QWidget):
    def __init__(self, debug=False):
        super().__init__()
        self.setWindowTitle("Image search GUI")
        self.integrator = SpiralSearchIntegrator(debug=debug)
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()

        # Beam selection
        beam_layout = QtWidgets.QHBoxLayout()
        beam_label = QtWidgets.QLabel("Select Beam:")
        self.beam_combo = QtWidgets.QComboBox()
        self.beam_combo.addItems([str(i) for i in range(1, 5)])
        beam_layout.addWidget(beam_label)
        beam_layout.addWidget(self.beam_combo)
        layout.addLayout(beam_layout)

        # Step size
        step_layout = QtWidgets.QHBoxLayout()
        step_label = QtWidgets.QLabel("Step Size:")
        self.step_box = QtWidgets.QDoubleSpinBox()
        self.step_box.setDecimals(1)
        self.step_box.setSingleStep(5.0)
        self.step_box.setValue(40.0)
        self.step_box.setMinimum(0.01)
        step_layout.addWidget(step_label)
        step_layout.addWidget(self.step_box)
        layout.addLayout(step_layout)

        # X/Y controls
        grid = QtWidgets.QGridLayout()
        self.x_plus_btn = QtWidgets.QPushButton("-X (Right)")
        self.x_minus_btn = QtWidgets.QPushButton("+X (Left)")
        self.y_plus_btn = QtWidgets.QPushButton("-Y (Up)")
        self.y_minus_btn = QtWidgets.QPushButton("+Y (Down)")
        # Arrange so +Y is up, +X is right
        grid.addWidget(self.y_plus_btn, 0, 1)  # Up
        grid.addWidget(self.x_minus_btn, 1, 0)  # Left
        grid.addWidget(self.x_plus_btn, 1, 2)  # Right
        grid.addWidget(self.y_minus_btn, 2, 1)  # Down
        layout.addLayout(grid)

        # Done button
        self.done_btn = QtWidgets.QPushButton("Done")
        layout.addWidget(self.done_btn)

        self.setLayout(layout)

        # Connect signals
        # +X is right, -X is left, +Y is up, -Y is down
        self.x_plus_btn.clicked.connect(lambda: self.move_offset(-1, 0))  # Right
        self.x_minus_btn.clicked.connect(lambda: self.move_offset(1, 0))  # Left
        self.y_plus_btn.clicked.connect(lambda: self.move_offset(0, -1))  # Up
        self.y_minus_btn.clicked.connect(lambda: self.move_offset(0, 1))  # Down
        self.done_btn.clicked.connect(self.finish_and_close)

    def move_offset(self, dx, dy):
        beam = int(self.beam_combo.currentText())
        step = self.step_box.value()
        # +X is right (positive real X), -X is left (negative real X)
        # +Y is up (negative real Y), -Y is down (positive real Y)
        x = dx * step
        y = -dy * step
        self.integrator.send_image_relative_offset(beam, x, y)

    def finish_and_close(self):
        self.integrator.write_pointing_offsets_to_db()
        self.close()


def main():
    parser = argparse.ArgumentParser(description="Spiral Search GUI")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (print commands instead of running)",
    )
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    win = SpiralSearchGUI(debug=args.debug)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
