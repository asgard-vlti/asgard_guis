"""
Various methods of drawing scrolling plots using pyqtgraph for speed and simplicity.
"""

import asgard_guis.ZMQ_control_client as Z
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
import argparse

from statemachine import StateMachine, State

import heapq

N_TSCOPES = 4
N_BASELINES = 6


class HeimdallrStateMachine(StateMachine):
    searching = State("Searching", initial=True)
    sidelobe = State("Sidelobe")
    offload_gd = State("Offload GD")
    servo_on = State("Servo On")

    # linear state machine, states are in order:
    # searching, sidelobe, offload_gd, servo
    # can only move forward one step at a time
    # but can move back any number of steps:
    # forward steps:
    to_sidelobe_from_searching = searching.to(sidelobe)
    to_offload_gd_from_sidelobe = sidelobe.to(offload_gd)
    to_servo_from_offload_gd = offload_gd.to(servo_on)

    # backward steps:
    to_searching_from_sidelobe = sidelobe.to(searching)
    to_searching_from_offload_gd = offload_gd.to(searching)
    to_searching_from_servo = servo_on.to(searching)

    to_sidelobe_from_offload_gd = offload_gd.to(sidelobe)
    to_sidelobe_from_servo = servo_on.to(sidelobe)

    to_offload_gd_from_servo = servo_on.to(offload_gd)

    def __init__(
        self,
        status_keys,
        status_shapes,
        buffer_length,
        best_gd_SNR_ref,
        reset_best_gd_SNR_func,
        server,
        *args,
        **kwargs,
    ):
        self.threshold_lower = 8
        self.threshold_upper = 25
        self.status_keys = status_keys
        self.status_buffers = {}
        for k in status_keys:
            shape = status_shapes[k]
            dtype = np.float64
            if isinstance(shape, tuple):
                if len(shape) == 0:
                    self.status_buffers[k] = np.full(
                        (buffer_length,), np.nan, dtype=dtype
                    )
                elif len(shape) == 1:
                    self.status_buffers[k] = np.full(
                        (buffer_length, shape[0]), np.nan, dtype=dtype
                    )
                else:
                    self.status_buffers[k] = np.full(
                        (buffer_length,) + shape, np.nan, dtype=dtype
                    )
            else:
                self.status_buffers[k] = np.full((buffer_length,), np.nan, dtype=dtype)
        self._status_idx = 0  # rolling index for lookback

        self.servo_start_gain = 0.05
        self.servo_final_gain = 0.4

        # Shared best_gd_SNR and reset logic
        self.best_gd_SNR = best_gd_SNR_ref
        self.reset_best_gd_SNR = reset_best_gd_SNR_func

        self.server = server
        super().__init__(*args, **kwargs)

    def update_status_buffers(self, status):
        """
        Update rolling buffers with new status dict (from main GUI).
        """
        for k in self.status_keys:
            arr = np.array(status[k])
            buf = self.status_buffers[k]
            if arr.ndim == 0 or (arr.ndim == 1 and arr.shape == ()):  # scalar
                buf[self._status_idx] = arr
            else:
                buf[self._status_idx] = arr
        self._status_idx = (self._status_idx + 1) % self.n_lookback

    def set_threshold(self, value):
        self.server.send(f"set_gd_threshold {value}")
        print(f"Set threshold to {value}")

    def on_enter_searching(self, event):
        from_state = event.source if hasattr(event, "source") else None
        self.set_threshold(self.threshold_lower)
        self.server.send('offload "gd"')
        if from_state == self.offload_gd:
            pass
        elif from_state == self.sidelobe:
            pass
        elif from_state == self.servo_on:
            self.server.send('servo "off"')

        # Reset best_gd_SNR history (same as GUI reset button)
        self.reset_best_gd_SNR()

    def on_enter_sidelobe(self):
        # Operations to perform when entering 'sidelobe'
        self.set_threshold(self.threshold_upper)

        # kicks and see what happens to gd
        print("Kicks should go here...")

    def on_enter_offload_gd(self, event):
        from_state = event.source if hasattr(event, "source") else None
        if from_state == self.searching:
            self.set_threshold(self.threshold_lower)
        elif from_state == self.sidelobe:
            self.set_threshold(self.threshold_lower)
        elif from_state == self.servo_on:
            self.server.send('servo "off"')
            self.set_threshold(self.threshold_lower)

    def on_enter_servo_on(self):
        # Operations to perform when entering 'servo_on'
        self.server.send(f"set_gain {self.servo_start_gain}")
        self.server.send('servo "on"')

    @property
    def M(self):
        return np.array(
            [
                [-1, 1, 0, 0],
                [-1, 0, 1, 0],
                [-1, 0, 0, 1],
                [0, -1, 1, 0],
                [0, -1, 0, 1],
                [0, 0, -1, 1],
            ]
        )

    def poll_transitions(self):
        """
        Poll transition conditions and trigger transitions as needed.
        This should be called after update_status_buffers.
        """
        print("Current State:", self.current_state.name)

        # State transitions
        if self.current_state == self.searching:
            if self.should_go_to_offload_gd(from_state="searching"):
                self.to_offload_gd_from_searching()
            elif self.should_go_to_sidelobe():
                self.to_sidelobe()
        elif self.current_state == self.sidelobe:
            if self.should_go_to_offload_gd(from_state="sidelobe"):
                self.to_offload_gd_from_sidelobe()
            elif self.should_go_to_searching():
                self.to_searching()
        elif self.current_state == self.offload_gd:
            if self.should_go_to_servo_on():
                self.to_servo_on()
            elif self.should_go_to_searching():
                self.to_searching()
            elif self.should_go_to_sidelobe():
                self.to_sidelobe()
        elif self.current_state == self.servo_on:
            if self.should_go_to_offload_gd(from_state="servo_on"):
                self.to_offload_gd_from_servo_on()
            elif self.should_go_to_searching():
                self.to_searching()

    # Placeholder condition methods
    def should_go_to_sidelobe(self):
        # check if the gd_snr is on average between threshold_lower and threshold_upper
        buf = self.status_buffers.get("gd_snr")
        if buf is not None:
            avg_gd_snr = np.mean(buf)
            return self.threshold_lower < avg_gd_snr < self.threshold_upper

    def should_go_to_offload_gd(self, from_state):
        if from_state == "searching":
            # check if median gd_snr for all baselines exceeds lower threshold
            buf = self.status_buffers.get("gd_snr")
            if buf is not None:
                median_gd_snr = np.median(buf, axis=0)
                return np.all(median_gd_snr > self.threshold_lower)

            # alternatively could check if history had 4 or 5 baselines and do the calc...
        elif from_state == "sidelobe":
            # check if median gd_snr for all baselines exceeds upper threshold
            buf = self.status_buffers.get("gd_snr")
            if buf is not None:
                median_gd_snr = np.median(buf, axis=0)
                return np.all(median_gd_snr > self.threshold_upper)
        elif from_state == "servo_on":
            # if all baselines gd_snr drop below lower threshold
            buf = self.status_buffers.get("gd_snr")
            if buf is not None:
                median_gd_snr = np.median(buf, axis=0)
                return np.all(median_gd_snr < self.threshold_lower)

    def should_go_to_servo_on(self):
        # if gd_snr has been above upper threshold over 95% of samples in the lookback
        buf = self.status_buffers.get("gd_snr")
        if buf is not None:
            above_threshold = buf > self.threshold_upper
            fraction_above = np.mean(above_threshold, axis=0)
            return np.all(fraction_above >= 0.95)

    def should_go_to_searching(self):
        # if the gd_snr has dropped below upper threshold consistently for at least 3 baselines out of 6
        buf = self.status_buffers.get("gd_snr")
        if buf is not None:
            below_threshold = buf < self.threshold_upper
            fraction_below = np.mean(below_threshold, axis=0)
            return np.sum(fraction_below >= 0.95) <= 3


def main():
    # --- State Machine Control Window ---
    class StateMachineControlWindow(QtWidgets.QWidget):
        def __init__(self, sm, parent=None):
            super().__init__(parent)
            self.setWindowTitle("State Machine Control")
            self.setFixedSize(300, 150)
            layout = QtWidgets.QVBoxLayout()
            self.setLayout(layout)

            # Enable/disable tickbox
            self.enable_checkbox = QtWidgets.QCheckBox("Enable State Machine")
            self.enable_checkbox.setChecked(True)
            layout.addWidget(self.enable_checkbox)

            # Lower threshold input
            lower_layout = QtWidgets.QHBoxLayout()
            lower_label = QtWidgets.QLabel("Lower GD SNR threshold:")
            self.lower_spin = QtWidgets.QDoubleSpinBox()
            self.lower_spin.setDecimals(2)
            self.lower_spin.setRange(0, 100)
            self.lower_spin.setValue(sm.threshold_lower)
            lower_layout.addWidget(lower_label)
            lower_layout.addWidget(self.lower_spin)
            layout.addLayout(lower_layout)

            # Upper threshold input
            upper_layout = QtWidgets.QHBoxLayout()
            upper_label = QtWidgets.QLabel("Upper GD SNR threshold:")
            self.upper_spin = QtWidgets.QDoubleSpinBox()
            self.upper_spin.setDecimals(2)
            self.upper_spin.setRange(0, 100)
            self.upper_spin.setValue(sm.threshold_upper)
            upper_layout.addWidget(upper_label)
            upper_layout.addWidget(self.upper_spin)
            layout.addLayout(upper_layout)

            # Connect signals
            self.enable_checkbox.stateChanged.connect(self.on_enable_changed)
            self.lower_spin.valueChanged.connect(self.on_lower_changed)
            self.upper_spin.valueChanged.connect(self.on_upper_changed)
            self.sm = sm

        def on_enable_changed(self, state):
            enabled = state == QtCore.Qt.Checked
            # Store on the state machine for access in update()
            setattr(self.sm, "active", enabled)

        def on_lower_changed(self, value):
            self.sm.threshold_lower = value

        def on_upper_changed(self, value):
            self.sm.threshold_upper = value

    # --- Create QApplication instance before any usage ---
    app = QtWidgets.QApplication([])

    # --- Parse arguments before constructing state machine ---
    parser = argparse.ArgumentParser(
        description="Real-time scrolling plots for Heimdallr."
    )
    parser.add_argument(
        "--update-time",
        type=int,
        default=50,
        help="Update interval in ms (default: 50)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples to hold (default: 100)",
    )
    parser.add_argument(
        "--linewidth",
        type=float,
        default=2.0,
        help="Line width for plot curves (default: 2.0)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        # options=["print", "heim"],
        choices=["print", "heim"],
        default="heim",
        help="Output method for offsets (default: heim)",
    )
    args = parser.parse_args()

    samples = args.samples
    update_time = args.update_time
    linewidth = args.linewidth

    n_max_samples = 5  # number of samples to keep track of as the best so far
    best_gd_SNR = [[(0, 0) for __ in range(n_max_samples)] for _ in range(N_BASELINES)]

    def reset_best_gd_SNR():
        nonlocal best_gd_SNR
        best_gd_SNR = [
            [(0, 0) for __ in range(n_max_samples)] for _ in range(N_BASELINES)
        ]
        print("best_gd_SNR has been reset.")

    # --- Setup HeimdallrStateMachine ---
    # Determine status keys and shapes from first status message
    status0 = Z.send("status")
    status_keys = list(status0.keys())
    status_shapes = {k: np.array(status0[k]).shape for k in status_keys}
    heimdallr_sm = HeimdallrStateMachine(
        status_keys=status_keys,
        status_shapes=status_shapes,
        buffer_length=samples,
        best_gd_SNR_ref=best_gd_SNR,
        reset_best_gd_SNR_func=reset_best_gd_SNR,
        server=Z,
    )

    # ...existing code...
    # Create the main window first so 'win' is defined
    win = pg.GraphicsLayoutWidget(show=True, title="Scrolling Plots")
    win.setWindowTitle("Heimdallr Real-Time Plots")
    # Move to top right (flush with edges)
    screen = QtWidgets.QApplication.primaryScreen()
    screen_geometry = screen.availableGeometry()
    legend_fixed_height = 450
    total_height = screen_geometry.height()
    win_height = total_height - legend_fixed_height - 50
    win_width = 900
    win.resize(win_width, win_height)
    win_x = screen_geometry.right() - win.width()
    win_y = screen_geometry.top()
    win.move(win_x, win_y)

    # Now create and show the state machine control window
    sm_control_win = StateMachineControlWindow(heimdallr_sm)
    sm_control_win.move(win.x() - sm_control_win.width() - 20, win.y())
    sm_control_win.show()

    # --- Global hotkey to close all windows ---
    class GlobalHotkeyFilter(QtCore.QObject):
        def eventFilter(self, obj, event):
            if event.type() == QtCore.QEvent.KeyPress:
                key = event.key()
                print(f"Key pressed: {key}")
                if key in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
                    for widget in QtWidgets.QApplication.topLevelWidgets():
                        widget.close()
                    QtCore.QCoreApplication.quit()
                    print("Closed all windows via global hotkey.")
                    return True
            return super().eventFilter(obj, event)

    hotkey_filter = GlobalHotkeyFilter()
    app.installEventFilter(hotkey_filter)
    parser = argparse.ArgumentParser(
        description="Real-time scrolling plots for Heimdallr."
    )
    parser.add_argument(
        "--update-time",
        type=int,
        default=50,
        help="Update interval in ms (default: 50)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples to hold (default: 100)",
    )
    parser.add_argument(
        "--linewidth",
        type=float,
        default=2.0,
        help="Line width for plot curves (default: 2.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        # options=["print", "heim"],
        choices=["print", "heim"],
        default="heim",
        help="Output method for offsets (default: heim)",
    )
    args = parser.parse_args()

    samples = args.samples
    update_time = args.update_time
    linewidth = args.linewidth

    # Define color sets for each column (move before legend_win)
    TELESCOPE_COLORS = [
        pg.mkPen(color, width=linewidth)
        for color in ["#601A4A", "#EE442F", "#63ACBE", "#F9F4EC"]
    ]
    BASELINE_COLORS = [
        pg.mkPen(color, width=linewidth)
        for color in [
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#F0E442",  # correct
            "#0072B2",
            "#D55E00",  # 34 for real though?
        ]
    ]
    baseline_names = [
        "12",
        "13",
        "14",
        "23",
        "24",
        "34",
    ]
    BASELINE_POSITIONS = np.array(
        [
            [-0.12, -4.425],
            [-1.025, 2.46],
            [-3.81, 2.425],
            [-1.145, -1.965],
            [-3.93, -2.0],
            [-2.785, -0.035],
        ]
    )  # shape: (N_BASELINES, 2), adjust as needed

    # baseline_names = [
    #     "24",
    #     "14",
    #     "34",
    #     "23",
    #     "13",
    #     "12",
    # ]
    # BASELINE_POSITIONS = np.array(
    #     [
    #         [-3.93, -2.0],
    #         [-3.81, 2.425],
    #         [-2.785, -0.035],
    #         [-1.145, -1.965],
    #         [-1.025, 2.46],
    #         [-0.12, -4.425],
    #     ]
    # )  # shape: (N_BASELINES, 2), adjust as needed
    M = np.array(
        [
            [-1, 1, 0, 0],
            [-1, 0, 1, 0],
            [-1, 0, 0, 1],
            [0, -1, 1, 0],
            [0, -1, 0, 1],
            [0, 0, -1, 1],
        ]
    )
    # inv_M = np.linalg.pinv(M)  # Unused variable removed

    # Time axis: from -window to 0, in seconds
    time_axis = np.linspace(-samples * update_time / 1000.0, 0, samples)

    status = Z.send("status")
    v2_K1 = np.zeros((samples, N_BASELINES))
    v2_K2 = np.zeros((samples, N_BASELINES))
    pd_tel = np.zeros((samples, N_TSCOPES))
    gd_tel = np.zeros((samples, N_TSCOPES))
    dm = np.zeros((samples, N_TSCOPES))
    offload = np.zeros((samples, N_TSCOPES))
    gd_snr = np.zeros((samples, N_BASELINES))
    pd_snr = np.zeros((samples, N_BASELINES))

    gd_threshold = 5.0

    class GD_SNR_vs_Offset:
        def __init__(self, beam_no):
            self.beam_no = beam_no

            self.offsets = []
            self.gd_snr_mean = []
            self.M2 = []
            self.gd_snr_std = []
            self.n_measurements = []

        def add_measurement(self, offset, snr):
            # floating point check on if offset is already in the list
            is_offset_in_list = any(
                np.isclose(offset, o, atol=0.2) for o in self.offsets
            )
            if not is_offset_in_list:
                self.offsets.append(offset)
                idx = self.offsets.index(offset)
                self.n_measurements.append(1)
                self.gd_snr_mean.append(snr)
                self.gd_snr_std.append(0.0)
                self.M2.append(0.0)
            else:
                idx = -1
                for i, o in enumerate(self.offsets):
                    if np.isclose(offset, o, atol=1e-3):
                        idx = i
                        break

                # Welford's online algorithm for mean and stddev
                # n = self.n_measurements[idx]  # Unused variable removed

                self.n_measurements[idx] += 1
                delta = snr - self.gd_snr_mean[idx]
                self.gd_snr_mean[idx] += delta / self.n_measurements[idx]
                self.M2[idx] += delta * (snr - self.gd_snr_mean[idx])
                if self.n_measurements[idx] > 1:
                    self.gd_snr_std[idx] = (
                        self.M2[idx] / (self.n_measurements[idx] - 1)
                    ) ** 0.5
                else:
                    self.gd_snr_std[idx] = 0.0

    gd_snr_vs_offsets = [
        GD_SNR_vs_Offset(i) for i in [1, 2, 4]
    ]  # one per telescope relative to 3

    # tracking_states: 4 vector of strings
    tracking_states = ["" for _ in range(N_TSCOPES)]

    win = pg.GraphicsLayoutWidget(show=True, title="Scrolling Plots")
    win.setWindowTitle("Heimdallr Real-Time Plots")
    # Move to top right (flush with edges)
    screen = QtWidgets.QApplication.primaryScreen()
    screen_geometry = screen.availableGeometry()
    # Calculate available height for win and scatter_win
    legend_fixed_height = 450
    total_height = screen_geometry.height()
    # Assign win to take the remaining height above the legend
    win_height = total_height - legend_fixed_height - 50
    win_width = 900
    win.resize(win_width, win_height)
    win_x = screen_geometry.right() - win.width()
    win_y = screen_geometry.top()
    win.move(win_x, win_y)

    # --- Color legend window ---
    legend_win = QtWidgets.QWidget()
    legend_win.setWindowTitle("Color Legend")
    legend_win.setFixedSize(350, legend_fixed_height)
    legend_layout = QtWidgets.QVBoxLayout()
    legend_win.setLayout(legend_layout)
    # Move below win, left-aligned with win
    legend_x = win_x
    legend_y = win_y + win.height()
    legend_win.move(legend_x, legend_y)
    # Set dark theme for legend_win
    legend_win.setStyleSheet(
        """
        QWidget {
            background-color: #222;
            color: #EEE;
        }
        QLabel {
            color: #EEE;
        }
    """
    )

    # --- Button to compute and print offsets from median OPD values ---
    def print_offsets_from_n_best(n):
        assert n >= 4, "n should be at least 4 to compute median"
        # For each baseline, take the median OPD from best_gd_SNR
        median_opds = []
        snrs = []
        for baseline_idx in range(N_BASELINES):
            # best_gd_SNR[baseline_idx] is a list of (gd_snr, opd) tuples
            opds = [opd for _, opd in best_gd_SNR[baseline_idx]]
            snrs = [snr for snr, _ in best_gd_SNR[baseline_idx]]
            median_opds.append(np.median(opds) if opds else 0.0)
            snrs.append(np.median(snrs) if snrs else 0.0)

        # find which are the n best snrs (indices)
        best_indices = np.argsort(snrs)[-n:]
        # Compute estimated OPDs for the best baselines
        est_opls = (
            np.linalg.pinv(M[best_indices, :]) @ np.array(median_opds)[best_indices]
        )

        if args.output == "print":
            # print with format x1, x2, x3, x4 to 3 decimal places
            print(
                "Estimated OPLs from best baselines: "
                + ", ".join(f"{opl:.3f}" for opl in est_opls)
            )
        elif args.output == "heim":
            # Send the estimated OPLs to the Heimdallr server
            msg = f"dls {','.join(f'{opl:.3f}' for opl in est_opls)}"
            Z.send(msg)

    # --- Three offset buttons for n=4,5,6 ---
    offset_buttons_layout = QtWidgets.QHBoxLayout()
    offset_button_4 = QtWidgets.QPushButton("Offsets n=4")
    offset_button_5 = QtWidgets.QPushButton("Offsets n=5")
    offset_button_6 = QtWidgets.QPushButton("Offsets n=6")
    offset_button_4.clicked.connect(lambda _: print_offsets_from_n_best(4))
    offset_button_5.clicked.connect(lambda _: print_offsets_from_n_best(5))
    offset_button_6.clicked.connect(lambda _: print_offsets_from_n_best(6))
    offset_buttons_layout.addWidget(offset_button_4)
    offset_buttons_layout.addWidget(offset_button_5)
    offset_buttons_layout.addWidget(offset_button_6)

    # --- Button to reset best_gd_SNR ---
    reset_button = QtWidgets.QPushButton("Reset best_gd_SNR")
    reset_button.clicked.connect(reset_best_gd_SNR)

    # Insert buttons above the Telescopes label
    legend_layout.addLayout(offset_buttons_layout)
    legend_layout.addWidget(reset_button)

    # Telescopes legend with tracking state swatch
    tel_label = QtWidgets.QLabel("<b>Telescopes</b>")
    legend_layout.addWidget(tel_label)
    telescope_names = ["T1", "T2", "T3", "T4"]
    tracking_state_swatches = []
    for i, name in enumerate(telescope_names):
        color = TELESCOPE_COLORS[i % N_TSCOPES].color()
        color_hex = color.name() if hasattr(color, "name") else color
        swatch = QtWidgets.QLabel()
        swatch.setFixedWidth(30)
        swatch.setFixedHeight(15)
        swatch.setStyleSheet(f"background-color: {color_hex}; border: 1px solid #333;")
        # Tracking state swatch
        state_swatch = QtWidgets.QLabel()
        state_swatch.setFixedWidth(70)
        state_swatch.setFixedHeight(15)
        state_swatch.setAlignment(QtCore.Qt.AlignCenter)
        state_swatch.setStyleSheet(
            "background-color: #888; color: black; border: 1px solid #333;"
        )
        tracking_state_swatches.append(state_swatch)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(swatch)
        row.addWidget(QtWidgets.QLabel(name))
        row.addWidget(state_swatch)
        row.addStretch()
        legend_layout.addLayout(row)

    def update_tracking_state_swatches():
        # Color: red for 'No fringes', dark green for 'Group 1', light green for 'Group 2', black text
        for i, state_label in enumerate(tracking_state_swatches):
            state = tracking_states[i] if i < len(tracking_states) else ""
            if state == "No fringes":
                bg = "#FF3333"  # red
            elif state == "Group 1":
                bg = "#228B22"  # dark green
            elif state == "Group 2":
                bg = "#90EE90"  # light green
            else:
                bg = "#888"
            state_label.setText(state)
            state_label.setStyleSheet(
                f"background-color: {bg}; color: black; border: 1px solid #333;"
            )

    # Timer to update tracking state swatches
    tracking_timer = QtCore.QTimer()
    tracking_timer.timeout.connect(update_tracking_state_swatches)
    tracking_timer.start(200)

    legend_layout.addSpacing(10)
    base_label = QtWidgets.QLabel("<b>Baselines</b>")
    legend_layout.addWidget(base_label)

    # --- New Figure for OPD vs V2_K1 and V2_K2 (all baselines on one plot) ---
    scatter_win = pg.GraphicsLayoutWidget(show=True, title="OPD vs GD SNR")
    scatter_win.setWindowTitle("Best OPD vs GD SNR (All Baselines)")
    # Calculate position and size so:
    # - left edge flush with right edge of legend
    # - bottom edge aligned with bottom of legend
    # - right edge aligned with right edge of win
    scatter_x = legend_x + legend_win.width()
    scatter_y = legend_y
    scatter_width = win_x + win.width() - scatter_x
    scatter_height = legend_win.height()
    # Make scatter_win and legend_win together span the full height
    # If legend is at the bottom, scatter_win's height = legend + win's height
    scatter_height = legend_fixed_height
    scatter_win.resize(scatter_width, scatter_height)
    scatter_win.move(scatter_x, scatter_y)
    scatter_plot = scatter_win.addPlot(
        row=0, col=0, title="All Baselines: OPD vs GD SNR"
    )
    scatter_plot.setLabel("left", "GD SNR")
    scatter_plot.setLabel("bottom", "OPD")
    scatter_plot.showGrid(x=True, y=True)
    # scatter_items_k1 = []
    # scatter_items_k2 = []
    scatter_items_gd = []
    for i in range(N_BASELINES):
        color = (
            BASELINE_COLORS[i].color()
            if hasattr(BASELINE_COLORS[i], "color")
            else BASELINE_COLORS[i]
        )
        scatter_gd = pg.ScatterPlotItem(
            pen=BASELINE_COLORS[i],
            brush=color,
            symbol="o",
            size=12,
            name=f"GD {baseline_names[i]}",
        )
        scatter_plot.addItem(scatter_gd)
        scatter_items_gd.append(scatter_gd)

        # scatter_k2 = pg.ScatterPlotItem(
        #     pen=BASELINE_COLORS[i],
        #     brush=color,
        #     symbol="o",
        #     size=12,
        #     name=f"K2 {baseline_names[i]}",
        # )
        # scatter_k1 = pg.ScatterPlotItem(
        #     pen=BASELINE_COLORS[i],
        #     brush=None,
        #     symbol="x",
        #     size=12,
        #     name=f"K1 {baseline_names[i]}",
        # )
        # scatter_plot.addItem(scatter_k2)
        # scatter_plot.addItem(scatter_k1)
        # scatter_items_k2.append(scatter_k2)
        # scatter_items_k1.append(scatter_k1)

    # --- Baseline positions and circle plot ---
    baseline_plot_widget = pg.PlotWidget()
    baseline_plot_widget.setBackground("#222")
    baseline_plot_widget.setFixedHeight(200)
    baseline_plot_widget.setFixedWidth(320)
    baseline_plot_widget.setMouseEnabled(x=False, y=False)
    baseline_plot_widget.hideAxis("bottom")
    baseline_plot_widget.hideAxis("left")
    baseline_plot_widget.setAspectLocked(True)
    scatter = pg.ScatterPlotItem(
        x=BASELINE_POSITIONS[:, 0],
        y=BASELINE_POSITIONS[:, 1],
        size=30,
        brush=[BASELINE_COLORS[i % N_BASELINES].color() for i in range(N_BASELINES)],
        pen=pg.mkPen("w", width=2),
    )
    baseline_plot_widget.addItem(scatter)
    # Add text labels inside circles
    for i, name in enumerate(baseline_names):
        text = pg.TextItem(name, color="k", anchor=(0.5, 0.5), border=None, fill=None)
        text.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        text.setPos(BASELINE_POSITIONS[i, 0], BASELINE_POSITIONS[i, 1])
        baseline_plot_widget.addItem(text)

    # and negative case for pspec
    scatter = pg.ScatterPlotItem(
        x=-BASELINE_POSITIONS[:, 0],
        y=-BASELINE_POSITIONS[:, 1],
        size=30,
        brush=[BASELINE_COLORS[i % N_BASELINES].color() for i in range(N_BASELINES)],
        pen=pg.mkPen("w", width=2),
    )
    baseline_plot_widget.addItem(scatter)
    # Add text labels inside circles
    for i, name in enumerate(baseline_names):
        text = pg.TextItem(name, color="k", anchor=(0.5, 0.5), border=None, fill=None)
        text.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        text.setPos(-BASELINE_POSITIONS[i, 0], -BASELINE_POSITIONS[i, 1])
        baseline_plot_widget.addItem(text)

    legend_layout.addWidget(baseline_plot_widget)

    legend_win.show()

    # --- Offset Tweaker Window ---
    class OffsetTweakerWindow(QtWidgets.QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Heimdallr Offset Tweaker")
            self.setFixedSize(350, 220)
            layout = QtWidgets.QVBoxLayout()
            self.setLayout(layout)

            self.sliders = []
            self.labels = []
            self.value_labels = []
            slider_names = ["offset 1", "offset 2", "offset 4"]
            for name in slider_names:
                row = QtWidgets.QHBoxLayout()
                label = QtWidgets.QLabel(name)
                slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
                slider.setMinimum(-50)
                slider.setMaximum(50)
                slider.setValue(0)
                slider.setTickInterval(5)
                slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
                value_label = QtWidgets.QLabel("0.0")
                slider.valueChanged.connect(
                    lambda val, value_label=value_label: value_label.setText(
                        f"{val/10:.1f}"
                    )
                )
                row.addWidget(label)
                row.addWidget(slider)
                row.addWidget(value_label)
                layout.addLayout(row)
                self.sliders.append(slider)
                self.labels.append(label)
                self.value_labels.append(value_label)

            self.apply_button = QtWidgets.QPushButton("Apply")
            self.apply_button.clicked.connect(self.apply_offsets)
            layout.addWidget(self.apply_button)

        def apply_offsets(self):
            # Get slider values as floats from -5.0 to 5.0
            values = [slider.value() / 10.0 for slider in self.sliders]
            msg = f"tweak_gd_offsets {values[0]:.2f},{values[1]:.2f},{values[2]:.2f}"
            Z.send(msg)
            QtWidgets.QMessageBox.information(self, "Offsets Applied", f"Sent: {msg}")
            # Reset sliders to 0 after apply
            for slider, value_label in zip(self.sliders, self.value_labels):
                slider.setValue(0)
                value_label.setText("0.0")

    # Create and show the offset tweaker window
    offset_tweaker_win = OffsetTweakerWindow()
    # Place it to the right of the main win
    offset_tweaker_win.move(win.x() + win.width() + 20, win.y())
    offset_tweaker_win.show()

    # --- New Figure: GD SNR vs Offset (with error bars) ---
    gd_snr_vs_offset_win = pg.GraphicsLayoutWidget(show=True, title="GD SNR vs Offset")
    gd_snr_vs_offset_win.setWindowTitle("GD SNR vs Offset (per telescope)")
    # Place to the right of offset tweaker window
    gd_snr_vs_offset_win.resize(400, 300)
    gd_snr_vs_offset_win.move(
        offset_tweaker_win.x() + offset_tweaker_win.width() + 20, offset_tweaker_win.y()
    )
    gd_snr_vs_offset_plot = gd_snr_vs_offset_win.addPlot(
        title="GD SNR vs Offset (T1, T2, T4 rel. T3)"
    )
    gd_snr_vs_offset_plot.setLabel("left", "GD SNR (mean ± std)")
    gd_snr_vs_offset_plot.setLabel("bottom", "arg(GD phasor) (rad)")
    gd_snr_vs_offset_plot.showGrid(x=True, y=True)
    # One error bar item per telescope (relative to T3)
    gd_snr_vs_offset_errorbars = []
    gd_snr_vs_offset_scatter = []
    offset_colors = [pg.mkPen("#601A4A"), pg.mkPen("#EE442F"), pg.mkPen("#F9F4EC")]
    for i in range(3):
        scatter = pg.ScatterPlotItem(
            pen=offset_colors[i],
            brush=offset_colors[i].color(),
            symbol="o",
            size=10,
            name=f"T{[1, 2, 4][i]} vs T3",
        )
        errorbar = pg.ErrorBarItem(pen=offset_colors[i])
        gd_snr_vs_offset_plot.addItem(scatter)
        gd_snr_vs_offset_plot.addItem(errorbar)
        gd_snr_vs_offset_scatter.append(scatter)
        gd_snr_vs_offset_errorbars.append(errorbar)

    # --- Button to reset gd_snr_vs_offsets ---
    def reset_gd_snr_vs_offsets():
        for obj in gd_snr_vs_offsets:
            obj.offsets.clear()
            obj.gd_snr_mean.clear()
            obj.gd_snr_std.clear()
            obj.M2.clear()
            obj.n_measurements.clear()
        # Clear the plot
        for i in range(3):
            gd_snr_vs_offset_scatter[i].setData([])
            gd_snr_vs_offset_errorbars[i].setData(x=[], y=[], top=[], bottom=[])

    reset_gd_snr_vs_offsets_button = QtWidgets.QPushButton("Reset All GD SNR vs Offset")
    reset_gd_snr_vs_offsets_button.clicked.connect(reset_gd_snr_vs_offsets)
    # Add button below the plot
    layout = QtWidgets.QVBoxLayout()
    layout.addWidget(gd_snr_vs_offset_win)
    layout.addWidget(reset_gd_snr_vs_offsets_button)
    # Create a container widget to hold both
    gd_snr_vs_offset_container = QtWidgets.QWidget()
    gd_snr_vs_offset_container.setLayout(layout)
    gd_snr_vs_offset_container.move(gd_snr_vs_offset_win.x(), gd_snr_vs_offset_win.y())
    gd_snr_vs_offset_container.resize(
        gd_snr_vs_offset_win.width(), gd_snr_vs_offset_win.height() + 40
    )
    gd_snr_vs_offset_win.setParent(gd_snr_vs_offset_container)
    gd_snr_vs_offset_container.show()
    # --- Left Column: Telescopes ---
    # Subheader
    telescopes_label = pg.LabelItem(justify="center", color="w")
    telescopes_label.setText("<span style='font-size:16pt'><b>Telescopes</b></span>")
    win.addItem(telescopes_label, row=0, col=0)

    # gd_tel
    p_gd_tel = win.addPlot(row=1, col=0, title="Group Delay (wavelengths)")
    p_gd_tel.setLabel("left", "GD (wavelengths)")
    p_gd_tel.setLabel("bottom", "Time (s)")
    p_gd_tel.showGrid(x=True, y=True)
    c_gd_tel = [
        p_gd_tel.plot(time_axis, np.zeros(samples), pen=TELESCOPE_COLORS[i % N_TSCOPES])
        for i in range(N_TSCOPES)
    ]

    # pd_tel
    p_pd_tel = win.addPlot(row=2, col=0, title="Phase Delay (wavelengths)")
    p_pd_tel.setLabel("left", "PD (wavelengths)")
    p_pd_tel.setLabel("bottom", "Time (s)")
    p_pd_tel.showGrid(x=True, y=True)
    c_pd_tel = [
        p_pd_tel.plot(time_axis, np.zeros(samples), pen=TELESCOPE_COLORS[i % N_TSCOPES])
        for i in range(N_TSCOPES)
    ]

    # offload
    p_offload = win.addPlot(row=3, col=0, title="Offloaded Piston (microns)")
    p_offload.setLabel("left", "Offloaded Piston (μm)")
    p_offload.setLabel("bottom", "Time (s)")
    p_offload.showGrid(x=True, y=True)
    c_offload = [
        p_offload.plot(
            time_axis, np.zeros(samples), pen=TELESCOPE_COLORS[i % N_TSCOPES]
        )
        for i in range(N_TSCOPES)
    ]

    # dm
    p_dm = win.addPlot(row=4, col=0, title="Mirror Piston (fractional stroke)")
    p_dm.setLabel("left", "Mirror Piston")
    p_dm.setLabel("bottom", "Time (s)")
    p_dm.showGrid(x=True, y=True)
    c_dm = [
        p_dm.plot(time_axis, np.zeros(samples), pen=TELESCOPE_COLORS[i % N_TSCOPES])
        for i in range(N_TSCOPES)
    ]

    # --- Right Column: Baselines ---
    # Subheader
    baselines_label = pg.LabelItem(justify="center", color="w")
    baselines_label.setText("<span style='font-size:16pt'><b>Baselines</b></span>")
    win.addItem(baselines_label, row=0, col=1)

    # v2_K1
    p_v2_K1 = win.addPlot(row=1, col=1, title="V² K1")
    p_v2_K1.setLabel("left", "V² K1")
    p_v2_K1.setLabel("bottom", "Time (s)")
    p_v2_K1.showGrid(x=True, y=True)
    c_v2_K1 = [
        p_v2_K1.plot(time_axis, np.zeros(samples), pen=BASELINE_COLORS[i % N_BASELINES])
        for i in range(N_BASELINES)
    ]

    # v2_K2
    p_v2_K2 = win.addPlot(row=2, col=1, title="V² K2")
    p_v2_K2.setLabel("left", "V² K2")
    p_v2_K2.setLabel("bottom", "Time (s)")
    p_v2_K2.showGrid(x=True, y=True)
    c_v2_K2 = [
        p_v2_K2.plot(time_axis, np.zeros(samples), pen=BASELINE_COLORS[i % N_BASELINES])
        for i in range(N_BASELINES)
    ]

    # gd_snr
    p_gd_snr = win.addPlot(row=3, col=1, title="Group Delay SNR")
    p_gd_snr.setLabel("left", "GD SNR")
    p_gd_snr.setLabel("bottom", "Time (s)")
    p_gd_snr.showGrid(x=True, y=True)
    c_gd_snr = [
        p_gd_snr.plot(
            time_axis, np.zeros(samples), pen=BASELINE_COLORS[i % N_BASELINES]
        )
        for i in range(N_BASELINES)
    ]
    # Add a horizontal line for gd_threshold
    gd_threshold_line = pg.InfiniteLine(
        pos=gd_threshold, angle=0, pen=pg.mkPen("r", style=QtCore.Qt.DashLine)
    )
    p_gd_snr.addItem(gd_threshold_line)

    # pd_snr
    p_pd_snr = win.addPlot(row=4, col=1, title="Phase Delay SNR")
    p_pd_snr.setLabel("left", "PD SNR")
    p_pd_snr.setLabel("bottom", "Time (s)")
    p_pd_snr.showGrid(x=True, y=True)
    c_pd_snr = [
        p_pd_snr.plot(
            time_axis, np.zeros(samples), pen=BASELINE_COLORS[i % N_BASELINES]
        )
        for i in range(N_BASELINES)
    ]

    # --- Store curves for update ---
    curves = [
        c_gd_tel,  # 0
        c_pd_tel,  # 1
        c_offload,  # 2
        c_dm,  # 3
        c_v2_K1,  # 4
        c_v2_K2,  # 5
        c_gd_snr,  # 6
        c_pd_snr,  # 7
    ]

    def status_given_gd_snr(gd_snr):
        # gd_var = 1.83**2 / ((gd_snr) ** 2)
        gd_var = 1 / ((gd_snr) ** 2)
        gd_var = np.where(gd_snr < gd_threshold, 1e6, gd_var)

        M = np.array(
            [
                [-1, 1, 0, 0],
                [-1, 0, 1, 0],
                [-1, 0, 0, 1],
                [0, -1, 1, 0],
                [0, -1, 0, 1],
                [0, 0, -1, 1],
            ]
        )
        M_dag = 1 / 4 * M.T

        # units wavelengths
        # 1.83**2/((gd_snr)**2)
        W = np.diag(1 / gd_var)

        Igd = M @ np.linalg.pinv(M.T @ W @ M) @ M.T @ W
        # print(Igd)
        # print(np.linalg.matrix_rank(Igd))
        # # eigs
        # print(np.linalg.eigvals(Igd))

        # evecs

        cov_gd = M_dag @ Igd @ W @ Igd.T @ M_dag.T
        # print(cov_gd)

        tracking_states = [""] * 4

        # count the number of zeros in each column
        zero_counts = np.sum(np.isclose(cov_gd, 0, atol=1e-3), axis=0)
        # print(zero_counts)

        # if the count is 4, then the state is no fringes
        matching_matrix = np.logical_not(np.isclose(cov_gd, 0, atol=1e-3))

        # print(matching_matrix)
        # if there are 4 zero counts, then the state is no fringes
        # otherwise, the state is Group X, where X=1 or 2, and the group is where
        # there is a match of the columns of the matrix
        group_0 = []
        group_1 = []
        if np.sum(zero_counts == 4) == 4:
            if np.median(gd_var) > 1e5:
                tracking_states = ["No fringes"] * 4
            else:
                tracking_states = ["Group 1"] * 4
        else:
            for col_idx in range(matching_matrix.shape[1]):
                if zero_counts[col_idx] >= 3:
                    tracking_states[col_idx] = "No fringes"
                else:
                    if np.array_equal(
                        matching_matrix[:, col_idx], matching_matrix[:, 0]
                    ):
                        group_0.append(col_idx + 1)
                        tracking_states[col_idx] = "Group 1"
                    else:
                        group_1.append(col_idx + 1)
                        tracking_states[col_idx] = "Group 2"

        # if there is only one "Group 2", change it to "no fringes"
        if len(group_1) == 1:
            col_idx = group_1[0] - 1
            tracking_states[col_idx] = "No fringes"

        return tracking_states

    def update():
        nonlocal status, v2_K1, v2_K2, pd_tel, gd_tel, dm, offload, gd_snr, pd_snr
        nonlocal tracking_states, gd_threshold, gd_snr_vs_offsets
        status = Z.send("status")
        # Update state machine buffers and transitions
        # Only update/poll state machine if enabled
        if getattr(heimdallr_sm, "active", True):
            heimdallr_sm.update_status_buffers(status)
            heimdallr_sm.poll_transitions()
        arrays = [
            (gd_tel, "gd_tel"),
            (pd_tel, "pd_tel"),
            (offload, "dl_offload"),
            (dm, "dm_piston"),
            (v2_K1, "v2_K1"),
            (v2_K2, "v2_K2"),
            (gd_snr, "gd_snr"),
            (pd_snr, "pd_snr"),
        ]
        for arr, key in arrays:
            arr[:] = np.roll(arr, -1, axis=0)
            arr[-1] = status[key]

        settings = Z.send("settings")
        gd_threshold = float(settings.get("gd_threshold", gd_threshold))
        # Update the horizontal line position
        gd_threshold_line.setValue(gd_threshold)

        for i in range(N_TSCOPES):
            curves[0][i].setData(time_axis, gd_tel[:, i])
            curves[1][i].setData(time_axis, pd_tel[:, i])
            curves[2][i].setData(time_axis, offload[:, i])
            curves[3][i].setData(time_axis, dm[:, i])
        for i in range(N_BASELINES):
            curves[4][i].setData(time_axis, v2_K1[:, i])
            curves[5][i].setData(time_axis, v2_K2[:, i])
            curves[6][i].setData(time_axis, gd_snr[:, i])
            curves[7][i].setData(time_axis, pd_snr[:, i])

        tracking_states = status_given_gd_snr(gd_snr[-1])

        # print(M.shape, offload[-1].shape)
        opds = M @ offload[-1]

        # Update best samples if V2_K1 or V2_K2 is among the best so far
        for baseline_idx in range(N_BASELINES):
            cur_gdSNR = gd_snr[-1, baseline_idx]
            heapq.heappushpop(
                heimdallr_sm.best_gd_SNR[baseline_idx],
                (cur_gdSNR, opds[baseline_idx]),
            )

        # Update scatter plots for best_v2_K1 and best_v2_K2
        for i in range(N_BASELINES):
            # Unpack (value, opd) pairs
            # k1_points = best_v2_K1[i]
            # k2_points = best_v2_K2[i]

            # y1, x1 = zip(*k1_points)
            # y2, x2 = zip(*k2_points)

            y1, x1 = zip(*best_gd_SNR[i])

            scatter_items_gd[i].setData(x=x1, y=y1)

            # scatter_items_k1[i].setData(x=x1, y=y1)
            # scatter_items_k2[i].setData(x=x2, y=y2)

        # update gd_snr_vs_offsets
        baselines_of_interest = [1, 3, 5]  # baselines involving telescope 1,2,4 with 3
        gd_re = np.array(status["gd_phasor_real"])
        gd_im = np.array(status["gd_phasor_imag"])
        gd_phasor = gd_re + 1j * gd_im

        gd_offsets = [np.angle(gd_phasor[i]) for i in baselines_of_interest]

        for i, baseline_idx in enumerate(baselines_of_interest):
            if gd_snr[-1, baseline_idx] > gd_threshold:
                gd_snr_vs_offsets[i].add_measurement(
                    gd_offsets[i], gd_snr[-1, baseline_idx]
                )

        # --- Update GD SNR vs Offset plot ---
        for i, gd_obj in enumerate(gd_snr_vs_offsets):
            offsets = np.array(gd_obj.offsets, dtype=float)
            means = np.array(gd_obj.gd_snr_mean, dtype=float)
            stds = np.array(gd_obj.gd_snr_std, dtype=float)
            if offsets.size > 0 and means.size > 0 and stds.size > 0:
                spots = [{"pos": (offsets[j], means[j])} for j in range(len(offsets))]
                gd_snr_vs_offset_scatter[i].setData(spots)
                # Error bars: x=offsets, y=means, top=stds, bottom=stds
                err_data = dict(x=offsets, y=means, top=stds, bottom=stds)
                gd_snr_vs_offset_errorbars[i].setData(**err_data)
            else:
                gd_snr_vs_offset_scatter[i].setData([])
                gd_snr_vs_offset_errorbars[i].setData(
                    x=np.array([]),
                    y=np.array([]),
                    top=np.array([]),
                    bottom=np.array([]),
                )

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(update_time)

    # Handle KeyboardInterrupt to close all windows
    try:
        app.exec_()
    except KeyboardInterrupt:
        # Close all top-level windows
        for widget in QtWidgets.QApplication.topLevelWidgets():
            widget.close()
        QtCore.QCoreApplication.quit()
        print("Closed all windows due to KeyboardInterrupt.")


if __name__ == "__main__":
    main()
