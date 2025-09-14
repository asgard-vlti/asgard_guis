"""
Various methods of drawing scrolling plots using pyqtgraph for speed and simplicity.
"""

import ZMQ_control_client as Z
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
import argparse

import heapq

N_TSCOPES = 4
N_BASELINES = 6


def main():
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
    inv_M = np.linalg.pinv(M)

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

    # tracking_states: 4 vector of strings
    tracking_states = ["" for _ in range(N_TSCOPES)]

    n_max_samples = 5  # number of samples to keep track of as the best so far
    # best_v2_K1 = [[(0, 0) for __ in range(n_max_samples)] for _ in range(N_BASELINES)]
    best_gd_SNR = [[(0, 0) for __ in range(n_max_samples)] for _ in range(N_BASELINES)]
    # best_v2_K2 = [[(0, 0) for __ in range(n_max_samples)] for _ in range(N_BASELINES)]

    app = QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="Scrolling Plots")
    win.resize(1200, 900)
    win.setWindowTitle("Heimdallr Real-Time Plots")

    # --- Color legend window ---
    legend_win = QtWidgets.QWidget()
    legend_win.setWindowTitle("Color Legend")
    legend_win.setFixedSize(350, 450)
    legend_layout = QtWidgets.QVBoxLayout()
    legend_win.setLayout(legend_layout)
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
    def print_offsets_from_median_opd():
        # For each baseline, take the median OPD from best_gd_SNR
        median_opds = []
        for baseline_idx in range(N_BASELINES):
            # best_gd_SNR[baseline_idx] is a list of (gd_snr, opd) tuples
            opds = [opd for _, opd in best_gd_SNR[baseline_idx]]
            if opds:
                median_opd = float(np.median(opds))
            else:
                median_opd = 0.0
            median_opds.append(median_opd)
        median_opds = np.array(median_opds)
        offsets = inv_M @ median_opds
        print("Median OPDs per baseline:", median_opds)
        print("Computed offsets (inv_M @ median_opds):", offsets)

    offset_button = QtWidgets.QPushButton("Print Offsets from Median OPD")
    offset_button.clicked.connect(print_offsets_from_median_opd)

    # --- Button to reset best_gd_SNR ---
    def reset_best_gd_SNR():
        nonlocal best_gd_SNR
        best_gd_SNR = [
            [(0, 0) for __ in range(n_max_samples)] for _ in range(N_BASELINES)
        ]
        print("best_gd_SNR has been reset.")

    reset_button = QtWidgets.QPushButton("Reset best_gd_SNR")
    reset_button.clicked.connect(reset_best_gd_SNR)

    # Insert buttons above the Telescopes label
    legend_layout.addWidget(offset_button)
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
    scatter_win = pg.GraphicsLayoutWidget(show=True, title="OPD vs V2_K1 and V2_K2")
    scatter_win.resize(1200, 800)

    scatter_win.setWindowTitle("Best OPD vs vis (All Baselines)")
    scatter_plot = scatter_win.addPlot(
        row=0, col=0, title="All Baselines: OPD vs V² K1/K2"
    )
    scatter_plot.setLabel("left", "GD Value")
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
        gd_var = np.where(gd_snr < 30, 1e6, gd_var)

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

        return tracking_states

    def update():
        nonlocal status, v2_K1, v2_K2, pd_tel, gd_tel, dm, offload, gd_snr, pd_snr, tracking_states
        status = Z.send("status")
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
            # cur_v2K1 = v2_K1[-1, baseline_idx]
            # cur_v2K2 = v2_K2[-1, baseline_idx]

            # heapq.heappushpop(
            #     best_v2_K1[baseline_idx],
            #     (cur_v2K1, opds[baseline_idx]),
            # )
            # heapq.heappushpop(
            #     best_v2_K2[baseline_idx],
            #     (cur_v2K2, opds[baseline_idx]),
            # )
            cur_gdSNR = gd_snr[-1, baseline_idx]
            heapq.heappushpop(
                best_gd_SNR[baseline_idx],
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

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(update_time)

    app.exec_()


if __name__ == "__main__":
    main()
