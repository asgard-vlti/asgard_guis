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
            "#F0E442",
            "#0072B2",
            "#D55E00",
        ]
    ]
    M = np.array(
        [
            [-1, 1, 0, 0],
            [-1, 0, 1, 0],
            [-1, 0, 0, 1],
            [0, -1, 1, 0],
            [0, -1, 0, 1],
            [0, 0, -1, 1],
        ]
    ).T

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

    n_max_samples = 20  # number of samples to keep track of as the best so far
    best_v2_K1 = [[] for _ in range(N_BASELINES)]
    best_v2_K2 = [[] for _ in range(N_BASELINES)]

    baseline_names = [
        "24",
        "14",
        "34",
        "23",
        "13",
        "12",
    ]

    app = QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="Scrolling Plots")
    win.resize(1200, 900)
    win.setWindowTitle("Heimdallr Real-Time Plots")

    # --- Color legend window ---
    legend_win = QtWidgets.QWidget()
    legend_win.setWindowTitle("Color Legend")
    legend_win.setFixedSize(350, 350)
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

    # Telescopes legend
    tel_label = QtWidgets.QLabel("<b>Telescopes</b>")
    legend_layout.addWidget(tel_label)
    telescope_names = ["T1", "T2", "T3", "T4"]
    for i, name in enumerate(telescope_names):
        color = TELESCOPE_COLORS[i % N_TSCOPES].color()
        color_hex = color.name() if hasattr(color, "name") else color
        swatch = QtWidgets.QLabel()
        swatch.setFixedWidth(30)
        swatch.setFixedHeight(15)
        swatch.setStyleSheet(f"background-color: {color_hex}; border: 1px solid #333;")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(swatch)
        row.addWidget(QtWidgets.QLabel(name))
        row.addStretch()
        legend_layout.addLayout(row)

    legend_layout.addSpacing(10)
    base_label = QtWidgets.QLabel("<b>Baselines</b>")
    legend_layout.addWidget(base_label)

    # --- New Figure for OPD vs V2_K1 and V2_K2 ---
    # Place this after baseline_names is defined
    # (Moved here to ensure baseline_names is defined)
    scatter_win = pg.GraphicsLayoutWidget(show=True, title="OPD vs V2_K1 and V2_K2")
    scatter_win.resize(1200, 800)
    scatter_win.setWindowTitle("Best OPD vs V2_K1 and V2_K2")
    scatter_plots = []
    scatter_items_k1 = []
    scatter_items_k2 = []
    for i in range(N_BASELINES):
        p = scatter_win.addPlot(
            row=i // 3, col=i % 3, title=f"Baseline {baseline_names[i]}"
        )
        p.setLabel("left", "V² Value")
        p.setLabel("bottom", "OPD")
        p.showGrid(x=True, y=True)
        # Scatter for K2 (filled), K1 (open)
        color = (
            BASELINE_COLORS[i].color()
            if hasattr(BASELINE_COLORS[i], "color")
            else BASELINE_COLORS[i]
        )
        scatter_k2 = pg.ScatterPlotItem(
            pen=BASELINE_COLORS[i],
            brush=color,
            symbol="o",
            size=12,
            name="K2",
        )
        scatter_k1 = pg.ScatterPlotItem(
            pen=BASELINE_COLORS[i],
            brush=color,
            symbol="x",
            size=12,
            name="K1",
        )
        p.addItem(scatter_k2)
        p.addItem(scatter_k1)
        scatter_plots.append(p)
        scatter_items_k2.append(scatter_k2)
        scatter_items_k1.append(scatter_k1)

    # --- Baseline positions and circle plot ---
    BASELINE_POSITIONS = np.array(
        [
            [-3.93, -2.0],
            [-3.81, 2.425],
            [-2.785, -0.035],
            [-1.145, -1.965],
            [-1.025, 2.46],
            [-0.12, -4.425],
        ]
    )  # shape: (N_BASELINES, 2), adjust as needed

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

    # plots = []  # Unused variable removed
    curves = []

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

    def update():
        nonlocal status, v2_K1, v2_K2, pd_tel, gd_tel, dm, offload, gd_snr, pd_snr
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

        print(M.shape, offload[-1].shape)
        opds = M @ offload[-1]

        # Update best samples if V2_K1 or V2_K2 is among the best so far

        for baseline_idx in range(N_BASELINES):
            if len(best_v2_K1[baseline_idx]) < n_max_samples:
                # we are still accumulating the first samples
                heapq.heappush(
                    best_v2_K1[baseline_idx],
                    (opds[baseline_idx], v2_K1[-1, baseline_idx]),
                )
                heapq.heappush(
                    best_v2_K2[baseline_idx],
                    (opds[baseline_idx], v2_K2[-1, baseline_idx]),
                )
            else:
                # add the using heappushpop
                heapq.heappushpop(
                    best_v2_K1[baseline_idx],
                    (opds[baseline_idx], v2_K1[-1, baseline_idx]),
                )
                heapq.heappushpop(
                    best_v2_K2[baseline_idx],
                    (opds[baseline_idx], v2_K2[-1, baseline_idx]),
                )

        # Update scatter plots for best_v2_K1 and best_v2_K2
        for i in range(N_BASELINES):
            # Unpack (opd, value) pairs
            k1_points = best_v2_K1[i]
            k2_points = best_v2_K2[i]
            if k1_points:
                x1, y1 = zip(*k1_points)
            else:
                x1, y1 = [], []
            if k2_points:
                x2, y2 = zip(*k2_points)
            else:
                x2, y2 = [], []
            scatter_items_k1[i].setData(x=x1, y=y1)
            scatter_items_k2[i].setData(x=x2, y=y2)

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(update_time)

    app.exec_()


if __name__ == "__main__":
    main()
