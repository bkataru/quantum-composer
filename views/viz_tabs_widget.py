from typing import Dict, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QLabel
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from visualizations.workers import VizWorker

ALL_VIZ_TYPES = [
    ("Bloch Multivector", "bloch_multivector"),
    ("DCN", "dcn"),
    ("State City", "state_city"),
    ("PauliVec", "paulivec"),
    ("Hinton", "hinton"),
    ("Q-Beads", "qbeads"),
    ("Bloch Sphere", "bloch_sphere"),
    ("Circuit", "circuit"),
    ("QAOA Landscape", "cost_landscape_qaoa"),
    ("VQE Landscape", "cost_landscape_vqe"),
    ("Dynamic Flow", "dynamic_flow"),
]


class VizCanvasTab(QWidget):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.label = label
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(6, 5), dpi=100, facecolor="#121212")
        self.canvas = FigureCanvas(self.figure)
        self.placeholder = QLabel(
            f"{label}\n\nRun a simulation, then click this tab\nto generate the visualization."
        )
        self.placeholder.setStyleSheet("color: #888; font-size: 13px;")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.stack = QVBoxLayout()
        self.stack.addWidget(self.placeholder)
        self.stack.addWidget(self.canvas)
        self.canvas.hide()
        self.layout.addLayout(self.stack)

        self._figure_stored: Optional[Figure] = None

    def set_figure(self, fig: Figure):
        self._figure_stored = fig
        self.placeholder.hide()
        old_canvas = self.canvas
        self.canvas = FigureCanvas(fig)
        self.figure = fig
        self.stack.removeWidget(old_canvas)
        self.stack.insertWidget(1, self.canvas)
        old_canvas.deleteLater()
        self.canvas.draw()

    def has_figure(self) -> bool:
        return self._figure_stored is not None

    def clear_figure(self):
        self._figure_stored = None
        self.placeholder.show()
        old_canvas = self.canvas
        self.figure = Figure(figsize=(6, 5), dpi=100, facecolor="#121212")
        self.canvas = FigureCanvas(self.figure)
        self.stack.removeWidget(old_canvas)
        self.stack.insertWidget(1, self.canvas)
        old_canvas.deleteLater()


class VizTabsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)

        self._tabs: Dict[str, VizCanvasTab] = {}
        self._workers: Dict[str, VizWorker] = {}
        self._statevector = None
        self._num_qubits = 3

        for label, viz_type in ALL_VIZ_TYPES:
            tab = VizCanvasTab(label)
            self.tab_widget.addTab(tab, label.split(" ")[0])
            self._tabs[viz_type] = tab

        self.tab_widget.currentChanged.connect(self._on_tab_changed)

    def set_statevector(self, statevector, num_qubits: int):
        self._statevector = statevector
        self._num_qubits = num_qubits
        for tab in self._tabs.values():
            tab.clear_figure()

    def _on_tab_changed(self, index: int):
        if index < 0 or index >= len(ALL_VIZ_TYPES):
            return
        viz_type = ALL_VIZ_TYPES[index][1]
        tab = self._tabs[viz_type]
        if tab.has_figure() or self._statevector is None:
            return
        self._launch_worker(viz_type)

    def _launch_worker(self, viz_type: str):
        if viz_type in self._workers and self._workers[viz_type].isRunning():
            return
        tab = self._tabs[viz_type]
        tab.placeholder.setText(f"Generating {ALL_VIZ_TYPES[[t[1] for t in ALL_VIZ_TYPES].index(viz_type)][0]}...")
        tab.placeholder.show()

        worker = VizWorker(viz_type, self._statevector, self._num_qubits)
        worker.finished.connect(self._on_worker_finished)
        worker.error.connect(self._on_worker_error)
        worker.start()
        self._workers[viz_type] = worker

    def _on_worker_finished(self, viz_type: str, fig):
        if viz_type in self._tabs:
            self._tabs[viz_type].set_figure(fig)
        if viz_type in self._workers:
            del self._workers[viz_type]

    def _on_worker_error(self, error_msg: str):
        print(f"VizWorker error: {error_msg}")
