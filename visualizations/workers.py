import traceback
from typing import Any, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PyQt6.QtCore import QThread, pyqtSignal


def _get_statevector_data(statevector) -> Any:
    from visualizations.qiskit_bridge import statevector_to_list
    return statevector_to_list(statevector)


class VizWorker(QThread):
    finished = pyqtSignal(str, object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, viz_type: str, statevector, num_qubits: int):
        super().__init__()
        self.viz_type = viz_type
        self.statevector = statevector
        self.num_qubits = num_qubits

    def run(self):
        try:
            self.progress.emit(f"Generating {self.viz_type}...")
            fig = self._generate()
            self.finished.emit(self.viz_type, fig)
            plt.close('all')
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}")
        finally:
            self.progress.emit("")

    def _generate(self):
        sv = self.statevector
        viz = self.viz_type

        if viz == "bloch_multivector":
            from visualizations.bloch_multivector import plot_bloch_multivector
            return plot_bloch_multivector(sv, title="Bloch Multivector", output_path=None)

        elif viz == "dcn":
            from visualizations.dcn import plot_dcn
            return plot_dcn(sv, title="DCN Visualization", output_path=None)

        elif viz == "state_city":
            from visualizations.state_city import plot_state_city
            return plot_state_city(sv, title="State City", output_path=None)

        elif viz == "paulivec":
            from visualizations.paulivec import plot_paulivec
            return plot_paulivec(sv, title="PauliVec", output_path=None)

        elif viz == "hinton":
            from visualizations.hinton import plot_hinton
            return plot_hinton(sv, title="Hinton Diagram", output_path=None)

        elif viz == "qbeads":
            from visualizations.qbeads import plot_qbeads
            return plot_qbeads(sv, title="BEADS Visualization", output_path=None, variant="A")

        elif viz == "bloch_sphere":
            from visualizations.bloch_sphere import plot_bloch_sphere
            return plot_bloch_sphere(sv, output_path=None)

        elif viz == "circuit":
            from visualizations.circuit_diagram import plot_circuit
            from visualizations.qiskit_bridge import is_quantum_circuit
            if is_quantum_circuit(sv):
                return plot_circuit(sv, output_path=None)
            sv_list = _get_statevector_data(sv)
            n_qubits = self.num_qubits
            circuit_dict = {
                'qubits': n_qubits,
                'gates': [],
                'name': 'Quantum Circuit',
            }
            return plot_circuit(circuit_dict, output_path=None)

        elif viz == "cost_landscape_qaoa":
            from visualizations.cost_landscape import plot_qaoa_landscape
            edges = [(i, (i + 1) % self.num_qubits) for i in range(self.num_qubits)]
            return plot_qaoa_landscape(edges, output_path=None)

        elif viz == "cost_landscape_vqe":
            from visualizations.cost_landscape import plot_vqe_landscape
            terms = [{"coeff": 1.0, "paulis": ["Z"]}]
            return plot_vqe_landscape(terms, output_path=None)

        elif viz == "dynamic_flow":
            from visualizations.dynamic_flow import plot_rabi_oscillation
            return plot_rabi_oscillation(omega=1.0, t_max=10.0, n_points=50, output_path=None)

        else:
            raise ValueError(f"Unknown visualization type: {viz}")
