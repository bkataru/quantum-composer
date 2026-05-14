import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Any, List, Optional, Tuple

from .qiskit_bridge import is_statevector, statevector_to_list


def pauli_expectations(state: Any, n_qubits: int) -> Tuple[List[str], np.ndarray]:
    state_array = np.array(state, dtype=complex)
    if len(state_array) == 2 ** n_qubits:
        rho = np.outer(state_array, state_array.conj())
    else:
        rho = state_array.reshape((2**n_qubits, 2**n_qubits))
    n_terms = 4 ** n_qubits
    labels = []
    values = np.zeros(n_terms)
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    paulis = [I, X, Y, Z]
    pauli_names = ['I', 'X', 'Y', 'Z']
    for i in range(n_terms):
        indices = []
        temp = i
        for _ in range(n_qubits):
            indices.append(temp % 4)
            temp //= 4
        label = ''
        matrix = None
        for qubit_idx, idx in enumerate(indices):
            label += pauli_names[idx]
            if matrix is None:
                matrix = paulis[idx]
            else:
                matrix = np.kron(paulis[idx], matrix)
        labels.append(label)
        values[i] = np.trace(rho @ matrix).real
    return labels, values


def plot_paulivec(state: Any, title: str = "PauliVec Visualization", output_path: Optional[str] = None, dpi: int = 150) -> Optional[plt.Figure]:
    if is_statevector(state):
        from .qiskit_bridge import statevector_to_list
        state = statevector_to_list(state)
    state_array = np.array(state, dtype=complex)
    if len(state_array.shape) == 1:
        n_qubits = int(np.log2(len(state_array)))
    else:
        n_qubits = int(np.log2(state_array.shape[0]))
    if 2 ** n_qubits != (len(state_array) if len(state_array.shape) == 1 else state_array.shape[0]):
        raise ValueError("State length must be a power of 2")
    labels, values = pauli_expectations(state, n_qubits)
    fig_width = max(8, len(labels) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    fig.suptitle(title, fontsize=12, y=0.95)
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, values, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('Pauli Term', fontsize=10)
    ax.set_ylabel('Expectation ⟨σ⟩', fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        return None
    return fig
