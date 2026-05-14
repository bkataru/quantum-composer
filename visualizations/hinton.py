import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Any, List, Optional

from .qiskit_bridge import is_statevector, statevector_to_list


def state_to_density(state_vector: List[complex]) -> np.ndarray:
    psi = np.array(state_vector, dtype=complex)
    return np.outer(psi, psi.conj())


def plot_hinton(state: Any, title: str = "Hinton Diagram", output_path: Optional[str] = None, dpi: int = 150) -> Optional[plt.Figure]:
    if is_statevector(state):
        state_vector = statevector_to_list(state)
    else:
        state_vector = state
    state_array = np.array(state_vector, dtype=complex)
    if len(state_array.shape) == 1:
        n = len(state_array)
        if n == 0:
            raise ValueError("State vector cannot be empty")
        rho = state_to_density(state_array)
        n_qubits = int(np.log2(n))
        if 2 ** n_qubits != n:
            raise ValueError(f"State vector length {n} is not power of 2")
    else:
        rho = state_array
        n = rho.shape[0]
        n_qubits = int(np.log2(n))
    fig, (ax_real, ax_imag) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=12, y=0.98)
    _plot_hinton_subplot(ax_real, np.real(rho), f'Real Part (n={n_qubits} qubits)', cmap='RdBu_r', vmin=-1, vmax=1)
    _plot_hinton_subplot(ax_imag, np.imag(rho), f'Imaginary Part (n={n_qubits} qubits)', cmap='RdBu_r', vmin=-1, vmax=1)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        return None
    return fig


def _plot_hinton_subplot(ax: plt.Axes, matrix: np.ndarray, title: str, cmap: str = 'RdBu_r', vmin: float = -1, vmax: float = 1) -> None:
    n = matrix.shape[0]
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    ax.set_xticks(np.arange(n+1)-0.5, minor=True)
    ax.set_yticks(np.arange(n+1)-0.5, minor=True)
    ax.grid(True, which='minor', color='black', linewidth=0.5, alpha=0.3)
    labels = [format(i, f'0{n.bit_length()-1}b') for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Basis State |k⟩')
    ax.set_ylabel('Basis State ⟨k|')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
