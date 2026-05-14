import matplotlib
import numpy as np
from typing import Any, List, Optional

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .qiskit_bridge import is_statevector


def partial_trace(rho: np.ndarray, qubit_idx: int, n_qubits: int) -> np.ndarray:
    dim = 2 ** n_qubits
    rho_reduced = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            total = 0j
            for state in range(dim):
                if ((state >> qubit_idx) & 1) == i:
                    for state2 in range(dim):
                        if ((state2 >> qubit_idx) & 1) == j:
                            if (state & ~(1 << qubit_idx)) == (state2 & ~(1 << qubit_idx)):
                                total += rho[state, state2]
            rho_reduced[i, j] = total
    return rho_reduced


def bloch_vector_from_rho(rho_i: np.ndarray) -> np.ndarray:
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    bloch = np.array([np.trace(X @ rho_i).real, np.trace(Y @ rho_i).real, np.trace(Z @ rho_i).real])
    return bloch


def purity(rho_i: np.ndarray) -> float:
    return float(np.trace(rho_i @ rho_i).real)


def draw_bloch_sphere_3d(ax: Axes3D, bloch_vector: np.ndarray, opacity: float = 1.0, title: str = "") -> None:
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.3, linewidth=0.5)
    ax.plot([-1.2, 1.2], [0, 0], [0, 0], 'k-', alpha=0.5, linewidth=1)
    ax.plot([0, 0], [-1.2, 1.2], [0, 0], 'k-', alpha=0.5, linewidth=1)
    ax.plot([0, 0], [0, 0], [-1.2, 1.2], 'k-', alpha=0.5, linewidth=1)
    if bloch_vector is not None:
        x, y, z = bloch_vector
        ax.scatter([x], [y], [z], color='red', s=100, alpha=opacity)
        ax.plot([0, x], [0, y], [0, z], 'r-', alpha=0.8, linewidth=2)
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_box_aspect([1, 1, 1])


def plot_bloch_multivector(state: Any, title: str = "Bloch Multivector", output_path: Optional[str] = None, dpi: int = 150) -> Optional[plt.Figure]:
    if is_statevector(state):
        from .qiskit_bridge import statevector_to_list
        state_vector = statevector_to_list(state)
    else:
        state_vector = state
    n = len(state_vector)
    n_qubits = int(np.log2(n))
    if 2 ** n_qubits != n:
        raise ValueError(f"State vector length {n} is not a power of 2")
    state_array = np.array(state_vector, dtype=complex)
    rho = np.outer(state_array, state_array.conj())
    n_spheres = n_qubits
    n_cols = min(n_spheres, 4)
    n_rows = (n_spheres + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(n_cols * 3, n_rows * 3))
    fig.suptitle(title, fontsize=12, y=0.98)
    for i in range(n_spheres):
        qubit_idx = i
        rho_i = partial_trace(rho, qubit_idx, n_qubits)
        bloch_vec = bloch_vector_from_rho(rho_i)
        pur = purity(rho_i)
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
        draw_bloch_sphere_3d(ax, bloch_vec, opacity=pur, title=f'Qubit {n_qubits - 1 - i}')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi)
        plt.close()
        return None
    return fig
