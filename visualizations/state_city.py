import json
import os

import matplotlib
import numpy as np

from typing import Any, List, Optional, Union

import matplotlib.pyplot as plt


def parse_amplitude(amp: Any) -> complex:
    if isinstance(amp, (int, float)):
        return complex(amp, 0)
    elif isinstance(amp, str):
        amp = amp.replace(' ', '')
        if amp.endswith('j'):
            return complex(amp)
        else:
            return complex(float(amp), 0)
    elif isinstance(amp, (list, tuple)) and len(amp) == 2:
        return complex(amp[0], amp[1])
    else:
        raise ValueError(f"Unsupported amplitude format: {amp}")


def state_to_density(state_vector: List[complex]) -> np.ndarray:
    if not state_vector:
        raise ValueError("State vector cannot be empty")
    psi = np.array(state_vector, dtype=complex).reshape(-1, 1)
    return psi @ psi.conj().T


def plot_state_city(state: Any, title: str = "State City", output_path: Optional[str] = None, dpi: int = 150) -> Optional[plt.Figure]:
    from .qiskit_bridge import is_density_matrix, is_statevector, statevector_to_list, density_matrix_to_array
    if is_statevector(state):
        state_vector = statevector_to_list(state)
        rho = state_to_density(state_vector)
    elif is_density_matrix(state):
        rho = density_matrix_to_array(state)
    else:
        state_array = np.array(state, dtype=complex)
        if len(state_array.shape) == 2:
            rho = state_array
        elif len(state_array.shape) == 1:
            rho = state_to_density(state_array)
        else:
            raise ValueError(f"Unexpected state shape: {state_array.shape}")
    dim = rho.shape[0]
    n_qubits = int(np.log2(dim))
    if 2 ** n_qubits != dim:
        raise ValueError(f"Density matrix dimension {dim} is not a power of 2")
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(title, fontsize=14, y=0.98)
    real_vals = np.real(rho)
    imag_vals = np.imag(rho)
    max_val = max(abs(real_vals).max(), abs(imag_vals).max(), 1e-10)
    xpos, ypos = np.meshgrid(range(dim), range(dim), indexing='ij')
    xpos, ypos = xpos.flatten(), ypos.flatten()
    zpos = np.zeros_like(xpos)
    dx = dy = 0.8 * np.ones_like(zpos)
    ax1 = fig.add_subplot(121, projection='3d')
    dz_real = real_vals.flatten()
    colors_real = ['red' if v >= 0 else 'blue' for v in dz_real]
    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz_real, color=colors_real, alpha=0.8)
    ax1.set_title('Real Part')
    ax1.set_xlabel('Row')
    ax1.set_ylabel('Col')
    ax1.set_zlabel('Amplitude')
    ax1.set_xticks(range(dim))
    ax1.set_yticks(range(dim))
    ax1.set_zlim(-max_val * 1.2, max_val * 1.2)
    ax2 = fig.add_subplot(122, projection='3d')
    dz_imag = imag_vals.flatten()
    colors_imag = ['green' if v >= 0 else 'orange' for v in dz_imag]
    ax2.bar3d(xpos, ypos, zpos, dx, dy, dz_imag, color=colors_imag, alpha=0.8)
    ax2.set_title('Imaginary Part')
    ax2.set_xlabel('Row')
    ax2.set_ylabel('Col')
    ax2.set_zlabel('Amplitude')
    ax2.set_xticks(range(dim))
    ax2.set_yticks(range(dim))
    ax2.set_zlim(-max_val * 1.2, max_val * 1.2)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        return None
    return fig
