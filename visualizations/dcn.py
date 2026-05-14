import json
import os
from typing import Any, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle as MplCircle

from .qiskit_bridge import is_statevector, statevector_to_list


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


def is_separable_along_qubit(state_vector: List[complex], n_qubits: int, qubit_idx: int) -> bool:
    groups = {}
    for i, amp in enumerate(state_vector):
        key = i & ~(1 << qubit_idx)
        q_val = (i >> qubit_idx) & 1
        if key not in groups:
            groups[key] = [0+0j, 0+0j]
        groups[key][q_val] = amp
    for key in groups:
        g0, g1 = groups[key][0], groups[key][1]
        if abs(g0) > 1e-10 and abs(g1) > 1e-10:
            ratio = g1 / g0
            for key2 in groups:
                h0, h1 = groups[key2][0], groups[key2][1]
                if abs(h0) > 1e-10 and abs(h1) > 1e-10:
                    if abs(h1/h0 - ratio) > 1e-10:
                        return False
                elif abs(h0) > 1e-10 or abs(h1) > 1e-10:
                    return False
    return True


def compute_separability(state_vector: List[complex], n_qubits: int) -> List[bool]:
    return [is_separable_along_qubit(state_vector, n_qubits, q) for q in range(n_qubits)]


def plot_dcn(state: Any, title: str = "DCN Visualization", output_path: Optional[str] = None, dpi: int = 150) -> Optional[plt.Figure]:
    if is_statevector(state):
        state_vector = statevector_to_list(state)
    else:
        state_vector = state
    n = len(state_vector)
    n_qubits = int(np.log2(n))
    if 2 ** n_qubits != n:
        raise ValueError(f"State vector length {n} is not a power of 2")
    separable = compute_separability(state_vector, n_qubits)
    fig, ax = plt.subplots(figsize=(max(8, n_qubits * 3), max(6, n_qubits * 2)))
    fig.suptitle(title, fontsize=14, y=0.98)
    ax.set_xlim(-1, n_qubits + 1)
    ax.set_ylim(-1, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    for q in range(n_qubits):
        x_pos = q + 0.5
        top_y = 2.5
        bot_y = -0.5
        ax.plot([x_pos, x_pos], [bot_y, top_y], 'gray', linewidth=1, alpha=0.3)
        ax.text(x_pos, -0.3, f'q{q}', ha='center', fontsize=10)
        if separable[q]:
            ax.text(x_pos, top_y + 0.1, 'S', ha='center', fontsize=8, color='green')
        else:
            ax.text(x_pos, top_y + 0.1, 'E', ha='center', fontsize=8, color='red')
    max_amp = max(abs(a) for a in state_vector) if state_vector else 1.0
    if max_amp == 0:
        max_amp = 1.0
    selected = [(i, a) for i, a in enumerate(state_vector) if abs(a) > 1e-6]
    x_offsets = np.linspace(0.15, 0.85, n_qubits + 1) if n_qubits > 0 else [0.5]
    for idx, (basis_idx, amp) in enumerate(selected):
        magnitude = abs(amp) / max_amp
        phase = np.angle(amp)
        circle_radius = 0.15 + 0.25 * magnitude
        row = idx // n_qubits
        col = idx % n_qubits
        x_pos = col + x_offsets[col] if col < len(x_offsets) else col + 0.5
        y_pos = 2.0 - row * 0.6
        circle = MplCircle((x_pos, y_pos), circle_radius, facecolor='lightblue', edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.add_patch(circle)
        phase_line_len = circle_radius * 0.8
        line_x = x_pos + phase_line_len * np.cos(phase)
        line_y = y_pos + phase_line_len * np.sin(phase)
        ax.plot([x_pos, line_x], [y_pos, line_y], 'black', linewidth=2)
        basis_label = format(basis_idx, f'0{n_qubits}b')
        ax.text(x_pos, y_pos - circle_radius - 0.1, f'|{basis_label}⟩', ha='center', fontsize=7)
        mag_pct = int(magnitude * 100)
        ax.text(x_pos, y_pos + circle_radius + 0.05, f'{mag_pct}%', ha='center', fontsize=7)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Basis State'),
        plt.Line2D([0], [0], marker='', color='black', linewidth=2, label='Phase Direction'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        return None
    return fig
