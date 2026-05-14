import matplotlib
import numpy as np

import json
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt


def parse_complex(val):
    if isinstance(val, (int, float)):
        return complex(val, 0)
    elif isinstance(val, str):
        return complex(val.replace(' ', ''))
    return complex(val)


def bloch_vector(state) -> np.ndarray:
    from .qiskit_bridge import is_statevector, statevector_to_list
    if is_statevector(state):
        sv_list = statevector_to_list(state)
    else:
        sv_list = state
    alpha = parse_complex(sv_list[0])
    beta = parse_complex(sv_list[1]) if len(sv_list) >= 2 else 0j
    theta = 2 * np.arccos(np.abs(alpha))
    phi = np.angle(beta) - np.angle(alpha) if abs(beta) > 1e-10 else 0
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])


def draw_bloch_sphere(ax, vector: np.ndarray, title: str = "", trajectory: Optional[List[np.ndarray]] = None) -> None:
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='cyan', alpha=0.3, linewidth=0, edgecolor='gray')
    ax.quiver(0, 0, 0, 1.5, 0, 0, color='red', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 1.5, 0, color='green', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 1.5, color='blue', arrow_length_ratio=0.1)
    if trajectory is not None and len(trajectory) > 1:
        traj = np.array(trajectory)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r-', linewidth=2, alpha=0.5)
    ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color='black', linewidth=3, arrow_length_ratio=0.2)
    ax.scatter([vector[0]], [vector[1]], [vector[2]], color='black', s=50)
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.view_init(elev=20, azim=30)


def plot_rabi_oscillation(omega: float = 1.0, t_max: float = 10.0, n_points: int = 100, output_path: Optional[str] = None, dpi: int = 150) -> plt.Figure:
    times = np.linspace(0, t_max, n_points)
    trajectory = []
    for t in times:
        state = [np.cos(omega * t / 2), 1j * np.sin(omega * t / 2)]
        trajectory.append(bloch_vector(state))
    n = len(trajectory)
    cols = min(6, n)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(5*cols, 5*rows))
    step = max(1, n // 12)
    for idx, i in enumerate(range(0, n, step)):
        if idx >= 12:
            break
        ax = fig.add_subplot(rows, cols, idx+1, projection='3d')
        draw_bloch_sphere(ax, trajectory[i], f"t={times[i]:.2f}", trajectory[:i+1])
    fig.suptitle(f"Rabi Oscillation (ω={omega})", fontsize=16)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi)
        plt.close()
    else:
        return fig


def plot_time_evolution(states: Union[List[Any], Any], title: str = "Time Evolution", output_path: Optional[str] = None, dpi: int = 150) -> plt.Figure:
    from .qiskit_bridge import is_statevector, statevector_to_list
    trajectory = []
    for state in states:
        if is_statevector(state):
            sv_list = statevector_to_list(state)
            trajectory.append(bloch_vector(sv_list))
        else:
            trajectory.append(bloch_vector(state))
    n = len(trajectory)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(5*cols, 5*rows))
    for i, vec in enumerate(trajectory):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        draw_bloch_sphere(ax, vec, f"t={i}", trajectory[:i+1])
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi)
        plt.close()
    else:
        return fig


def plot_dynamic_flow(input_file: str, output_path: Optional[str] = None, dpi: int = 150) -> plt.Figure:
    with open(input_file, 'r') as f:
        data = json.load(f)
    if 'trajectory' in data:
        states = [np.array(s) for s in data['trajectory']]
        return plot_time_evolution(states, data.get('name', 'Bloch Trajectory'), output_path, dpi)
    else:
        return plot_density_evolution(data, output_path, dpi)


def plot_density_evolution(data: Dict[str, Any], output_path: Optional[str] = None, dpi: int = 150) -> plt.Figure:
    qubits = data['qubits']
    dim = 2 ** qubits
    stages = data['stages']
    n = len(stages)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(6*cols, 5*rows))
    for i, stage in enumerate(stages):
        state_vector = stage['state_vector']
        psi = np.array(state_vector, dtype=complex).reshape(-1, 1)
        rho = psi @ psi.conj().T
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        xpos, ypos = np.meshgrid(range(dim), range(dim), indexing='ij')
        xpos, ypos = xpos.flatten(), ypos.flatten()
        zpos = np.zeros_like(xpos)
        dx = dy = 0.8 * np.ones_like(zpos)
        real_vals = np.real(rho).flatten()
        colors = ['red' if v >= 0 else 'blue' for v in real_vals]
        ax.bar3d(xpos, ypos, zpos, dx, dy, real_vals, color=colors, alpha=0.8)
        ax.set_title(stage.get('name', f'Stage {i+1}'))
        ax.set_xlabel('Row')
        ax.set_ylabel('Col')
        ax.set_xticks(range(dim))
        ax.set_yticks(range(dim))
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi)
        plt.close()
    else:
        return fig
