import matplotlib
import numpy as np
import re
from typing import Any, List, Optional, Union

import matplotlib.pyplot as plt


def parse_ket(s: str) -> np.ndarray:
    s = s.strip().lower()
    if s == '|0>':
        return np.array([0, 0, 1])
    elif s == '|1>':
        return np.array([0, 0, -1])
    elif s == '|+>':
        return np.array([1, 0, 0])
    elif s == '|->':
        return np.array([-1, 0, 0])
    elif s == '|+i>':
        return np.array([0, 1, 0])
    elif s == '|-i>':
        return np.array([0, -1, 0])
    else:
        raise ValueError(f"Unknown ket: {s}")


def parse_complex_pair(s: str) -> np.ndarray:
    s = s.strip().strip('()')
    parts = s.split(',')
    if len(parts) != 2:
        raise ValueError("Complex pair must have exactly two entries")

    def to_complex(t: str) -> complex:
        t = t.strip().replace(' ', '')
        if t == 'i':
            return 1j
        elif t == '-i':
            return -1j
        elif 'j' in t:
            return complex(t)
        else:
            return complex(float(t), 0)

    alpha = to_complex(parts[0])
    beta = to_complex(parts[1])
    norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
    if norm == 0:
        raise ValueError("State vector cannot be zero")
    alpha /= norm
    beta /= norm
    theta = 2 * np.arccos(np.abs(alpha))
    phi = np.angle(beta) - np.angle(alpha)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])


def parse_angles(s: str) -> np.ndarray:
    theta_match = re.search(r'theta\s*=\s*([0-9.-]+)(?:\s*deg)?', s, re.I)
    phi_match = re.search(r'phi\s*=\s*([0-9.-]+)(?:\s*deg)?', s, re.I)
    if not theta_match or not phi_match:
        raise ValueError("Angle specification must contain theta and phi")
    theta = float(theta_match.group(1))
    phi = float(phi_match.group(1))
    if 'deg' in s.lower():
        theta = np.radians(theta)
        phi = np.radians(phi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])


def parse_bloch_vector(s: str) -> np.ndarray:
    s = s.strip().strip('()')
    parts = s.split(',')
    if len(parts) != 3:
        raise ValueError("Bloch vector must have three coordinates")
    vec = np.array([float(p.strip()) for p in parts])
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec /= norm
    else:
        raise ValueError("Bloch vector too close to zero")
    return vec


def parse_stage(line: str) -> Optional[np.ndarray]:
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    for parser in [parse_ket, parse_angles, parse_bloch_vector, parse_complex_pair]:
        try:
            if parser == parse_ket:
                if line.startswith('|') and '>' in line:
                    return parser(line)
            elif parser == parse_angles:
                if 'theta' in line.lower() and 'phi' in line.lower():
                    return parser(line)
            elif parser == parse_bloch_vector:
                if line.count('(') == 1 and line.count(')') == 1 and line.count(',') == 2:
                    return parser(line)
            elif parser == parse_complex_pair:
                if '(' in line and ')' in line and line.count(',') == 1:
                    return parser(line)
        except (ValueError, IndexError):
            continue
    raise ValueError(f"Unable to parse line: {line}")


def state_to_bloch(state_vector) -> np.ndarray:
    from .qiskit_bridge import is_statevector, statevector_to_list
    if is_statevector(state_vector):
        sv_list = statevector_to_list(state_vector)
    else:
        sv_list = state_vector
    alpha = complex(sv_list[0])
    beta = complex(sv_list[1]) if len(sv_list) >= 2 else 0j
    theta = 2 * np.arccos(np.abs(alpha))
    phi = np.angle(beta) - np.angle(alpha) if abs(beta) > 1e-10 else 0
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])


def draw_bloch_sphere(ax, vector: np.ndarray, title: str = "") -> None:
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='cyan', alpha=0.3, linewidth=0, edgecolor='gray')
    ax.quiver(0, 0, 0, 1.5, 0, 0, color='red', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 1.5, 0, color='green', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 1.5, color='blue', arrow_length_ratio=0.1)
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


def plot_bloch_sphere(states: Union[str, List[Any], Any], output_path: Optional[str] = None, dpi: int = 150) -> plt.Figure:
    from .qiskit_bridge import is_statevector
    vectors = []
    titles = []
    stage_num = 0
    if is_statevector(states):
        vec = state_to_bloch(states)
        vectors.append(vec)
        titles.append("State")
    elif isinstance(states, str):
        with open(states, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            try:
                vec = parse_stage(line)
                if vec is not None:
                    stage_num += 1
                    vectors.append(vec)
                    titles.append(f"Stage {stage_num}")
            except Exception as e:
                print(f"Error in line {i+1}: {line.strip()}\n{e}")
    else:
        for item in states:
            if is_statevector(item):
                vec = state_to_bloch(item)
                stage_num += 1
                vectors.append(vec)
                titles.append(f"State {stage_num}")
            elif isinstance(item, str):
                try:
                    vec = parse_stage(item)
                    if vec is not None:
                        stage_num += 1
                        vectors.append(vec)
                        titles.append(f"Stage {stage_num}")
                except Exception as e:
                    print(f"Error parsing '{item}': {e}")
            else:
                vec = state_to_bloch(item)
                stage_num += 1
                vectors.append(vec)
                titles.append(f"State {stage_num}")
    if not vectors:
        raise ValueError("No valid stages found.")
    n = len(vectors)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(5*cols, 5*rows))
    for idx, (vec, title) in enumerate(zip(vectors, titles)):
        ax = fig.add_subplot(rows, cols, idx+1, projection='3d')
        draw_bloch_sphere(ax, vec, title)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi)
        plt.close()
    else:
        return fig
