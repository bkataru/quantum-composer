import math
import numpy as np
from typing import Dict, Optional, Tuple


def zeta(N: int) -> float:
    return np.sqrt(2 ** N)


def eta(j: int) -> float:
    return np.sqrt(4 * np.pi / (2 * j + 1))


_CANONICAL_XI: Dict[Tuple[int, int], Dict[int, float]] = {
    (0, 0): {0: 1.0},
    (1, 0): {1: 1.0},
    (2, 0): {0: np.sqrt(1/3), 2: np.sqrt(2/3)},
    (3, 1): {1: np.sqrt(3/5), 3: np.sqrt(2/5)},
    (4, 1): {0: np.sqrt(7/35), 2: np.sqrt(20/35), 4: np.sqrt(8/35)},
    (5, 1): {1: np.sqrt(27/63), 3: np.sqrt(28/63), 5: np.sqrt(8/63)},
    (6, 1): {0: np.sqrt(33/231), 2: np.sqrt(110/231), 4: np.sqrt(72/231), 6: np.sqrt(16/231)},
}


def get_canonical_xi(linearity: int, tau: int, j: int) -> float:
    key = (linearity, tau)
    if key in _CANONICAL_XI:
        return _CANONICAL_XI[key].get(j, 0.0)
    return 0.0


def get_gub_xi(linearity: int, tau: int, j: int, N: int = 3) -> float:
    if linearity == 2:
        if j == 1:
            return 1.0 / np.sqrt(2)
    elif linearity == 3:
        if tau == 2:
            if j == 1:
                return 3.0 / (3.0 + np.sqrt(3))
            elif j == 2:
                return 1.0 / np.sqrt(2)
        elif tau == 3:
            if j == 1:
                return 1.0 / np.sqrt(2)
            elif j == 2:
                return 1.0 / np.sqrt(2)
        elif tau == 4:
            if j == 0:
                return 1.0 / np.sqrt(2)
    return 0.0


def scaling_factor(j: int, linearity: int, tau: int, N: int, method: str = "canonical") -> float:
    z = zeta(N)
    e = eta(j)
    if method == "canonical":
        xi = get_canonical_xi(linearity, tau, j)
    else:
        xi = get_gub_xi(linearity, tau, j, N)
    return z * xi * e


def get_all_scaling_factors(N: int, n_qubits: int, method: str = "canonical") -> Dict[str, Dict[int, float]]:
    result = {}
    for k in range(n_qubits):
        label = f'Q_{k}'
        result[label] = {1: scaling_factor(1, 1, 0, N, method)}
    if n_qubits >= 2:
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                label_even = f'E_{{{i},{j}}}_even'
                result[label_even] = {0: scaling_factor(0, 2, 0, N, method), 2: scaling_factor(2, 2, 0, N, method)}
                label_odd = f'E_{{{i},{j}}}_odd'
                result[label_odd] = {1: scaling_factor(1, 2, 0, N, "gub")}
    if n_qubits >= 3:
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                for k in range(j + 1, n_qubits):
                    label_tau1 = f'E_{{{i},{j},{k}tau1}}_odd'
                    result[label_tau1] = {1: scaling_factor(1, 3, 1, N, method), 3: scaling_factor(3, 3, 1, N, method)}
    return result


def qbead_scaling(N: int, qubit_idx: int) -> Dict[int, float]:
    return {1: scaling_factor(1, 1, 0, N, "canonical")}


def ebead_even_scaling(N: int, q1: int, q2: int) -> Dict[int, float]:
    return {0: scaling_factor(0, 2, 0, N, "canonical"), 2: scaling_factor(2, 2, 0, N, "canonical")}


def ebead_odd_scaling(N: int, q1: int, q2: int) -> Dict[int, float]:
    return {1: scaling_factor(1, 2, 0, N, "gub")}


def ebead_tau1_scaling(N: int, q1: int, q2: int, q3: int) -> Dict[int, float]:
    return {1: scaling_factor(1, 3, 1, N, "canonical"), 3: scaling_factor(3, 3, 1, N, "canonical")}


def print_scaling_table(N: int = 3, n_qubits: int = 3):
    print(f"Scaling Factors for N={N}, n_qubits={n_qubits}")
    print("=" * 60)
    factors = get_all_scaling_factors(N, n_qubits)
    for label, j_dict in sorted(factors.items()):
        print(f"\n{label}:")
        for j, s in sorted(j_dict.items()):
            xi_method = "canonical" if "tau" not in label or "tau1" in label else "gub"
            xi = s / (zeta(N) * eta(j)) if zeta(N) * eta(j) != 0 else 0
            print(f"  j={j}: s={s:.6f} (zeta={zeta(N):.4f}, xi={xi:.6f}, eta={eta(j):.4f})")
