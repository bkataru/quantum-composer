import numpy as np
from typing import Dict, List, Tuple, Any, Optional

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY_2 = np.array([[1, 0], [0, 1]], dtype=complex)


def _norm(N: int) -> float:
    return 1.0 / np.sqrt(2 ** N)


def lisa_1q_sigma_z(k: int, N: int) -> np.ndarray:
    norm = _norm(N)
    ops = []
    for i in range(N):
        if i == k:
            ops.append(SIGMA_Z)
        else:
            ops.append(IDENTITY_2)
    Z_k = ops[0]
    for op in ops[1:]:
        Z_k = np.kron(Z_k, op)
    return norm * Z_k


def lisa_1q_sigma_x(k: int, N: int) -> np.ndarray:
    norm = _norm(N)
    ops = []
    for i in range(N):
        if i == k:
            ops.append(SIGMA_X)
        else:
            ops.append(IDENTITY_2)
    X_k = ops[0]
    for op in ops[1:]:
        X_k = np.kron(X_k, op)
    return norm * X_k


def lisa_1q_sigma_y(k: int, N: int) -> np.ndarray:
    norm = _norm(N)
    ops = []
    for i in range(N):
        if i == k:
            ops.append(SIGMA_Y)
        else:
            ops.append(IDENTITY_2)
    Y_k = ops[0]
    for op in ops[1:]:
        Y_k = np.kron(Y_k, op)
    return norm * Y_k


def lisa_2q_even_0(q1: int, q2: int, N: int) -> np.ndarray:
    norm = _norm(N)
    result = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(2**N):
        for j in range(2**N):
            b1_i = (i >> q1) & 1
            b2_i = (i >> q2) & 1
            b1_j = (j >> q1) & 1
            b2_j = (j >> q2) & 1
            zz = (1 - 2*b1_i) * (1 - 2*b1_j) * (1 - 2*b2_i) * (1 - 2*b2_j)
            xx = 1.0 if (i == j or (i ^ (1<<q1) ^ (1<<q2)) == j) else 0.0
            yy_sign = 1.0
            if ((i >> q1) & 1) != ((j >> q1) & 1):
                yy_sign *= -1j
            if ((i >> q2) & 1) != ((j >> q2) & 1):
                yy_sign *= -1j
            yy = yy_sign if (i == j or (i ^ (1<<q1) ^ (1<<q2)) == j) else 0.0
            result[i, j] = (zz + xx + yy) / np.sqrt(3)
    return norm * result


def lisa_2q_even_2(q1: int, q2: int, N: int) -> np.ndarray:
    norm = _norm(N)
    result = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(2**N):
        for j in range(2**N):
            b1_i = (i >> q1) & 1
            b2_i = (i >> q2) & 1
            b1_j = (j >> q1) & 1
            b2_j = (j >> q2) & 1
            zz = (1 - 2*b1_i) * (1 - 2*b1_j) * (1 - 2*b2_i) * (1 - 2*b2_j)
            xx = 1.0 if (i == j or (i ^ (1<<q1) ^ (1<<q2)) == j) else 0.0
            yy_sign = 1.0
            if ((i >> q1) & 1) != ((j >> q1) & 1):
                yy_sign *= -1j
            if ((i >> q2) & 1) != ((j >> q2) & 1):
                yy_sign *= -1j
            yy = yy_sign if (i == j or (i ^ (1<<q1) ^ (1<<q2)) == j) else 0.0
            result[i, j] = (2 * zz - xx - yy) / np.sqrt(6)
    return norm * result


def lisa_2q_odd_1(q1: int, q2: int, N: int) -> np.ndarray:
    norm = _norm(N)
    result = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(2**N):
        for j in range(2**N):
            b1_i, b2_i = (i >> q1) & 1, (i >> q2) & 1
            b1_j, b2_j = (j >> q1) & 1, (j >> q2) & 1
            zx = (1 - 2*b1_i) * (1.0 if b2_i == b2_j else 0.0)
            xz = (1.0 if b1_i == b1_j else 0.0) * (1 - 2*b2_i)
            result[i, j] = (zx - xz) / (2 * np.sqrt(2))
    return norm * result


def lisa_3q_tau1_1(q1: int, q2: int, q3: int, N: int) -> np.ndarray:
    norm = _norm(N)
    result = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(2**N):
        for j in range(2**N):
            b1_i, b2_i, b3_i = (i >> q1) & 1, (i >> q2) & 1, (i >> q3) & 1
            b1_j, b2_j, b3_j = (j >> q1) & 1, (j >> q2) & 1, (j >> q3) & 1
            zzz = (1 - 2*b1_i) * (1 - 2*b1_j) * (1 - 2*b2_i) * (1 - 2*b2_j) * (1 - 2*b3_i) * (1 - 2*b3_j)
            result[i, j] = zzz / np.sqrt(5)
    return norm * result


def lisa_3q_tau1_3(q1: int, q2: int, q3: int, N: int) -> np.ndarray:
    norm = _norm(N)
    result = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(2**N):
        for j in range(2**N):
            b1_i, b2_i, b3_i = (i >> q1) & 1, (i >> q2) & 1, (i >> q3) & 1
            b1_j, b2_j, b3_j = (j >> q1) & 1, (j >> q2) & 1, (j >> q3) & 1
            zzz = (1 - 2*b1_i) * (1 - 2*b1_j) * (1 - 2*b2_i) * (1 - 2*b2_j) * (1 - 2*b3_i) * (1 - 2*b3_j)
            result[i, j] = zzz / np.sqrt(10)
    return norm * result


def decompose_density_1q(rho: np.ndarray, k: int, N: int) -> Dict[str, complex]:
    norm = _norm(N)
    T10 = lisa_1q_sigma_z(k, N)
    c1 = np.trace(rho @ T10.conj().T)
    result = {'j1': c1}
    c0 = np.trace(rho) / (2**N)
    result['j0'] = c0
    return result


def decompose_density_2q(rho_pair: np.ndarray, q1: int, q2: int, N: int) -> Dict[str, complex]:
    result = {}
    T00_even = lisa_2q_even_0(q1, q2, N)
    result['even_0'] = np.trace(rho_pair @ T00_even.conj().T)
    T20_even = lisa_2q_even_2(q1, q2, N)
    result['even_2'] = np.trace(rho_pair @ T20_even.conj().T)
    T10_odd = lisa_2q_odd_1(q1, q2, N)
    result['odd_1'] = np.trace(rho_pair @ T10_odd.conj().T)
    return result


def decompose_density_3q(rho_tri: np.ndarray, q1: int, q2: int, q3: int, N: int) -> Dict[str, complex]:
    result = {}
    T10_tau1 = lisa_3q_tau1_1(q1, q2, q3, N)
    result['tau1_1'] = np.trace(rho_tri @ T10_tau1.conj().T)
    T30_tau1 = lisa_3q_tau1_3(q1, q2, q3, N)
    result['tau1_3'] = np.trace(rho_tri @ T30_tau1.conj().T)
    return result


def get_qbead_coefficients(rho: np.ndarray, qubit_idx: int, N: int) -> Dict[int, complex]:
    rx = np.real(np.trace(rho @ SIGMA_X))
    ry = np.real(np.trace(rho @ SIGMA_Y))
    rz = np.real(np.trace(rho @ SIGMA_Z))
    norm = np.sqrt(2**N)
    eta_1 = np.sqrt(4 * np.pi / 3)
    coeffs = {1: complex(rz * norm / eta_1, 0)}
    return coeffs


def get_ebead_coefficients(rho_pair: np.ndarray, q1: int, q2: int, N: int) -> Dict[str, Dict[int, complex]]:
    decomp = decompose_density_2q(rho_pair, q1, q2, N)
    result = {'even': {0: decomp.get('even_0', 0), 2: decomp.get('even_2', 0)}, 'odd': {1: decomp.get('odd_1', 0)}}
    return result


def get_bead_spec_for_qubits(n_qubits: int) -> List[Dict[str, Any]]:
    beads = []
    for k in range(n_qubits):
        beads.append({'type': 'Q', 'qubits': (k,), 'label': f'Q_{k}', 'symmetry': None, 'tau': None, 'ranks': [1]})
    if n_qubits >= 2:
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                beads.append({'type': 'E', 'qubits': (i, j), 'label': f'E_{{{i},{j}}}_even', 'symmetry': 'even', 'tau': None, 'ranks': [0, 2]})
                beads.append({'type': 'E', 'qubits': (i, j), 'label': f'E_{{{i},{j}}}_odd', 'symmetry': 'odd', 'tau': None, 'ranks': [1]})
    if n_qubits >= 3:
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                for k in range(j + 1, n_qubits):
                    beads.append({'type': 'E', 'qubits': (i, j, k), 'label': f'E_{{{i},{j},{k}tau1}}_odd', 'symmetry': 'odd', 'tau': 'tau1', 'ranks': [1, 3]})
    return beads
