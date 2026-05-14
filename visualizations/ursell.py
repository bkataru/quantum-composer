import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.linalg import sqrtm


def pauli_expectation(rho: np.ndarray, pauli_type: str, qubit_idx: int, N: int) -> complex:
    pauli_matrices = {
        'x': np.array([[0, 1], [1, 0]], dtype=complex),
        'y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'z': np.array([[1, 0], [0, -1]], dtype=complex),
    }
    if pauli_type not in pauli_matrices:
        raise ValueError(f"Unknown Pauli type: {pauli_type}")
    sigma = pauli_matrices[pauli_type]
    return np.trace(rho @ sigma)


def all_pauli_expectations(rho_red: np.ndarray) -> Dict[str, float]:
    result = {}
    for p in ['x', 'y', 'z']:
        result[p] = np.real(pauli_expectation(rho_red, p, 0, 1))
    return result


def bilinear_total_correlation(rho_pair: np.ndarray, alpha: str = 'z', beta: str = 'z') -> float:
    pauli = {'x': np.array([[0, 1], [1, 0]], dtype=complex), 'y': np.array([[0, -1j], [1j, 0]], dtype=complex), 'z': np.array([[1, 0], [0, -1]], dtype=complex)}
    sigma_alpha = pauli[alpha]
    sigma_beta = pauli[beta]
    op = np.kron(sigma_alpha, sigma_beta)
    return np.real(np.trace(rho_pair @ op))


def bilinear_compound_correlation(rho1: np.ndarray, rho2: np.ndarray, alpha: str = 'z', beta: str = 'z') -> float:
    exp1 = pauli_expectation(rho1, alpha, 0, 1)
    exp2 = pauli_expectation(rho2, beta, 0, 1)
    return np.real(exp1 * exp2)


def bilinear_connected_correlation(rho_pair: np.ndarray, rho1: np.ndarray, rho2: np.ndarray, alpha: str = 'z', beta: str = 'z') -> float:
    T = bilinear_total_correlation(rho_pair, alpha, beta)
    C = bilinear_compound_correlation(rho1, rho2, alpha, beta)
    return T - C


def bilinear_all_correlations(rho_pair: np.ndarray, rho1: np.ndarray, rho2: np.ndarray) -> Dict[str, Dict[str, float]]:
    paulis = ['x', 'y', 'z']
    result = {'total': {}, 'compound': {}, 'connected': {}}
    for a in paulis:
        for b in paulis:
            key = f'{a}{b}'
            T = bilinear_total_correlation(rho_pair, a, b)
            C = bilinear_compound_correlation(rho1, rho2, a, b)
            E = T - C
            result['total'][key] = T
            result['compound'][key] = C
            result['connected'][key] = E
    return result


def trilinear_total_correlation(rho_tri: np.ndarray, alpha: str = 'z', beta: str = 'z', gamma: str = 'z') -> float:
    pauli = {'x': np.array([[0, 1], [1, 0]], dtype=complex), 'y': np.array([[0, -1j], [1j, 0]], dtype=complex), 'z': np.array([[1, 0], [0, -1]], dtype=complex)}
    op = np.kron(np.kron(pauli[alpha], pauli[beta]), pauli[gamma])
    return np.real(np.trace(rho_tri @ op))


def trilinear_compound_correlation(rho1: np.ndarray, rho2: np.ndarray, rho3: np.ndarray, E12: float, E13: float, E23: float, alpha: str = 'z', beta: str = 'z', gamma: str = 'z') -> float:
    exp1 = pauli_expectation(rho1, alpha, 0, 1)
    exp2 = pauli_expectation(rho2, beta, 0, 1)
    exp3 = pauli_expectation(rho3, gamma, 0, 1)
    C_product = np.real(exp1 * exp2 * exp3)
    C_connected = np.real(exp1 * E23 + exp2 * E13 + exp3 * E12)
    return C_product + C_connected


def trilinear_connected_correlation(rho_tri: np.ndarray, rho1: np.ndarray, rho2: np.ndarray, rho3: np.ndarray, E12: float = 0.0, E13: float = 0.0, E23: float = 0.0, alpha: str = 'z', beta: str = 'z', gamma: str = 'z') -> float:
    T = trilinear_total_correlation(rho_tri, alpha, beta, gamma)
    C = trilinear_compound_correlation(rho1, rho2, rho3, E12, E13, E23, alpha, beta, gamma)
    return T - C


def modified_density_matrix_2q(rho_full: np.ndarray, n_qubits: int, q1: int, q2: int) -> np.ndarray:
    paulis = ['x', 'y', 'z']
    pauli_mat = {'x': np.array([[0, 1], [1, 0]], dtype=complex), 'y': np.array([[0, -1j], [1j, 0]], dtype=complex), 'z': np.array([[1, 0], [0, -1]], dtype=complex)}
    rho1 = _get_reduced_2x2(rho_full, n_qubits, q1)
    rho2 = _get_reduced_2x2(rho_full, n_qubits, q2)
    subtraction = np.zeros_like(rho_full, dtype=complex)
    for alpha in paulis:
        for beta in paulis:
            exp1 = np.real(np.trace(rho1 @ pauli_mat[alpha]))
            exp2 = np.real(np.trace(rho2 @ pauli_mat[beta]))
            op = _embed_2q_operator(pauli_mat[alpha], pauli_mat[beta], q1, q2, n_qubits)
            subtraction += exp1 * exp2 * op
    subtraction = subtraction / 4.0
    return rho_full - subtraction


def _get_reduced_2x2(rho_full: np.ndarray, n_qubits: int, qubit_idx: int) -> np.ndarray:
    from .qbeads import compute_reduced_density_matrix
    dim = 2 ** n_qubits
    rho_red = np.zeros((2, 2), dtype=complex)
    mask_keep = 1 << qubit_idx
    for a in range(dim):
        for b in range(dim):
            bit_a_keep = (a >> qubit_idx) & 1
            bit_b_keep = (b >> qubit_idx) & 1
            bits_a_other = a & ~mask_keep
            bits_b_other = b & ~mask_keep
            if bits_a_other == bits_b_other:
                rho_red[bit_a_keep, bit_b_keep] += rho_full[a, b]
    return rho_red


def _embed_2q_operator(op1: np.ndarray, op2: np.ndarray, q1: int, q2: int, n_qubits: int) -> np.ndarray:
    ops = []
    for i in range(n_qubits):
        if i == q1:
            ops.append(op1)
        elif i == q2:
            ops.append(op2)
        else:
            ops.append(np.eye(2, dtype=complex))
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def entanglement_norm(rho_full: np.ndarray, n_qubits: int) -> float:
    if n_qubits == 2:
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        rho_tilde = np.kron(sigma_y, sigma_y) @ rho_full.conj().T @ np.kron(sigma_y, sigma_y)
        product = rho_full @ rho_tilde
        eigenvalues = np.linalg.eigvals(product)
        eigenvalues = np.sqrt(np.abs(eigenvalues))
        eigenvalues = np.sort(eigenvalues)[::-1]
        concurrence = max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] + eigenvalues[3])
        return concurrence
    total_norm = 0.0
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            total_norm += 0.0
    return total_norm


def compute_all_ebeads(state_vector: List[complex], n_qubits: int) -> List[Dict]:
    from .qbeads import compute_reduced_density_matrix, compute_pair_density_matrix
    results = []
    if n_qubits >= 2:
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                rho1 = compute_reduced_density_matrix(state_vector, n_qubits, i)
                rho2 = compute_reduced_density_matrix(state_vector, n_qubits, j)
                rho_pair = compute_pair_density_matrix(state_vector, n_qubits, i, j)
                E_zz = bilinear_connected_correlation(rho_pair, rho1, rho2, 'z', 'z')
                strength = abs(E_zz)
                if strength > 0.01:
                    results.append({'qubits': (i, j), 'type': 'pairwise', 'connected': {'zz': E_zz}, 'strength': strength})
    return results
