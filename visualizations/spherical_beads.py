import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
try:
    from scipy.special import sph_harm_y, lpmv
    def sph_harm(m, j, phi, theta):
        return sph_harm_y(j, m, theta, phi)
except ImportError:
    try:
        from scipy.special import sph_harm
    except ImportError:
        sph_harm = None
    lpmv = None


def real_sph_harm(j: int, m: int, theta: float, phi: float) -> float:
    if m > 0:
        Y_pos = sph_harm(m, j, phi, theta)
        Y_neg = sph_harm(-m, j, phi, theta)
        return np.sqrt(2) * np.real(Y_pos + Y_neg) / np.sqrt(2)
    elif m < 0:
        m_abs = abs(m)
        Y_pos = sph_harm(m_abs, j, phi, theta)
        Y_neg = sph_harm(-m_abs, j, phi, theta)
        return np.sqrt(2) * np.imag(Y_pos - Y_neg) / np.sqrt(2)
    else:
        Y = sph_harm(0, j, phi, theta)
        return np.real(Y)


def real_sph_harm_grid(j: int, m: int, theta_grid: np.ndarray, phi_grid: np.ndarray) -> np.ndarray:
    if m > 0:
        Y_pos = sph_harm(m, j, phi_grid, theta_grid)
        Y_neg = sph_harm(-m, j, phi_grid, theta_grid)
        return np.sqrt(2) * np.real(Y_pos)
    elif m < 0:
        m_abs = abs(m)
        Y_pos = sph_harm(m_abs, j, phi_grid, theta_grid)
        Y_neg = sph_harm(-m_abs, j, phi_grid, theta_grid)
        return np.sqrt(2) * np.imag(Y_pos)
    else:
        Y = sph_harm(0, j, phi_grid, theta_grid)
        return np.real(Y)


def bead_function(theta: float, phi: float, coefficients: Dict[int, Dict[int, complex]], scaling_factors: Dict[int, float], bead_type: str = "Q") -> float:
    value = 0.0
    for j, m_coeffs in coefficients.items():
        if j not in scaling_factors:
            continue
        s_j = scaling_factors[j]
        for m, c_jm in m_coeffs.items():
            Y_jm = real_sph_harm(j, m, theta, phi)
            value += s_j * np.real(c_jm) * Y_jm
    return float(value)


def bead_function_grid(theta_grid: np.ndarray, phi_grid: np.ndarray, coefficients: Dict[int, Dict[int, complex]], scaling_factors: Dict[int, float]) -> np.ndarray:
    result = np.zeros_like(theta_grid, dtype=float)
    for j, m_coeffs in coefficients.items():
        if j not in scaling_factors:
            continue
        s_j = scaling_factors[j]
        for m, c_jm in m_coeffs.items():
            Y_jm = real_sph_harm_grid(j, m, theta_grid, phi_grid)
            result += s_j * np.real(c_jm) * Y_jm
    return result


def generate_sphere_mesh(n_theta: int = 30, n_phi: int = 30, radius: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    X = radius * np.sin(theta_grid) * np.cos(phi_grid)
    Y = radius * np.sin(theta_grid) * np.sin(phi_grid)
    Z = radius * np.cos(theta_grid)
    return X, Y, Z, theta_grid, phi_grid


def compute_bead_colors(theta_grid: np.ndarray, phi_grid: np.ndarray, coefficients: Dict[int, Dict[int, complex]], scaling_factors: Dict[int, float], color_scheme: str = "red-green", bead_type: str = "Q") -> np.ndarray:
    values = bead_function_grid(theta_grid, phi_grid, coefficients, scaling_factors)
    from .beads_colors import value_to_color_array
    colors = value_to_color_array(values, color_scheme, bead_type)
    return colors


def qbead_coefficients_from_bloch(rx: float, ry: float, rz: float, scaling_factors: Dict[int, float]) -> Dict[int, Dict[int, complex]]:
    eta_1 = np.sqrt(4 * np.pi / 3)
    if 1 not in scaling_factors:
        return {1: {0: complex(0, 0)}}
    s_1 = scaling_factors[1]
    if abs(s_1) > 1e-10:
        c_10 = complex(rz * eta_1 / s_1, 0)
    else:
        c_10 = complex(0, 0)
    if abs(s_1) > 1e-10:
        c_11 = complex(rx * eta_1 / s_1 / np.sqrt(2), 0)
        c_1m1 = complex(ry * eta_1 / s_1 / np.sqrt(2), 0)
    else:
        c_11 = complex(0, 0)
        c_1m1 = complex(0, 0)
    return {1: {0: c_10, 1: c_11, -1: c_1m1}}


def ebead_even_coefficients(c0: complex, c2: complex, scaling_factors: Dict[int, float]) -> Dict[int, Dict[int, complex]]:
    result = {}
    if 0 in scaling_factors:
        result[0] = {0: c0}
    if 2 in scaling_factors:
        result[2] = {0: c2}
    return result


def ebead_odd_coefficients(c1: complex, scaling_factors: Dict[int, float]) -> Dict[int, Dict[int, complex]]:
    result = {}
    if 1 in scaling_factors:
        result[1] = {0: c1}
    return result


def reduced_density_to_qbead_coeffs(rho_red: np.ndarray, scaling_factors: Dict[int, float], N: int) -> Dict[int, Dict[int, complex]]:
    rx = float(np.real(np.trace(rho_red @ np.array([[0, 1], [1, 0]], dtype=complex))))
    ry = float(np.real(np.trace(rho_red @ np.array([[0, -1j], [1j, 0]], dtype=complex))))
    rz = float(np.real(np.trace(rho_red @ np.array([[1, 0], [0, -1]], dtype=complex))))
    return qbead_coefficients_from_bloch(rx, ry, rz, scaling_factors)


def bead_surface_with_values(coefficients: Dict[int, Dict[int, complex]], scaling_factors: Dict[int, float], n_theta: int = 30, n_phi: int = 30, radius: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    X = radius * np.sin(theta_grid) * np.cos(phi_grid)
    Y = radius * np.sin(theta_grid) * np.sin(phi_grid)
    Z = radius * np.cos(theta_grid)
    values = bead_function_grid(theta_grid, phi_grid, coefficients, scaling_factors)
    return X, Y, Z, values
