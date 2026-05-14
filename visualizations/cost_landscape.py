import os

import matplotlib
import numpy as np

import json
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

VALID_PAULIS = {"I", "X", "Y", "Z"}


def get_examples_dir() -> str:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(module_dir), "examples")


def validate_qaoa_input(data: Dict[str, Any]) -> List[tuple[int, int]]:
    if "edges" not in data:
        raise ValueError("Missing required key 'edges' in input data.")
    edges_raw = data["edges"]
    if not isinstance(edges_raw, list):
        raise ValueError(f"'edges' must be a list, got {type(edges_raw).__name__}.")
    edges = []
    for i, edge in enumerate(edges_raw):
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError(f"Edge {i} must be a pair [u, v], got {edge}.")
        u, v = edge
        if not isinstance(u, int) or not isinstance(v, int):
            raise ValueError(f"Edge {i} has non-integer vertices: [{u}, {v}].")
        if u < 0 or v < 0:
            raise ValueError(f"Edge {i} has negative vertex: [{u}, {v}].")
        edges.append((u, v))
    return edges


def validate_vqe_input(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "terms" not in data:
        raise ValueError("Missing required key 'terms' in input data.")
    terms_raw = data["terms"]
    if not isinstance(terms_raw, list):
        raise ValueError(f"'terms' must be a list, got {type(terms_raw).__name__}.")
    terms = []
    for i, term in enumerate(terms_raw):
        if not isinstance(term, dict):
            raise ValueError(f"Term {i} must be an object with 'coeff' and 'paulis'.")
        coeff = term.get("coeff")
        if coeff is None:
            raise ValueError(f"Term {i} missing required key 'coeff'.")
        if not isinstance(coeff, (int, float)):
            raise ValueError(f"Term {i} 'coeff' must be a number.")
        paulis = term.get("paulis", [])
        if not isinstance(paulis, list):
            raise ValueError(f"Term {i} 'paulis' must be a list.")
        for j, pauli in enumerate(paulis):
            if not isinstance(pauli, str):
                raise ValueError(f"Term {i}, Pauli {j} must be a string.")
            if pauli.upper() not in VALID_PAULIS:
                raise ValueError(f"Term {i}, Pauli '{pauli}' is invalid.")
        terms.append({"coeff": float(coeff), "paulis": [p.upper() for p in paulis]})
    return terms


def qaoa_cost(gamma: np.ndarray, beta: np.ndarray, edges: List[tuple[int, int]]) -> np.ndarray:
    g = np.asarray(gamma, dtype=float)
    b = np.asarray(beta, dtype=float)
    n_edges = len(edges)
    if n_edges == 0:
        return np.zeros_like(g, dtype=float)
    cost = np.zeros_like(g, dtype=float)
    for u, v in edges:
        cost += 0.5 - 0.5 * (np.cos(2*g) * (np.cos(2*b)**2 - np.sin(2*b)**2) + np.sin(2*g) * 2*np.sin(2*b))
    return cost / n_edges


def vqe_energy(theta: np.ndarray, terms: List[Dict[str, Any]]) -> np.ndarray:
    t = np.asarray(theta, dtype=float)
    n_terms = len(terms)
    if n_terms == 0:
        return np.zeros_like(t, dtype=float)
    energy = np.zeros_like(t, dtype=float)
    for term in terms:
        coeff = term.get("coeff", 1.0)
        paulis = term.get("paulis", [])
        product = np.ones_like(t)
        for pauli in paulis:
            op = pauli.upper() if isinstance(pauli, str) else "I"
            if op == "Z":
                product *= np.cos(t)
            elif op == "X":
                product *= np.sin(t)
            elif op == "Y":
                product *= np.ones_like(t)
            elif op == "I":
                product *= np.ones_like(t)
        energy += coeff * product
    return energy


def plot_qaoa_landscape(edges: List[tuple[int, int]], gamma_range: Tuple[float, float] = (0, np.pi), beta_range: Tuple[float, float] = (0, np.pi), resolution: int = 50, output_path: Optional[str] = None, dpi: int = 150) -> plt.Figure:
    gammas = np.linspace(float(gamma_range[0]), float(gamma_range[1]), resolution)
    betas = np.linspace(float(beta_range[0]), float(beta_range[1]), resolution)
    G, B = np.meshgrid(gammas, betas)
    Z = qaoa_cost(G, B, edges)
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(G, B, Z, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Cost (Cut Value)')
    plt.xlabel('γ (problem parameter)')
    plt.ylabel('β (mixer parameter)')
    n_qubits = max(max(e) for e in edges) + 1 if edges else 0
    plt.title(f'QAOA Cost Landscape: MaxCut ({n_qubits} qubits, {len(edges)} edges)')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi)
        plt.close()
    else:
        return plt.gcf()


def plot_vqe_landscape(terms: List[Dict[str, Any]], theta_range: Tuple[float, float] = (0, 2*np.pi), resolution: int = 100, output_path: Optional[str] = None, dpi: int = 150) -> plt.Figure:
    thetas = np.linspace(float(theta_range[0]), float(theta_range[1]), resolution)
    energies = vqe_energy(thetas, terms)
    plt.figure(figsize=(10, 6))
    plt.plot(thetas, energies, 'b-', linewidth=2)
    plt.xlabel('θ (variational parameter)')
    plt.ylabel('Energy')
    plt.title(f'VQE Energy Landscape: {len(terms)} terms')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi)
        plt.close()
    else:
        return plt.gcf()
