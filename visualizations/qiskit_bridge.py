from typing import Any, Dict, List

import numpy as np


def _try_import_qiskit():
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import DensityMatrix, Statevector
        return Statevector, DensityMatrix, QuantumCircuit
    except ImportError:
        return None


def is_statevector(obj: Any) -> bool:
    qiskit = _try_import_qiskit()
    if qiskit is None:
        return False
    statevector = qiskit[0]
    return isinstance(obj, statevector)


def is_density_matrix(obj: Any) -> bool:
    qiskit = _try_import_qiskit()
    if qiskit is None:
        return False
    density_matrix = qiskit[1]
    return isinstance(obj, density_matrix)


def is_quantum_circuit(obj: Any) -> bool:
    qiskit = _try_import_qiskit()
    if qiskit is None:
        return False
    quantum_circuit = qiskit[2]
    return isinstance(obj, quantum_circuit)


def statevector_to_list(sv) -> List[complex]:
    return list(sv.data)


def density_matrix_to_array(dm) -> np.ndarray:
    return np.array(dm.data, dtype=complex)


def circuit_to_dict(qc) -> Dict[str, Any]:
    num_qubits = qc.num_qubits
    gates = []
    name_map = {
        'h': 'H', 'x': 'X', 'y': 'Y', 'z': 'Z',
        's': 'S', 't': 'T',
        'rx': 'RX', 'ry': 'RY', 'rz': 'RZ',
        'cx': 'CNOT', 'cnot': 'CNOT', 'CNOT': 'CNOT',
        'cz': 'CZ', 'p': 'P', 'u': 'U',
        'measure': 'measure',
    }
    for instruction in qc.data:
        gate_info = instruction.operation
        gate_name = gate_info.name.lower()
        mapped_name = name_map.get(gate_name, gate_name.upper())
        gate_dict = {'type': mapped_name}
        if len(instruction.qubits) == 1:
            gate_dict['qubit'] = instruction.qubits[0]._index
        elif len(instruction.qubits) == 2:
            gate_dict['control'] = instruction.qubits[0]._index
            gate_dict['target'] = instruction.qubits[1]._index
        if hasattr(gate_info, 'params') and gate_info.params:
            if mapped_name in ['RX', 'RY', 'RZ']:
                gate_dict['theta'] = float(gate_info.params[0])
            elif mapped_name == 'P':
                gate_dict['phi'] = float(gate_info.params[0])
        gates.append(gate_dict)
    return {
        'qubits': num_qubits,
        'gates': gates,
        'name': getattr(qc, 'name', 'Quantum Circuit'),
    }


def extract_statevector_from_circuit(qc) -> List[complex]:
    try:
        from qiskit.quantum_info import Statevector
        sv = Statevector.from_instructions(qc.data)
        return list(sv.data)
    except Exception as e:
        raise RuntimeError(f"Failed to extract statevector from circuit: {e}")
