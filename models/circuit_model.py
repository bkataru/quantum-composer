import json
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np # Needed for 'pi' evaluation if we wanted to get fancy

class CircuitModel:
    def __init__(self, num_qubits=3):
        self.num_qubits = num_qubits
        self.operations = []

    def add_gate(self, gate_type, qubit_index, time_index, target_index=None, params=None):
        self.remove_gate(qubit_index, time_index)
        self.operations.append({
            'gate': gate_type,
            'qubit': qubit_index,
            'target': target_index,
            'params': params, # NEW: Store rotation angle
            'index': time_index
        })
        self.operations.sort(key=lambda x: x['index'])

    def remove_gate(self, qubit_index, time_index):
        self.operations = [op for op in self.operations 
                           if not (op['qubit'] == qubit_index and op['index'] == time_index)]

    def get_operations(self):
        return self.operations

    def run_simulation(self):
        qc = QuantumCircuit(self.num_qubits)
        
        for op in self.operations:
            g = op['gate'].lower()
            q = op['qubit']
            t = op.get('target')
            p = op.get('params') # Get parameters
           
            # --- Rotation Gates ---
            # We explicitly cast to float to ensure Qiskit accepts it
            if g == 'rx': qc.rx(float(p), q)
            elif g == 'ry': qc.ry(float(p), q)
            elif g == 'rz': qc.rz(float(p), q)
            elif g == 'p':  qc.p(float(p), q) # Phase Gate
            
            # --- Standard Gates ---
            elif g == 'h': qc.h(q)
            elif g == 'x': qc.x(q)
            elif g == 'y': qc.y(q)
            elif g == 'z': qc.z(q)
            
            # --- Multi Qubit ---
            elif t is not None:
                if g == 'cx': qc.cx(q, t)
                elif g == 'swap': qc.swap(q, t)
                elif g == 'cz': qc.cz(q, t)

        qc.measure_all()
        
        simulator = AerSimulator()
        compiled_circuit = transpile(qc, simulator)
        result = simulator.run(compiled_circuit, shots=1024).result()
        return result.get_counts()

    # (Keep to_json and load_from_json as they were)
    def to_json(self):
        return json.dumps({"num_qubits": self.num_qubits, "operations": self.operations})

    def load_from_json(self, json_str):
        data = json.loads(json_str)
        self.num_qubits = data["num_qubits"]
        self.operations = data["operations"]
