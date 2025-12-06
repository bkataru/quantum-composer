class QiskitCodeGenerator:
    @staticmethod
    def generate(num_qubits, operations):
        code = [
            "from qiskit import QuantumCircuit",
            "import numpy as np",
            "",
            f"qc = QuantumCircuit({num_qubits})",
            ""
        ]
        
        sorted_ops = sorted(operations, key=lambda x: x['index'])
        processed_barriers = set()

        for op in sorted_ops:
            gate = op['gate'].lower()
            q = op['qubit']
            t = op.get('target')
            p = op.get('params')
           
            # Rotation Gates (With Parameters)
            if gate in ['rx', 'ry', 'rz', 'p'] and p is not None:
                code.append(f"qc.{gate}({p}, {q})")
            
            # Standard Single Qubit
            elif gate in ['h', 'x', 'y', 'z', 's', 't']:
                code.append(f"qc.{gate}({q})")
            
            # Multi Qubit
            elif t is not None:
                code.append(f"qc.{gate}({q}, {t})")
            
        code.append("")
        code.append("qc.measure_all()")
        
        return "\n".join(code)
