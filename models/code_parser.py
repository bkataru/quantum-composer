import sys
from io import StringIO
from qiskit import QuantumCircuit

class QiskitCodeParser:
    @staticmethod
    def parse_to_model(code_str, num_qubits):
        """
        Executes the code string to get a QuantumCircuit, 
        then maps instructions back to the visual model format,
        respecting multi-qubit spans.
        """
        # 1. Safe-ish Execution Environment
        local_scope = {}
        
        try:
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            exec(code_str, globals(), local_scope)
            
            sys.stdout = old_stdout 
            
            if 'qc' not in local_scope:
                return None 
            
            qc = local_scope['qc']
            if not isinstance(qc, QuantumCircuit):
                return None

        except Exception as e:
            sys.stdout = sys.__stdout__
            return None

        # 2. Reverse Engineer: Circuit -> Model Operations
        operations = []
        
        # Track the next available time slot for EACH qubit
        qubit_cursors = {i: 0 for i in range(num_qubits)}
        
        # Standard Qiskit gate name mapping
        gate_map = {
            'h': 'H', 'x': 'X', 'y': 'Y', 'z': 'Z',
            'rx': 'RX', 'ry': 'RY', 'rz': 'RZ', 'p': 'P',
            's': 'S', 'sdg': 'Sdg', 't': 'T', 'tdg': 'Tdg', 'sx': 'SX',
            'cx': 'CX', 'cy': 'CY', 'cz': 'CZ', 'ch': 'CH', 'swap': 'SWAP'
        }

        for instruction in qc.data:
            op_name = instruction.operation.name
            qubits = instruction.qubits
            params = instruction.operation.params
            
            if op_name not in gate_map:
                continue
                
            ui_gate = gate_map[op_name]
            
            # Identify Qubit Indices
            q_indices = [qc.find_bit(q).index for q in qubits]
            main_qubit = q_indices[0]
            target_qubit = q_indices[1] if len(q_indices) > 1 else None
            
            # --- CRITICAL FIX: Determine affected range ---
            # If it's a multi-qubit gate (like CX(0, 2)), it affects 0, 1, and 2.
            # If it's single qubit, it only affects [0].
            # If it's a barrier, it affects ALL qubits.
            
            affected_qubits = []
            
            if target_qubit is not None:
                start, end = min(main_qubit, target_qubit), max(main_qubit, target_qubit)
                affected_qubits = list(range(start, end + 1))
            else:
                affected_qubits = [main_qubit]
            
            # 1. Find the earliest column where ALL affected qubits are free
            # We look at the cursors for everyone involved.
            min_valid_time = 0
            for q in affected_qubits:
                if q < num_qubits: # Safety check
                    min_valid_time = max(min_valid_time, qubit_cursors[q])
            
            current_time = min_valid_time
            
            # 2. Add the operation
            # Note: For Barrier, we only add one entry per column in the model usually,
            # but the visualizer expects one per qubit. To keep parser simple, 
            # we just add the "Primary" operation here. The visualizer handles the rest?
            # Actually, your Controller handles Barrier by adding it to all.
            # But the PARSER should output exactly what the model expects.
            
            param_val = params[0] if params else None
            
            if ui_gate == "BARRIER":
                # For barrier, the model expects an entry on every qubit
                for q in range(num_qubits):
                    operations.append({
                        'gate': 'BARRIER',
                        'qubit': q,
                        'target': None,
                        'params': None,
                        'index': current_time
                    })
            else:
                operations.append({
                    'gate': ui_gate,
                    'qubit': main_qubit,
                    'target': target_qubit,
                    'params': param_val,
                    'index': current_time
                })
            
            # 3. Advance Cursors
            # Every qubit involved in this span must now move to the NEXT column
            for q in affected_qubits:
                if q < num_qubits:
                    qubit_cursors[q] = current_time + 1
                
        return operations
