# CircuitView and DropLabel API

## CircuitView
The `CircuitView` class represents the main visual interface for the quantum circuit.

### Signals
- `gate_dropped(str, int, int)`: Emitted when a gate is dropped.
- `gate_deleted(int, int)`: Emitted when a gate is deleted.
- `gate_moved(int, int, int, int)`: Emitted when a gate is moved.

### Methods
- `__init__(num_qubits: int = 3, num_steps: int = 10)`: Initializes the circuit view.
- `setup_grid()`: Sets up the grid layout.
- `paintEvent(event)`: Custom painting logic.
- `clear_grid()`: Clears all gates.
- `place_gate_visual(...)`: Places a gate visually.
- `place_connector_visual(...)`: Places a connector visually.
- `show_circuit_array()`: Displays the circuit array.

## DropLabel
The `DropLabel` class represents a draggable and droppable label for placing quantum gates.

### Signals
- `gate_placed(str, int, int)`: Emitted when a gate is placed.
- `gate_removed(int, int)`: Emitted when a gate is removed.
- `gate_repositioned(int, int, int, int)`: Emitted when a gate is repositioned.

### Methods
- `__init__(qubit_idx: int, time_idx: int)`: Initializes the label.
- `contextMenuEvent(event)`: Displays a context menu.
- `mouseMoveEvent(event)`: Initiates a drag operation.
- `dragEnterEvent(event)`: Handles drag enter.
- `dragMoveEvent(event)`: Updates shadow preview.
- `dropEvent(event)`: Handles drop event.
- `dragLeaveEvent(event)`: Clears shadow preview.
- `set_visual_gate(...)`: Sets the visual representation of a gate.
- `clear_visual()`: Clears the visual representation.