from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QScrollArea
from PyQt6.QtCore import Qt, QMimeData
from PyQt6.QtGui import QDrag

class DraggableButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setObjectName("GateButton")
        self.gate_type = text

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.MouseButton.LeftButton:
            drag = QDrag(self)
            mime = QMimeData()
            mime.setText(self.gate_type)
            drag.setMimeData(mime)
            drag.exec(Qt.DropAction.CopyAction)

class PaletteView(QWidget):
    def __init__(self):
        super().__init__()
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        header = QLabel("Gates")
        header.setStyleSheet("font-weight: bold; font-size: 16px; margin-bottom: 5px;")
        main_layout.addWidget(header)

        # Scroll Area for many gates
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")
        
        content = QWidget()
        layout = QVBoxLayout(content)
        
        # --- EXPANDED GATE LIST ---
        # Grouped by type for logic (visually just a list here)
        # Updated List
        gates = [
            'H', 'X', 'Y', 'Z', 
            'RX', 'RY', 'RZ', 'P', # Rotations
            'CX', 'SWAP', 'CZ'

        ]
        for g in gates:
            btn = DraggableButton(g)
            layout.addWidget(btn)
        
        layout.addStretch()
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
