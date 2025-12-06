from PyQt6.QtWidgets import QPlainTextEdit, QWidget, QVBoxLayout, QLabel

class CodeEditorView(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        header = QLabel("Qiskit Code (Editable)")
        header.setStyleSheet("font-weight: bold; margin-bottom: 5px; color: #FFFFFF;") 
        layout.addWidget(header)

        self.editor = QPlainTextEdit()
        self.editor.setReadOnly(False)
        
        # Dark Theme Editor
        self.editor.setStyleSheet("""
            QPlainTextEdit {
                background-color: #000000;
                color: #FFFFFF;
                border: 1px solid #FFFFFF;
                font-family: 'Consolas', 'Monaco', monospace;
                padding: 10px;
            }
        """)
        layout.addWidget(self.editor)

    def update_code(self, code):
        self.editor.setPlainText(code)

    def get_code(self):
        return self.editor.toPlainText()
