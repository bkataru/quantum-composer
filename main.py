import sys
from PyQt6.QtWidgets import QApplication
from views.main_window import MainWindow
from controllers.main_controller import MainController

def main():
    app = QApplication(sys.argv)
    
    # Initialize MVC
    window = MainWindow()
    controller = MainController(window) # Controller takes ownership of logic
    
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
