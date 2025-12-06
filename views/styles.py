# Standard Light Theme with BLACK Gates
COLOR_BG = "#F4F6F8"       # Light Gray Background
COLOR_WIRE = "#AAAAAA"     # Gray Wire
COLOR_GATE_BG = "#000000"  # Pure Black Gates
COLOR_GATE_FG = "#FFFFFF"  # White Text
COLOR_CNOT_BG = "#000000"  # Black Control Dot
COLOR_TARGET_BG = "#000000" # Black Target Cross
COLOR_HOVER = "#D0D0D0"    # Darker Hover

# Shared CSS
# FIX: Added min-width/height to force gates to fill the 50x50 grid cells
GATE_CSS = f"""
    background-color: {COLOR_GATE_BG};
    color: {COLOR_GATE_FG};
    border: 1px solid #333;
    border-radius: 4px;
    font-family: 'Segoe UI', sans-serif;
    font-weight: bold;
    font-size: 14px;
    min-width: 46px; 
    min-height: 46px;
"""

LIGHT_THEME = f"""
QMainWindow {{
    background-color: {COLOR_BG};
}}
QWidget {{
    font-family: 'Segoe UI', sans-serif;
    font-size: 14px;
    color: #333333;
}}
/* Palette Buttons */
QPushButton#GateButton {{
    {GATE_CSS}
    padding: 0px; /* Reset padding since we set min-width */
}}
QPushButton#GateButton:hover {{
    background-color: #444444;
}}
/* Grid Wires */
QFrame#CircuitLine {{
    background-color: {COLOR_WIRE};
    min-height: 2px;
    max-height: 2px;
}}
/* Drop Zones */
QLabel#DropZone {{
    background-color: transparent;
    border: none;
}}
QLabel#DropZone:hover {{
    background-color: {COLOR_HOVER};
    border-radius: 4px;
}}
/* Menus */
QMenu {{
    background-color: #FFFFFF;
    border: 1px solid #CCC;
}}
"""
