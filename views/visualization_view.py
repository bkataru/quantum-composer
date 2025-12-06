from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class VisualizationView(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Dark Theme Matplotlib
        plt.style.use('dark_background') # Built-in dark style
        self.figure = Figure(figsize=(5, 4), dpi=100)
        
        # Match Window BG
        self.figure.patch.set_facecolor('#121212') 
        
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def plot_histogram(self, counts):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        states = list(counts.keys())
        values = list(counts.values())
        
        # White Bars for high contrast
        bars = ax.bar(states, values, color='#FFFFFF')
        
        ax.set_title("Simulation Results", color='white')
        ax.set_ylabel("Counts", color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        
        # Minimalist Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        
        ax.set_facecolor('#121212')

        self.canvas.draw()
