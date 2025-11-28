import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import chopper.plots

import importlib
import pkgutil


class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.refresh_button = QPushButton("refresh plot")
        self.refresh_button.clicked.connect(self.refresh_plot)
        layout.addWidget(self.refresh_button)
        self.update_plots()
        self.setLayout(layout)
        self.canvas.draw()

    def update_plots(self):
        self.plot_modules = tuple(
            name for _, name, _ in pkgutil.iter_modules(chopper.plots.__path__))

    def refresh_plot(self):
        self.update_plots()
        plot = importlib.import_module(f'chopper.plots.{self.plot_modules[0]}')
        plot.draw(self.figure)
        self.canvas.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("chopper")
        self.main_window = MatplotlibWidget()
        self.setCentralWidget(self.main_window)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
