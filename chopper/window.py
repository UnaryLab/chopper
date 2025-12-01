import sys
from PyQt6.QtWidgets import (
    QApplication,
    QListWidgetItem,
    QMainWindow,
    QListWidget,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QLabel,
    QCheckBox,
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import chopper.plots

from chopper.common.annotations import Framework

import importlib
import inspect
import pkgutil
from typing import Tuple


from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox


from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget, QListWidgetItem, QMenu
from PyQt6.QtCore import Qt


class FrameworkSelection(QWidget):
    def __init__(self, vals: list, name: str, parent=None):
        super().__init__(parent)
        label = QLabel(name)

        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.list.itemDoubleClicked.connect(self.show_dropdown)

        for val in vals:
            item = QListWidgetItem(val.name)
            item.setData(Qt.ItemDataRole.UserRole, val)
            self.list.addItem(item)

        self.add_button = QPushButton('add')
        self.add_button.clicked.connect(self.add_item)

        self.remove_button = QPushButton('remove')
        self.remove_button.clicked.connect(self.remove_item)

        add_rem = QHBoxLayout()
        add_rem.addWidget(self.add_button)
        add_rem.addWidget(self.remove_button)

        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addLayout(add_rem)
        layout.addWidget(self.list)
        self.setLayout(layout)

    def show_dropdown(self, item):
        menu = QMenu(self)
        for framework in Framework:
            action = menu.addAction(framework.name)
            action.triggered.connect(
                lambda checked, f=framework, i=item: self.set_framework(i, f))

        # Show menu at the item position
        rect = self.list.visualItemRect(item)
        pos = self.list.mapToGlobal(rect.topRight())
        menu.exec(pos)

    def set_framework(self, item, framework):
        item.setText(framework.name)
        item.setData(Qt.ItemDataRole.UserRole, framework)

    def add_item(self):
        item = QListWidgetItem(Framework.FSDPv1.name)
        item.setData(Qt.ItemDataRole.UserRole, Framework.FSDPv1)
        self.list.addItem(item)

    def remove_item(self):
        for item in self.list.selectedItems():
            row = self.list.row(item)
            self.list.takeItem(row)

    def get_values(self):
        return [self.list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.list.count())]


class BoolSelection(QCheckBox):
    def __init__(self, val: bool, name: str, parent=None):
        super().__init__(name, parent)
        self.setChecked(val)


class StrSelection(QWidget):
    def __init__(self, strings: Tuple[str], name: str, parent=None):
        super().__init__(parent)

        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list.setDragDropMode(QListWidget.DragDropMode.InternalMove)

        for s in strings:
            item = QListWidgetItem(s)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.list.addItem(item)

        self.add_button = QPushButton('add')
        self.add_button.clicked.connect(self.add_item)

        self.remove_button = QPushButton('remove')
        self.remove_button.clicked.connect(self.remove_item)

        add_rem = QHBoxLayout()
        add_rem.addWidget(self.add_button)
        add_rem.addWidget(self.remove_button)

        layout = QVBoxLayout()
        layout.addWidget(QLabel(name))
        layout.addLayout(add_rem)
        layout.addWidget(self.list)
        self.setLayout(layout)

    def add_item(self):
        item = QListWidgetItem("add string")
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        self.list.addItem(item)
        self.list.setCurrentItem(item)
        self.list.editItem(item)

    def remove_item(self):
        for item in self.list.selectedItems():
            row = self.list.row(item)
            self.list.takeItem(row)


class Selection(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.container = QWidget()
        self.container_layout = QVBoxLayout()
        self.container.setLayout(self.container_layout)

        self.divider = QPushButton('⇿')
        self.divider.setCursor(Qt.CursorShape.SplitHCursor)
        self.divider.setFixedWidth(20)

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.container, stretch=0)
        self.layout.addWidget(self.divider, stretch=0)
        self.setLayout(self.layout)

        self.dragging = False
        self.divider.mousePressEvent = self.start_drag
        self.divider.mouseMoveEvent = self.do_drag
        self.divider.mouseReleaseEvent = self.end_drag

    def add_selection(self, selection):
        self.container_layout.addWidget(selection)

    def start_drag(self, event):
        self.dragging = True

    def do_drag(self, event):
        if self.dragging:
            global_pos = self.divider.mapToGlobal(event.pos())
            local_pos = self.mapFromGlobal(global_pos)
            new_width = local_pos.x()
            if new_width >= 0:
                self.container.setFixedWidth(new_width)

    def end_drag(self, event):
        self.dragging = False


class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        inner = QHBoxLayout()
        self.selection = Selection()
        inner.addWidget(self.selection)
        inner.addWidget(self.canvas, stretch=0)
        layout.addLayout(inner)
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
        plot = importlib.import_module(f"chopper.plots.{self.plot_modules[0]}")
        sig = inspect.signature(plot.get_data)
        defaults = {
            (name, param.annotation): param.default
            for name, param in sig.parameters.items()
            if param.default is not inspect._empty
        }
        for (name, ann), vals in defaults.items():
            if ann == Tuple[str]:
                self.selection.add_selection(
                    StrSelection(vals, f"{name}: {ann}"))
            elif ann == bool:
                self.selection.add_selection(
                    BoolSelection(vals, f"{name}: {ann}"))
            elif ann == Tuple[Framework]:
                self.selection.add_selection(
                    FrameworkSelection(vals, f"{name}: {ann}"))
            else:
                raise TypeError("Unknown annotation")

        # input_data = plot.get_data()
        print(defaults.keys())
        # plot.draw(self.figure)
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
