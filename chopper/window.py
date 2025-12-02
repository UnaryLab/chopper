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
    QMenu,
    QScrollArea,
    QFrame,
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


class PlotSelection(QWidget):
    def __init__(self, vals, parent=None):
        self.parent = parent
        super().__init__(parent)
        label = QLabel('available plots')

        self.list = QListWidget()

        for val in vals:
            item = QListWidgetItem(val)
            self.list.addItem(item)

        self.list.itemSelectionChanged.connect(self.on_selection_changed)

        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.list)
        self.setLayout(layout)

    def on_selection_changed(self):
        self.parent.refresh_selections()

    def get_selected(self):
        return self.list.selectedItems()


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


class Selections(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.container = QWidget()
        self.container_layout = QVBoxLayout()
        self.container.setLayout(self.container_layout)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.container)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.divider = QFrame()
        self.divider.setFrameShape(QFrame.Shape.VLine)
        self.divider.setCursor(Qt.CursorShape.SplitHCursor)
        self.divider.setFixedWidth(8)
        self.divider.setStyleSheet("QFrame { background-color: #fff; }")
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.scroll)
        self.layout.addWidget(self.divider)
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
                self.scroll.setFixedWidth(new_width)

    def end_drag(self, event):
        self.dragging = False
        self.container_layout.isQuickItemType

    def clear(self):
        while self.container_layout.count() > 1:
            item = self.container_layout.takeAt(1)
            item.widget().deleteLater()


class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        inner = QHBoxLayout()
        self.selections = Selections()
        refresh_layout = QHBoxLayout()
        inner.addWidget(self.selections)
        inner.addWidget(self.canvas, stretch=1)
        layout.addLayout(inner)
        self.refresh_button = QPushButton("refresh plots")
        self.refresh_button.clicked.connect(lambda: self.refresh_plots(False))
        self.redraw_button = QPushButton("redraw plot")
        self.redraw_button.clicked.connect(self.redraw_plot)
        refresh_layout.addWidget(self.refresh_button)
        refresh_layout.addWidget(self.redraw_button)
        layout.addLayout(refresh_layout)
        self.refresh_plots(True)
        self.setLayout(layout)
        self.canvas.draw()

    def refresh_selections(self):
        self.selections.clear()
        self.selections.add_selection(self.plot_selection)

        selected = self.plot_selection.get_selected()
        if selected != []:
            assert len(selected) == 1
            self.plot = importlib.import_module(
                f"chopper.plots.{selected[0].text()}")
            sig = inspect.signature(self.plot.get_data)
            defaults = {
                (name, param.annotation): param.default
                for name, param in sig.parameters.items()
                if param.default is not inspect._empty
            }
            for (name, ann), vals in defaults.items():
                if ann == Tuple[str]:
                    self.selections.add_selection(
                        StrSelection(vals, f"{name}: {ann}"))
                elif ann == bool:
                    self.selections.add_selection(
                        BoolSelection(vals, f"{name}: {ann}"))
                elif ann == Tuple[Framework]:
                    self.selections.add_selection(
                        FrameworkSelection(vals, f"{name}: {ann}"))
                else:
                    raise TypeError("Unknown annotation")

    def refresh_plots(self, refresh_sels: bool = False):
        self.plot_modules = tuple(
            name for _, name, _ in pkgutil.iter_modules(chopper.plots.__path__))
        self.plot_selection = PlotSelection(self.plot_modules, parent=self)
        if refresh_sels:
            self.refresh_selections()

    def redraw_plot(self):
        input_data = self.plot.get_data()
        self.plot.draw(self.figure, input_data)
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
