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
    QGroupBox,
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import chopper.plots

from chopper.common.annotations import Framework

import importlib
import inspect
import pkgutil
import enum
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

        self.list.itemSelectionChanged.connect(self.parent.refresh_selections)

        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.list)
        self.setLayout(layout)

    def get_selected(self):
        return self.list.selectedItems()


class FrameworkSelection(QWidget):
    def __init__(self, vals: list, name: str, ann, parent=None):
        super().__init__(parent)
        self.name = name
        self.ann = ann
        label = QLabel(f'{self.name}: {self.ann}')

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

    def get_selections(self):
        return self.name, tuple(self.list.item(i).data(Qt.ItemDataRole.UserRole)
                                for i in range(self.list.count()))


class BoolSelection(QCheckBox):
    def __init__(self, val: bool, name: str, ann, parent=None):
        super().__init__(f'{name}: {ann}', parent)
        self.setChecked(val)
        self.name = name
        self.ann = ann

    def get_selections(self):
        return self.name, self.isChecked()


class StrSelection(QWidget):
    def __init__(self, strings: Tuple[str], name: str, ann, parent=None):
        super().__init__(parent)

        self.name = name
        self.ann = ann
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
        layout.addWidget(QLabel(f'{self.name}: {self.ann}'))
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

    def get_selections(self):
        return self.name, tuple(self.list.item(i).text() for i in range(self.list.count()))


class SelectionType(enum.Enum):
    plot = enum.auto()
    data = enum.auto()
    draw = enum.auto()


class Selections(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.container = QWidget()
        self.container_layout = QVBoxLayout()

        self.plot_layout = QVBoxLayout()

        self.data_box = QGroupBox("data args")
        self.data_box_placeholder = QLabel("check box to expand")
        self.data_box.setCheckable(True)
        self.data_box.setChecked(False)
        self.data_layout = QVBoxLayout()
        self.data_layout.addWidget(self.data_box_placeholder)
        self.data_box.setLayout(self.data_layout)
        self.toggle_box(self.data_box, False)
        self.data_box.toggled.connect(
            lambda checked: self.toggle_box(self.data_box, checked))

        self.draw_box = QGroupBox("draw args")
        self.draw_box_placeholder = QLabel("check box to expand")
        self.draw_box.setCheckable(True)
        self.draw_box.setChecked(False)
        self.draw_layout = QVBoxLayout()
        self.draw_layout.addWidget(self.draw_box_placeholder)
        self.draw_box.setLayout(self.draw_layout)
        self.toggle_box(self.draw_box, False)
        self.draw_box.toggled.connect(
            lambda checked: self.toggle_box(self.draw_box, checked))

        self.container_layout.addLayout(self.plot_layout)
        self.container_layout.addWidget(self.data_box)
        self.container_layout.addWidget(self.draw_box)
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

    def toggle_box(self, group_box, checked):
        for i in range(group_box.layout().count()):
            widget = group_box.layout().itemAt(i).widget()
            # placeholder text visibility should be inverse of selections
            if i == 0:
                widget.setVisible(not checked)
            else:
                widget.setVisible(checked)

    def add_selection(self, selection, stype: SelectionType):
        match stype:
            case SelectionType.plot:
                self.plot_layout.addWidget(selection)
            case SelectionType.data:
                selection.setVisible(self.data_box.isChecked())
                self.data_layout.addWidget(selection)
            case SelectionType.draw:
                selection.setVisible(self.draw_box.isChecked())
                self.draw_layout.addWidget(selection)

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

    def clear(self):
        # pop from layout
        # take index 1 to skip placeholder text
        for _ in range(self.data_layout.count()-1):
            item = self.data_layout.takeAt(1)
            item.widget().deleteLater()
        for _ in range(self.draw_layout.count()-1):
            item = self.draw_layout.takeAt(1)
            item.widget().deleteLater()

    def get_data_sels(self):
        return dict(self.data_layout.itemAt(i).widget().get_selections()
                    for i in range(
            1,  # skip placeholder text
            self.data_layout.count()))

    def get_draw_sels(self):
        return dict(self.draw_layout.itemAt(i).widget().get_selections()
                    for i in range(
            1,  # skip placeholder text
            self.draw_layout.count()))


class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.plot = None
        # WARN hold onto all plot data and selections
        # Maybe wastes memory
        self.plot_data = {}
        self.data_selections = {}
        self.draw_selections = {}
        layout = QVBoxLayout()
        inner = QHBoxLayout()
        self.selections = Selections()
        refresh_layout = QHBoxLayout()
        inner.addWidget(self.selections)
        inner.addWidget(self.canvas, stretch=1)
        layout.addLayout(inner)

        self.data_button = QPushButton("load data")
        self.data_button.clicked.connect(self.load_data)
        self.data_button.setDisabled(True)
        self.draw_button = QPushButton("redraw plot")
        self.draw_button.clicked.connect(self.draw_plot)
        self.draw_button.setDisabled(True)
        refresh_layout.addWidget(self.data_button)
        refresh_layout.addWidget(self.draw_button)

        layout.addLayout(refresh_layout)
        self.plot_modules = tuple(
            name for _, name, _ in pkgutil.iter_modules(chopper.plots.__path__))
        self.plot_selection = PlotSelection(self.plot_modules, parent=self)
        self.refresh_selections()
        self.setLayout(layout)
        self.canvas.draw()

    def refresh_selections(self):
        self.selections.clear()
        self.selections.add_selection(self.plot_selection, SelectionType.plot)

        plot_selected = self.plot_selection.get_selected()
        if plot_selected == []:
            return
        assert len(plot_selected) == 1
        self.plot = importlib.import_module(
            f"chopper.plots.{plot_selected[0].text()}")
        self.data_button.setEnabled(True)
        self.draw_button.setEnabled(self.plot in self.plot_data)

        data_sig = inspect.signature(self.plot.get_data)
        data_defaults = {
            (name, inspect.formatannotation(param.annotation)): param.default
            for name, param in data_sig.parameters.items()
            if param.default is not inspect._empty
        }
        plot_data_slot = self.data_selections.setdefault(self.plot, {})
        for (name, ann), vals in data_defaults.items():
            cache_vals = plot_data_slot.setdefault(name, vals)
            if ann == inspect.formatannotation(Tuple[str]):
                self.selections.add_selection(
                    StrSelection(cache_vals, name, ann), SelectionType.data)
            elif ann == inspect.formatannotation(bool):
                self.selections.add_selection(
                    BoolSelection(cache_vals, name, ann), SelectionType.data)
            elif ann == inspect.formatannotation(Tuple[Framework]):
                self.selections.add_selection(
                    FrameworkSelection(cache_vals, name, ann), SelectionType.data)
            else:
                raise TypeError(f"Unknown annotation: {ann}")

        draw_sig = inspect.signature(self.plot.draw)
        draw_defaults = {
            (name, inspect.formatannotation(param.annotation)): param.default
            for name, param in draw_sig.parameters.items()
            if param.default is not inspect._empty
        }

        plot_draw_slot = self.draw_selections.setdefault(self.plot, {})
        for (name, ann), vals in draw_defaults.items():
            cache_vals = plot_draw_slot.setdefault(name, vals)
            if ann == inspect.formatannotation(Tuple[str]):
                self.selections.add_selection(
                    StrSelection(cache_vals, name, ann), SelectionType.draw)
            elif ann == inspect.formatannotation(bool):
                self.selections.add_selection(
                    BoolSelection(cache_vals, name, ann), SelectionType.draw)
            elif ann == inspect.formatannotation(Tuple[Framework]):
                self.selections.add_selection(
                    FrameworkSelection(vals, name, ann), SelectionType.draw)
            else:
                raise TypeError(f"Unknown annotation: {ann}")

    def load_data(self):
        self.data_selections[self.plot] = self.selections.get_data_sels()
        self.plot_data[self.plot] = self.plot.get_data(
            **self.data_selections[self.plot])
        self.draw_button.setEnabled(True)

    def draw_plot(self):
        assert self.plot in self.plot_data, "Plot data is not loaded"
        self.draw_selections[self.plot] = self.selections.get_draw_sels()
        print(self.draw_selections[self.plot])
        self.plot.draw(
            self.figure, self.plot_data[self.plot], **self.draw_selections[self.plot])
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
