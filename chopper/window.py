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
    QLineEdit,
    QSplitter,
    QFileDialog,
    QToolButton,
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import QSize
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import chopper.plots

from chopper.common.annotations import Framework

import importlib
import inspect
import pkgutil
import enum
from typing import Any
from abc import abstractmethod


class PlotSelection(QWidget):
    def __init__(self, vals, parent=None):
        self.parent = parent
        super().__init__(parent)
        label = QLabel("available plots")

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
        label = QLabel(f"{self.name}: {self.ann}")

        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.list.itemDoubleClicked.connect(self.show_dropdown)

        for val in vals:
            item = QListWidgetItem(val.name)
            item.setData(Qt.ItemDataRole.UserRole, val)
            self.list.addItem(item)

        self.add_button = QPushButton("add")
        self.add_button.clicked.connect(self.add_item)

        self.remove_button = QPushButton("remove")
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
                lambda checked, f=framework, i=item: self.set_framework(i, f)
            )

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
        return self.name, tuple(
            self.list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.list.count())
        )


class BoolSelection(QCheckBox):
    def __init__(self, val: bool, name: str, ann, parent=None):
        super().__init__(f"{name}: {ann}", parent)
        self.setChecked(val)
        self.name = name
        self.ann = ann

    def get_selections(self):
        return self.name, self.isChecked()


class TextlistSelection(QWidget):
    def __init__(self, strings: list[Any], name: str, ann, parent=None):
        super().__init__(parent)

        self.name = name
        self.ann = ann

        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        # Better styling for cleaner look
        self.list.setAlternatingRowColors(True)

        for s in strings:
            item = QListWidgetItem(str(s))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.list.addItem((item))

        # Mac-style +/- buttons
        self.add_button = QPushButton("+")
        self.add_button.setFixedSize(30, 30)
        self.add_button.clicked.connect(self.add_item)

        self.remove_button = QPushButton("−")
        self.remove_button.setFixedSize(30, 30)
        self.remove_button.clicked.connect(self.remove_item)

        # Buttons at bottom like Mac style
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.remove_button)
        button_layout.addStretch()

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"{self.name}: {self.ann}"))
        layout.addWidget(self.list)
        layout.addLayout(button_layout)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def add_item(self):
        item = QListWidgetItem("new item")
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        self.list.addItem(item)
        self.list.setCurrentItem(item)
        self.list.editItem(item)

    def remove_item(self):
        for item in self.list.selectedItems():
            row = self.list.row(item)
            self.list.takeItem(row)


class StrlistSelection(QWidget):
    def __init__(self, strings: list[str], name: str, ann, parent=None):
        super().__init__(parent)

        self.name = name
        self.ann = ann

        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.list.setAlternatingRowColors(True)

        for s in strings:
            item = QListWidgetItem(str(s))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.list.addItem(item)

        # Mac-style +/- buttons
        self.add_button = QPushButton("+")
        self.add_button.setFixedSize(30, 30)
        self.add_button.clicked.connect(self.add_item)

        self.remove_button = QPushButton("−")
        self.remove_button.setFixedSize(30, 30)
        self.remove_button.clicked.connect(self.remove_item)

        # File browse button for string lists
        self.browse_file_button = QPushButton("File...")
        self.browse_file_button.clicked.connect(self.browse_file)

        # Buttons at bottom
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.remove_button)
        button_layout.addWidget(self.browse_file_button)
        button_layout.addStretch()

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"{self.name}: {self.ann}"))
        layout.addWidget(self.list)
        layout.addLayout(button_layout)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def add_item(self):
        item = QListWidgetItem("new item")
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        self.list.addItem(item)
        self.list.setCurrentItem(item)
        self.list.editItem(item)

    def remove_item(self):
        for item in self.list.selectedItems():
            row = self.list.row(item)
            self.list.takeItem(row)

    def browse_file(self):
        """Open file dialog to select file to add to list."""
        path = QFileDialog.getOpenFileName(self, f"Select File")[0]
        if path:
            item = QListWidgetItem(path)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.list.addItem(item)

    def get_selections(self):
        return self.name, tuple(
            self.list.item(i).text() for i in range(self.list.count())
        )


class IntlistSelection(QWidget):
    def __init__(self, ints: list[int], name: str, ann, parent=None):
        super().__init__(parent)

        self.name = name
        self.ann = ann

        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.list.setAlternatingRowColors(True)

        for i in ints:
            item = QListWidgetItem(str(i))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.list.addItem(item)

        # Mac-style +/- buttons (no browse for numbers)
        self.add_button = QPushButton("+")
        self.add_button.setFixedSize(30, 30)
        self.add_button.clicked.connect(self.add_item)

        self.remove_button = QPushButton("−")
        self.remove_button.setFixedSize(30, 30)
        self.remove_button.clicked.connect(self.remove_item)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.remove_button)
        button_layout.addStretch()

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"{self.name}: {self.ann}"))
        layout.addWidget(self.list)
        layout.addLayout(button_layout)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def add_item(self):
        item = QListWidgetItem("0")
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        self.list.addItem(item)
        self.list.setCurrentItem(item)
        self.list.editItem(item)

    def remove_item(self):
        for item in self.list.selectedItems():
            row = self.list.row(item)
            self.list.takeItem(row)

    def get_selections(self):
        return self.name, tuple(
            int(self.list.item(i).text()) for i in range(self.list.count())
        )


class FloatlistSelection(QWidget):
    def __init__(self, floats: list[float], name: str, ann, parent=None):
        super().__init__(parent)

        self.name = name
        self.ann = ann

        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.list.setAlternatingRowColors(True)

        for f in floats:
            item = QListWidgetItem(str(f))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.list.addItem(item)

        # Mac-style +/- buttons (no browse for numbers)
        self.add_button = QPushButton("+")
        self.add_button.setFixedSize(30, 30)
        self.add_button.clicked.connect(self.add_item)

        self.remove_button = QPushButton("−")
        self.remove_button.setFixedSize(30, 30)
        self.remove_button.clicked.connect(self.remove_item)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.remove_button)
        button_layout.addStretch()

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"{self.name}: {self.ann}"))
        layout.addWidget(self.list)
        layout.addLayout(button_layout)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def add_item(self):
        item = QListWidgetItem("0.0")
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        self.list.addItem(item)
        self.list.setCurrentItem(item)
        self.list.editItem(item)

    def remove_item(self):
        for item in self.list.selectedItems():
            row = self.list.row(item)
            self.list.takeItem(row)

    def get_selections(self):
        return self.name, tuple(
            float(self.list.item(i).text()) for i in range(self.list.count())
        )


class TextSelection(QWidget):
    def __init__(self, string: Any, name: str, ann, parent=None):
        super().__init__(parent)

        self.name = name
        self.ann = ann
        self.line_edit = QLineEdit(str(string))

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"{self.name}: {self.ann}"))
        layout.addWidget(self.line_edit)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)


class StrSelection(QWidget):
    def __init__(self, string: str, name: str, ann, parent=None):
        super().__init__(parent)

        self.name = name
        self.ann = ann
        self.line_edit = QLineEdit(str(string))

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"{self.name}: {self.ann}"))

        # Add horizontal layout for text field + browse button
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.line_edit)

        # File browse button for strings
        browse_file_btn = QPushButton("File...")
        browse_file_btn.clicked.connect(self.browse_file)
        input_layout.addWidget(browse_file_btn)

        layout.addLayout(input_layout)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def browse_file(self):
        """Open file dialog to select file."""
        path = QFileDialog.getOpenFileName(self, f"Select File", self.line_edit.text() or ".")[0]
        if path:
            self.line_edit.setText(path)

    def get_selections(self):
        return self.name, self.line_edit.text()


class IntSelection(QWidget):
    def __init__(self, num: int, name: str, ann, parent=None):
        super().__init__(parent)

        self.name = name
        self.ann = ann
        self.line_edit = QLineEdit(str(num))

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"{self.name}: {self.ann}"))
        layout.addWidget(self.line_edit)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def get_selections(self):
        return self.name, int(self.line_edit.text())


class FloatSelection(QWidget):
    def __init__(self, num: float, name: str, ann, parent=None):
        super().__init__(parent)

        self.name = name
        self.ann = ann
        self.line_edit = QLineEdit(str(num))

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"{self.name}: {self.ann}"))
        layout.addWidget(self.line_edit)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def get_selections(self):
        return self.name, float(self.line_edit.text())


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
            lambda checked: self.toggle_box(self.data_box, checked)
        )

        self.draw_box = QGroupBox("draw args")
        self.draw_box_placeholder = QLabel("check box to expand")
        self.draw_box.setCheckable(True)
        self.draw_box.setChecked(False)
        self.draw_layout = QVBoxLayout()
        self.draw_layout.addWidget(self.draw_box_placeholder)
        self.draw_box.setLayout(self.draw_layout)
        self.toggle_box(self.draw_box, False)
        self.draw_box.toggled.connect(
            lambda checked: self.toggle_box(self.draw_box, checked)
        )

        self.container_layout.addLayout(self.plot_layout)
        self.container_layout.addWidget(self.data_box)
        self.container_layout.addWidget(self.draw_box)
        self.container.setLayout(self.container_layout)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.container)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Use a simple layout without custom divider
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.scroll)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

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

    def clear(self):
        # pop from layout
        # take index 1 to skip placeholder text
        for _ in range(self.data_layout.count() - 1):
            item = self.data_layout.takeAt(1)
            item.widget().deleteLater()
        for _ in range(self.draw_layout.count() - 1):
            item = self.draw_layout.takeAt(1)
            item.widget().deleteLater()

    def get_data_sels(self):
        return dict(
            self.data_layout.itemAt(i).widget().get_selections()
            for i in range(
                1,  # skip placeholder text
                self.data_layout.count(),
            )
        )

    def get_draw_sels(self):
        return dict(
            self.draw_layout.itemAt(i).widget().get_selections()
            for i in range(
                1,  # skip placeholder text
                self.draw_layout.count(),
            )
        )


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

        self.selections = Selections()

        # Use QSplitter for resizable divider
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.selections)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)  # Selections panel doesn't stretch
        splitter.setStretchFactor(1, 1)  # Canvas stretches
        splitter.setSizes([300, 700])  # Initial sizes

        layout = QVBoxLayout()
        layout.addWidget(splitter)

        refresh_layout = QHBoxLayout()

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
            name for _, name, _ in pkgutil.iter_modules(chopper.plots.__path__)
        )
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
        self.plot = importlib.import_module(f"chopper.plots.{plot_selected[0].text()}")
        self.data_button.setEnabled(True)
        self.draw_button.setEnabled(self.plot in self.plot_data)

        data_sig = inspect.signature(self.plot.get_data)
        data_defaults = {
            (name, inspect.formatannotation(param.annotation)): param.default
            for name, param in data_sig.parameters.items()
            if param.default is not inspect._empty
        }

        selection_map = {
            inspect.formatannotation(list[str]): StrlistSelection,
            inspect.formatannotation(list[int]): IntlistSelection,
            inspect.formatannotation(list[float]): FloatlistSelection,
            inspect.formatannotation(bool): BoolSelection,
            inspect.formatannotation(list[Framework]): FrameworkSelection,
            inspect.formatannotation(str): StrSelection,
            inspect.formatannotation(int): IntSelection,
            inspect.formatannotation(float): FloatSelection,
        }

        plot_data_slot = self.data_selections.setdefault(self.plot, {})
        for (name, ann), vals in data_defaults.items():
            cache_vals = plot_data_slot.setdefault(name, vals)
            self.selections.add_selection(
                selection_map[ann](cache_vals, name, ann), SelectionType.data
            )

        draw_sig = inspect.signature(self.plot.draw)
        draw_defaults = {
            (name, inspect.formatannotation(param.annotation)): param.default
            for name, param in draw_sig.parameters.items()
            if param.default is not inspect._empty
        }

        plot_draw_slot = self.draw_selections.setdefault(self.plot, {})
        for (name, ann), vals in draw_defaults.items():
            cache_vals = plot_draw_slot.setdefault(name, vals)
            self.selections.add_selection(
                selection_map[ann](cache_vals, name, ann), SelectionType.draw
            )

    def load_data(self):
        self.data_selections[self.plot] = self.selections.get_data_sels()
        self.plot_data[self.plot] = self.plot.get_data(
            **self.data_selections[self.plot]
        )
        self.draw_button.setEnabled(True)

    def draw_plot(self):
        assert self.plot in self.plot_data, "Plot data is not loaded"
        self.draw_selections[self.plot] = self.selections.get_draw_sels()
        self.plot.draw(
            self.figure, self.plot_data[self.plot], **self.draw_selections[self.plot]
        )
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
