"""Qt widget selectors for plot parameter configuration.

Provides specialized Qt widgets for configuring different types of plot parameters
including plots, frameworks, booleans, strings, integers, floats, and lists.
Used by the main GUI window for interactive plot customization.
"""
from PyQt6.QtWidgets import (
    QListWidgetItem,
    QListWidget,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QLabel,
    QCheckBox,
    QMenu,
    QLineEdit,
    QFileDialog,
    QGroupBox,
)
from PyQt6.QtCore import Qt

from chopper.common.annotations import Framework, PaperMode


class PlotSelection(QWidget):
    """Widget for selecting available visualization plots.
    
    Displays a list of available plot modules and provides a reload
    button for hot-reloading plot code during development.
    
    Attributes:
        list: QListWidget containing available plot names
        reload_button: Button to reload the selected plot module
        parent: Parent widget reference
    """
    def __init__(self, vals, parent=None):
        self.parent = parent
        super().__init__(parent)
        label = QLabel("available plots")

        self.list = QListWidget()

        for val in vals:
            item = QListWidgetItem(val)
            self.list.addItem(item)

        self.list.itemSelectionChanged.connect(self.parent.refresh_selections)

        # Reload button under plot list
        self.reload_button = QPushButton("reload module")
        self.reload_button.clicked.connect(self.parent.reload_module)
        self.reload_button.setDisabled(True)

        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.list)
        layout.addWidget(self.reload_button)
        self.setLayout(layout)

    def get_selected(self):
        return self.list.selectedItems()


class FrameworkSelection(QWidget):
    """Widget for configuring framework selection parameters.
    
    Allows selecting and reordering Framework enum values (FSDPv1, FSDPv2)
    for trace analysis. Supports drag-and-drop reordering and double-click
    to change framework type.
    
    Attributes:
        name: Parameter name
        ann: Type annotation string
        list: QListWidget with Framework enum items
    """
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
    """Checkbox widget for boolean parameter configuration.
    
    Simple checkbox labeled with parameter name and type annotation.
    
    Attributes:
        name: Parameter name
        ann: Type annotation string
    """
    def __init__(self, val: bool, name: str, ann, parent=None):
        super().__init__(f"{name}: {ann}", parent)
        self.setChecked(val)
        self.name = name
        self.ann = ann

    def get_selections(self):
        return self.name, self.isChecked()


class TextlistSelection(QWidget):
    def __init__(self, strings: list, name: str, ann, parent=None):
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
        """Open file dialog to select file to replace selected item or add new."""
        path = QFileDialog.getOpenFileName(self, f"Select File")[0]
        if path:
            # Check if an item is selected
            selected_items = self.list.selectedItems()
            if selected_items:
                # Replace the first selected item
                selected_items[0].setText(path)
            else:
                # No selection, add new item
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
    def __init__(self, string, name: str, ann, parent=None):
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


class PaperModeSelection(QWidget):
    """Collapsible widget for paper mode settings.

    Provides a checkable group box that expands to show paper-specific
    layout parameters when enabled.

    Attributes:
        name: Parameter name
        ann: Type annotation string
        group_box: Collapsible QGroupBox containing settings
    """
    def __init__(self, val: PaperMode, name: str, ann, parent=None):
        super().__init__(parent)

        self.name = name
        self.ann = ann

        self.group_box = QGroupBox("paper_mode")
        self.group_box.setCheckable(True)
        self.group_box.setChecked(val.enabled)

        # Create input fields for each parameter
        self.left_edit = QLineEdit(str(val.left))
        self.right_edit = QLineEdit(str(val.right))
        self.bottom_edit = QLineEdit(str(val.bottom))
        self.top_edit = QLineEdit(str(val.top))
        self.wspace_edit = QLineEdit(str(val.wspace))
        self.hspace_edit = QLineEdit(str(val.hspace))
        self.ncol_edit = QLineEdit(str(val.ncol))
        self.figsize_ratio_edit = QLineEdit(str(val.figsize_ratio))
        self.legend_bbox_edit = QLineEdit(
            "" if val.legend_bbox is None else f"{val.legend_bbox[0]},{val.legend_bbox[1]}"
        )

        # Layout for the group box contents
        group_layout = QVBoxLayout()

        # Add labeled fields
        for label_text, edit in [
            ("left", self.left_edit),
            ("right", self.right_edit),
            ("bottom", self.bottom_edit),
            ("top", self.top_edit),
            ("wspace", self.wspace_edit),
            ("hspace", self.hspace_edit),
            ("ncol", self.ncol_edit),
            ("figsize_ratio", self.figsize_ratio_edit),
            ("legend_bbox (x,y)", self.legend_bbox_edit),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            row.addWidget(edit)
            group_layout.addLayout(row)

        self.group_box.setLayout(group_layout)

        # Toggle visibility of contents based on checkbox state
        self.toggle_contents(val.enabled)
        self.group_box.toggled.connect(self.toggle_contents)

        layout = QVBoxLayout()
        layout.addWidget(self.group_box)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def toggle_contents(self, checked: bool) -> None:
        """Show/hide the group box contents based on checked state."""
        group_layout = self.group_box.layout()
        assert group_layout is not None
        for i in range(group_layout.count()):
            item = group_layout.itemAt(i)
            assert item is not None
            inner_layout = item.layout()
            if inner_layout is not None:
                for j in range(inner_layout.count()):
                    inner_item = inner_layout.itemAt(j)
                    assert inner_item is not None
                    widget = inner_item.widget()
                    if widget:
                        widget.setVisible(checked)

    def get_selections(self):
        legend_bbox_text = self.legend_bbox_edit.text().strip()
        if legend_bbox_text:
            parts = legend_bbox_text.split(",")
            legend_bbox = (float(parts[0]), float(parts[1]))
        else:
            legend_bbox = None
        return self.name, PaperMode(
            enabled=self.group_box.isChecked(),
            left=float(self.left_edit.text()),
            right=float(self.right_edit.text()),
            bottom=float(self.bottom_edit.text()),
            top=float(self.top_edit.text()),
            wspace=float(self.wspace_edit.text()),
            hspace=float(self.hspace_edit.text()),
            ncol=int(self.ncol_edit.text()),
            figsize_ratio=float(self.figsize_ratio_edit.text()),
            legend_bbox=legend_bbox,
        )
