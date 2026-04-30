"""Interactive GUI for visualizing and analyzing distributed training traces.

Provides a Qt-based GUI application for loading trace data, selecting plots,
configuring parameters, and generating visualizations. Supports hot-reloading
of plot modules for rapid development iteration.
"""
import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QLabel,
    QScrollArea,
    QGroupBox,
    QSplitter,
    QFileDialog,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import chopper.plots

from chopper.common.annotations import PaperMode
from chopper.selectors import (
    PlotSelection,
    BoolSelection,
    StrSelection,
    IntSelection,
    FloatSelection,
    StrlistSelection,
    IntlistSelection,
    FloatlistSelection,
    PaperModeSelection,
)

import importlib
import inspect
import pkgutil
import enum
import pickle
import os


class SelectionType(enum.Enum):
    """Enumeration of selection panel types.

    Attributes:
        plot: Plot module selection
        data: Data loading parameter selection
        draw: Plot drawing parameter selection
    """
    plot = enum.auto()
    data = enum.auto()
    draw = enum.auto()


class Selections(QWidget):
    """Widget container for plot parameter selection panels.

    Manages three collapsible panels (plot, data, draw) containing parameter
    selectors dynamically generated from plot module function signatures.

    Attributes:
        plot_layout: Layout for plot selection widgets
        data_box: Collapsible group box for data loading parameters
        draw_box: Collapsible group box for drawing parameters
    """
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


class LoadDataThread(QThread):
    """Background thread for loading and processing trace data.

    Executes the plot modules get_data() function in a separate thread
    to prevent GUI freezing during file I/O operations.

    Attributes:
        module: Plot module containing get_data function
        param_map: Dict of parameter names to values
    """
    finished = pyqtSignal(object)
    error = pyqtSignal(Exception)

    def __init__(self, plot, data_selections):
        super().__init__()
        self.plot = plot
        self.data_selections = data_selections

    def run(self):
        try:
            result = self.plot.get_data(**self.data_selections)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(e)


class MatplotlibWidget(QWidget):
    """Qt widget for embedding matplotlib figures.

    Provides a canvas for displaying matplotlib plots with toolbar integration
    for zoom, pan, and save functionality.

    Attributes:
        figure: Matplotlib Figure object
        canvas: Qt canvas for rendering the figure
    """
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
        self.cache_dir = "chopper_cache"
        self.load_thread = None
        self.loading_plot = None

        self.selections = Selections()

        # Add matplotlib navigation toolbar for zoom, pan, etc.
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Create a widget to hold canvas and toolbar
        canvas_widget = QWidget()
        canvas_layout = QVBoxLayout()
        canvas_layout.addWidget(self.toolbar)
        canvas_layout.addWidget(self.canvas)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_widget.setLayout(canvas_layout)

        # Use QSplitter for resizable divider
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.selections)
        splitter.addWidget(canvas_widget)
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
        self.save_button = QPushButton("save figure")
        self.save_button.clicked.connect(self.save_figure)
        self.save_button.setDisabled(True)
        refresh_layout.addWidget(self.data_button)
        refresh_layout.addWidget(self.draw_button)
        refresh_layout.addWidget(self.save_button)

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

        # Load cached selections for this plot
        self.load_cache()

        # Reset button states (preserve loading state if this plot is loading)
        if self.loading_plot == self.plot:
            self.data_button.setEnabled(False)
            self.data_button.setText("loading...")
            self.draw_button.setEnabled(False)
            self.save_button.setEnabled(False)
        else:
            self.data_button.setEnabled(True)
            self.data_button.setText("load data")
            self.draw_button.setEnabled(self.plot in self.plot_data)
            self.save_button.setEnabled(self.plot in self.plot_data)
        self.plot_selection.reload_button.setEnabled(True)

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
            inspect.formatannotation(str): StrSelection,
            inspect.formatannotation(int): IntSelection,
            inspect.formatannotation(float): FloatSelection,
            inspect.formatannotation(PaperMode): PaperModeSelection,
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

        # Disable buttons and show loading status
        self.data_button.setEnabled(False)
        self.data_button.setText("loading...")
        self.draw_button.setEnabled(False)

        # Track which plot is loading
        self.loading_plot = self.plot
        self.load_thread = LoadDataThread(self.loading_plot, self.data_selections[self.plot])
        self.load_thread.finished.connect(lambda result: self.on_load_finished(self.loading_plot, result))
        self.load_thread.error.connect(lambda e: self.on_load_error(self.loading_plot, e))
        self.load_thread.start()

    def on_load_finished(self, loaded_plot, result):
        self.plot_data[loaded_plot] = result
        self.loading_plot = None
        # Only update UI if we're still on the same plot
        if loaded_plot == self.plot:
            self.draw_button.setEnabled(True)
            self.save_button.setEnabled(True)
            self.data_button.setEnabled(True)
            self.data_button.setText("load data")
        self.save_cache()

    def on_load_error(self, loaded_plot, e):
        self.loading_plot = None
        # Only show error if we're still on the same plot
        if loaded_plot == self.plot:
            self.data_button.setEnabled(True)
            self.data_button.setText("load data")
            error_msg = QMessageBox(self)
            error_msg.setIcon(QMessageBox.Icon.Critical)
            error_msg.setWindowTitle("Error Loading Data")
            error_msg.setText(f"Failed to load data: {type(e).__name__}")
            error_msg.setDetailedText(str(e))
            error_msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            error_msg.exec()

    def draw_plot(self):
        try:
            if self.plot not in self.plot_data:
                raise RuntimeError("Plot data is not loaded. Click 'load data' first.")

            self.draw_selections[self.plot] = self.selections.get_draw_sels()

            # Check if paper mode is enabled (now a PaperMode dataclass)
            paper_mode = self.draw_selections[self.plot].get('paper_mode', None)

            if paper_mode is not None and paper_mode.enabled:
                # Apply paper-specific rcParams
                import matplotlib.pyplot as plt

                plt.rcParams['font.family'] = 'Gill Sans'
                plt.rcParams['font.size'] = 8
                plt.rcParams['axes.labelsize'] = 8
                plt.rcParams['axes.titlesize'] = 8
                plt.rcParams['xtick.labelsize'] = 8
                plt.rcParams['ytick.labelsize'] = 8
                plt.rcParams['legend.fontsize'] = 8
                plt.rcParams['figure.titlesize'] = 8
                plt.rcParams['hatch.color'] = 'black'
                plt.rcParams['hatch.linewidth'] = 0.5
                plt.rcParams['mathtext.default'] = 'regular'
                plt.rcParams['mathtext.fontset'] = 'cm'

                # Calculate figure size based on paper column width
                width_pt = 243.91125 * paper_mode.ncol
                width_in = width_pt / 72.27
                height_in = width_in * paper_mode.figsize_ratio
                figsize = (width_in, height_in)

                # Recreate figure with proper size
                self.figure.set_size_inches(figsize)

            self.plot.draw(
                self.figure, self.plot_data[self.plot], **self.draw_selections[self.plot]
            )
            self.canvas.draw()
            # Save selections to cache
            self.save_cache()
        except Exception as e:
            # Show error dialog instead of crashing
            error_msg = QMessageBox(self)
            error_msg.setIcon(QMessageBox.Icon.Critical)
            error_msg.setWindowTitle("Error Drawing Plot")
            error_msg.setText(f"Failed to draw plot: {type(e).__name__}")
            error_msg.setDetailedText(str(e))
            error_msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            error_msg.exec()

    def save_figure(self):
        """Save figure at 300 DPI without the paper mode border."""
        try:
            # Get save path from user
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Figure",
                "",
                "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)",
            )
            if not path:
                return

            # Temporarily remove paper mode border patches
            saved_patches = list(self.figure.patches)
            self.figure.patches.clear()

            # Save at 300 DPI
            self.figure.savefig(path, dpi=300)

            # Restore patches
            self.figure.patches.extend(saved_patches)
            self.canvas.draw()

        except Exception as e:
            error_msg = QMessageBox(self)
            error_msg.setIcon(QMessageBox.Icon.Critical)
            error_msg.setWindowTitle("Error Saving Figure")
            error_msg.setText(f"Failed to save figure: {type(e).__name__}")
            error_msg.setDetailedText(str(e))
            error_msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            error_msg.exec()

    def reload_module(self):
        """Reload the current plot module to pick up code changes."""
        try:
            if self.plot is None:
                return

            # Store the old module reference
            old_plot = self.plot

            # Reload the module
            self.plot = importlib.reload(self.plot)

            # If we had cached data for the old module, transfer it to the new one
            if old_plot in self.plot_data:
                self.plot_data[self.plot] = self.plot_data.pop(old_plot)
            if old_plot in self.data_selections:
                self.data_selections[self.plot] = self.data_selections.pop(old_plot)
            if old_plot in self.draw_selections:
                self.draw_selections[self.plot] = self.draw_selections.pop(old_plot)

            # Refresh the parameter selections to pick up any signature changes
            self.refresh_selections()

            # If we have data loaded, automatically redraw with the new code
            if self.plot in self.plot_data:
                self.draw_plot()

            # Show success message
            success_msg = QMessageBox(self)
            success_msg.setIcon(QMessageBox.Icon.Information)
            success_msg.setWindowTitle("Module Reloaded")
            success_msg.setText("Plot module reloaded successfully!")
            success_msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            success_msg.exec()

        except Exception as e:
            # Show error dialog
            error_msg = QMessageBox(self)
            error_msg.setIcon(QMessageBox.Icon.Critical)
            error_msg.setWindowTitle("Error Reloading Module")
            error_msg.setText(f"Failed to reload module: {type(e).__name__}")
            error_msg.setDetailedText(str(e))
            error_msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            error_msg.exec()

    def save_cache(self):
        """Save current plot's selections to disk."""
        if self.plot is None:
            return
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            plot_name = self.plot.__name__.split('.')[-1]
            if self.plot in self.data_selections:
                with open(f"{self.cache_dir}/{plot_name}_data.pkl", 'wb') as f:
                    pickle.dump(self.data_selections[self.plot], f)
            if self.plot in self.draw_selections:
                with open(f"{self.cache_dir}/{plot_name}_draw.pkl", 'wb') as f:
                    pickle.dump(self.draw_selections[self.plot], f)
        except Exception:
            pass

    def load_cache(self):
        """Load current plot's selections from disk."""
        if self.plot is None:
            return
        try:
            plot_name = self.plot.__name__.split('.')[-1]
            data_file = f"{self.cache_dir}/{plot_name}_data.pkl"
            draw_file = f"{self.cache_dir}/{plot_name}_draw.pkl"
            if os.path.exists(data_file):
                with open(data_file, 'rb') as f:
                    self.data_selections[self.plot] = pickle.load(f)
            if os.path.exists(draw_file):
                with open(draw_file, 'rb') as f:
                    self.draw_selections[self.plot] = pickle.load(f)
        except Exception:
            pass


class MainWindow(QMainWindow):
    """Main application window for Chopper trace visualization.

    Provides the primary GUI interface with plot selection, parameter configuration,
    data loading, and visualization. Supports saving/loading sessions and exporting plots.

    Attributes:
        matplotlib_widget: Widget containing the plot canvas
        selections: Widget containing parameter selectors
        plot_button: Button to regenerate the current plot
        data: Cached data from get_data() function
        current_module: Currently selected plot module
    """
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
