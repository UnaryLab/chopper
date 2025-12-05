# Adding plots

Adding a script in this directory adds the script name to "available plots" in the GUI.

Plot scripts are expected to have the following functions:

1) `get_data(**kwargs)`

- `kwargs` are expected to have default values and type annotations.
- Type annotations tell the GUI which selection widget to add for a given variable.
- Default values populate the variable's selection widget.

2) `draw(fig: Figure, input_data, **kwargs)`

- `fig` is the GUI canvas for the plot to be drawn on.
- `input_data` is the return value of `get_data` above.
- `kwargs` are utilized just like `get_data`.
- These `kwargs` are for rapidly customizing the plot without re-loading data
