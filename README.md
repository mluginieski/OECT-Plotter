# Plotter Class

The `Plotter` class provides an easy-to-use interface for creating line and scatter plots with various customization options. It supports plotting multiple datasets on the same graph, adding annotations, adjusting axis scales, and saving the plots as image files.

It takes use of a `Settings.py` file where some important settings are defined, such as, graph size, font sizes, graph labels and units.

**Suggested usage:**
- `main.py` file: All data file names are defined in a `dictionary`, where the key is the file name, and additional information is provided through a `list`.
- `PlotterMain.py`: Various types of graphs are defined here through different functions, such as `IxV`, `Output`, `Transfer`, etc.
  
**Hierarchy:**
- main.py
    - PlotterMain.py
        - PlotterClass.py
          
**Examples:**</br>
Please, see some examples below or in the Codes/main.py and Codes/PlotterMain.py files.

## Class Constructor

### `__init__`

```python
def __init__(self, xlabel, ylabel, xunit='', yunit='', tiks_size=S.tiksSize, font_size=S.fontSize, fig_size=S.figSize):
```

**Description:** Initializes a new instance of the Plotter class.

**Parameters:**
- `xlabel` (<span style="color:green;">**str**</span>): Label for the x-axis.</br>
- `ylabel` (<span style="color:green">**str**</span>): Label for the y-axis.</br>
- `xunit` (<span style="color:green">**str**</span>, <span style="color:red">optional</span>): Unit for the x-axis, appended to the x-axis label. Default is an empty string.</br>
- `yunit` (<span style="color:green">**str**</span>, <span style="color:red">optional</span>): Unit for the y-axis, appended to the y-axis label. Default is an empty string.</br>
- `tiks_size` (<span style="color:green">**int**</span>, <span style="color:red">optional</span>): Font size for the axis ticks. Default is defined by `S.tiksSize`.</br>
- `font_size` (<span style="color:green">**int**</span>, <span style="color:red">optional</span>): Font size for axis labels and annotations. Default is defined by `S.fontSize`.</br>
- `fig_size` (<span style="color:green">**tuple**</span>, <span style="color:red">optional</span>): Figure size of the plot. Default is defined by `S.figSize`.

## Methods

### `set_folder`

```python
def set_folder(self, folder_name):
```

**Description:** Sets the folder where plots will be saved. Creates the folder if it does not exist.

**Parameters:**
- `folder_name` (<span style="color:green">**str**</span>): Path to the folder where plots will be saved.

### `add_plot`

```python
def add_plot(self, x, y, label=None, plot_type='line', annotate_text=None, annotate_pos=(0,0), annotate_cords='data'):
```

**Description:** Adds a dataset to be plotted.

**Parameters:**
- `x` (<span style="color:green">**array-like**</span>): Data for the x-axis.
- `y` (<span style="color:green">**array-like**</span>): Data for the y-axis.
- `label` (<span style="color:green">**str**</span>, <span style="color:red">optional</span>): Label for the dataset, used in the legend. Default is **`None`**.
- `plot_type` (<span style="color:green">**str**</span>, <span style="color:red">optional</span>): Type of plot for this dataset. Can be `'line'`, `'scatter'`,  `'both'` or `'line_scatter'`. Default is **`'line'`**.
- `annotate_text` (<span style="color:green">**str**</span>, <span style="color:red">optional</span>): Text to annotate on the plot. Default is None.
- `annotate_pos` (<span style="color:green">**tuple**</span>, <span style="color:red">optional</span>): Position for the annotation in the form `(x, y)`. Default is **`(0, 0)`**.
- `annotate_cords` (<span style="color:green">**str**</span>, <span style="color:red">optional</span>): Coordinate system for annotation position. Default is **`'data'`**.

### `plot`

```python
def plot(self, xlog_scale=False, ylog_scale=False, line_color=None, legend_title=None, ylim=None, save_name=None, close=True):
```

**Description:** Generates the plot with all the added datasets.

**Parameters:**
- `xlog_scale` (<span style="color:green">**bool**</span>, <span style="color:red">optional</span>): If `True`, use a logarithmic scale for the x-axis. Default is **`False`**.
- `ylog_scale` (<span style="color:green">**bool**</span>, <span style="color:red">optional</span>): If `True`, use a logarithmic scale for the y-axis. Default is **`False`**.
- `line_color` (<span style="color:green">**str**</span>, <span style="color:red">optional</span>): Color of the line plots. Default is **`None`** (automatic color assignment).
- `legend_title` (<span style="color:green">**str**</span>, <span style="color:red">optional</span>): Title for the legend. Default is **`None`**.
- `ylim` (<span style="color:green">**tuple**</span>, <span style="color:red">optional</span>): Limits for the y-axis in the form `(ymin, ymax)`. Default is **`None`**.
- `save_name` (<span style="color:green">**str**</span>, <span style="color:red">optional</span>): Filename for saving the plot. If `None`, the plot is not saved. Default is **`None`**.
- `close` (<span style="color:green">**bool**</span>, <span style="color:red">optional</span>): If True, closes the plot after saving to avoid display issues. Default is **`True`**.


## Notes
To plot multiple datasets on the same graph, use the `add_plot` method for each dataset before calling `plot`.</br>
The `plot_type` parameter in `add_plot` allows you to specify whether the data should be plotted as a line, scatter, both or line + scatter.</br>
Annotations can be added to individual datasets using the `annotate_text` and `annotate_pos` parameters in the `add_plot` method.</br>
The `plot` method clears the added datasets after generating the plot, so you can reuse the same `Plotter` instance for multiple plots.</br>

# Examples

## 1. Basic line plot

```Python
plotter = Plotter(xlabel='Time', ylabel='Value')
plotter.add_plot(x=[0, 1, 2, 3], y=[1, 2, 3, 4], label='Line 1')
plotter.plot(save_name='basic_line_plot')
```

## 2. Line and Scatter plot
```Python
plotter = Plotter(xlabel='Time', ylabel='Amplitude')
plotter.add_plot(x=[0, 1, 2, 3], y=[1, 4, 9, 16], label='Line and Scatter', plot_type='both')
plotter.add_plot(x=[0, 1, 2, 3], y=[1, 2, 3, 4], label='Line Only', plot_type='line')
plotter.plot(log_scale=False, save_name='combined_plot')
```

## 3. Adding annotations
```Python
plotter = Plotter(xlabel='Frequency', ylabel='Magnitude')
plotter.add_plot(
    x=[1, 2, 3, 4], 
    y=[10, 100, 1000, 10000], 
    label='Annotated Line', 
    plot_type='line', 
    annotate_text='Peak', 
    annotate_pos=(2, 100), 
    annotate_cords='data'
)
plotter.plot(log_scale=True, save_name='annotated_plot')
```
