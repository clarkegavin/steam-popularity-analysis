#visualisations/bar_chart.py
from .base import Visualisation
from logs.logger import get_logger

class BarChart(Visualisation):
    """
    Bar Chart Visualisation.
    """

    def __init__(self, title: str, xlabel=None, ylabel=None, figsize=(10,6), **params):
        super().__init__(title=title, figsize=figsize)
        self.logger = get_logger(self.__class__.__name__)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.figsize = figsize
        self.params = params  # optional style parameters
        self.logger.info(f"Initialized Bar Chart visualisation with title: {title}, xlabel: {xlabel}, ylabel: {ylabel}, figsize: {figsize}, params: {params}")

    def build(self):
        self.logger.info(f"Built Bar Chart visualisation with params: {self.params}")
        return self


    def plot(self, data, ax=None, title=None, **kwargs):
        """
        Draw a bar chart.
        - For categorical inputs (Series of object/categorical dtype, list-like of strings), compute value counts and plot frequencies.
        - If there are more than `top_n` categories (default 20), only plot the top `top_n` by frequency.
        - Accepts dict-like {label: value} by plotting values (limited to top_n labels by value if >top_n).
        - If `ax` is provided, draw into it; otherwise create a new figure.
        Returns (fig, ax) or (None, None) on failure.
        """
        import matplotlib.pyplot as plt
        try:
            import pandas as pd
            import numpy as np
        except Exception:
            pd = None
            np = None

        self.logger.info(f"Creating Bar Chart visualisation with data type: {type(data)}")

        created_fig = None
        if ax is None:
            created_fig, ax = plt.subplots(figsize=self.figsize)
        else:
            created_fig = ax.figure

        # Default top_n for categorical trimming
        top_n = int(self.params.get("top_n", 20))

        try:
            labels = None
            values = None

            # Dict-like input: treat as mapping label->value; if too many categories, take top by value
            if hasattr(data, "items") and not hasattr(data, "values"):
                items = list(data.items())
                # ensure values are numeric-ish for sorting; if not, convert to counts of keys
                try:
                    # sort by value descending
                    items_sorted = sorted(items, key=lambda kv: (kv[1] is None, -float(kv[1]) if kv[1] is not None else 0))
                except Exception:
                    items_sorted = items
                if len(items_sorted) > top_n:
                    items_sorted = items_sorted[:top_n]
                labels, values = zip(*items_sorted) if items_sorted else ([], [])

            # Pandas Series: compute value_counts to get frequencies
            elif pd is not None and isinstance(data, pd.Series):
                counts = data.value_counts(dropna=False)
                if len(counts) > top_n:
                    counts = counts.nlargest(top_n)
                labels = list(counts.index)
                values = list(counts.values)

            # List-like or ndarray: convert to Series and compute counts
            else:
                try:
                    if pd is not None:
                        ser = pd.Series(data)
                        counts = ser.value_counts(dropna=False)
                        if len(counts) > top_n:
                            counts = counts.nlargest(top_n)
                        labels = list(counts.index)
                        values = list(counts.values)
                    else:
                        # fallback: iterate and plot raw values (may be numeric)
                        values = list(data)
                        labels = list(range(len(values)))
                except Exception:
                    # final fallback: try to treat data as simple mapping of index->value
                    try:
                        labels = getattr(data, 'index', None)
                        values = getattr(data, 'values', data)
                    except Exception:
                        labels = None
                        values = data

            # Ensure labels are string-friendly when categorical
            if labels is not None:
                labels_plot = [str(l) for l in labels]
                ax.bar(labels_plot, values, **kwargs)
            else:
                ax.bar(range(len(values)), values, **kwargs)

            # Apply labels / title (title param overrides instance title)
            ax.set_title(title or self.title)
            if self.xlabel:
                ax.set_xlabel(self.xlabel)
            if self.ylabel:
                ax.set_ylabel(self.ylabel)

            rotation = self.params.get("xticks_rotation")
            if rotation is not None:
                ax.tick_params(axis='x', labelrotation=rotation)

            self.logger.info("Bar Chart visualisation created")
            return created_fig, ax

        except Exception as e:
            self.logger.exception(f"Error plotting bar chart: {e}")
            return None, None
