"""
Update ClassBalanceEDA to plot class balance for a single target or for all columns when target is None.
- Numeric columns -> histogram
- Categorical columns -> frequency bar chart
"""
from .base import EDAComponent
from logs.logger import get_logger
from visualisations.factory import VisualisationFactory
import os
import math
import pandas as pd

class ClassBalanceEDA(EDAComponent):
    """
    EDA component to analyze and visualize class balance in the target variable.
    When `target` is None, produce a multi-panel figure showing distributions for
    all columns: histograms for numeric columns and bar charts for categorical columns.
    """

    def __init__(self):
        self.logger = get_logger("ClassBalanceEDA")
        self.logger.info("Initialized ClassBalanceEDA component")

    def _is_categorical(self, series: pd.Series) -> bool:
        # treat object, category, bool or low-cardinality numeric as categorical
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series) or pd.api.types.is_bool_dtype(series):
            return True
        # numeric with small number of unique values -> categorical
        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) <= 20:
            return True
        return False

    def run(self, data, target=None, text_field=None, save_path=None, **kwargs):
        """
        Analyze and visualize class balance in the target variable or for all columns.

        Parameters:
        - data: pandas DataFrame
        - target: optional column name to show a single class balance plot
        - save_path: directory where to save the output image
        - kwargs: additional parameters forwarded to visualisations

        Returns:
        - filepath: path to the saved image
        """
        self.logger.info(f"Running ClassBalanceEDA on target: {target}")

        if save_path is None:
            save_path = os.getcwd()

        filepath = os.path.join(save_path, "class_balance.png")

        # If a specific target is provided, keep original behaviour
        if target is not None:
            if target not in data.columns:
                self.logger.error(f"Target column '{target}' not found in data")
                raise ValueError(f"Target column '{target}' not found in data")

            class_counts = data[target].value_counts().to_dict()
            self.logger.info(f"Class distribution for {target}: {class_counts}")

            viz = VisualisationFactory.get_visualisation(
                "bar_chart",
                title=f"Class Balance: {target}",
                xlabel=target,
                xticks_rotation=45,
                ylabel="Count",
                figsize=(10, 6),
                **kwargs,
            )

            fig, ax = viz.plot(data=class_counts)
            viz.save(fig, filepath)
            return filepath

        # When target is None: create subplots for all columns
        if not isinstance(data, pd.DataFrame):
            self.logger.error("`data` must be a pandas DataFrame when target is None")
            raise ValueError("`data` must be a pandas DataFrame when target is None")

        cols = list(data.columns)
        if len(cols) == 0:
            self.logger.error("Empty DataFrame provided")
            raise ValueError("Empty DataFrame provided")

        # Determine types
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(data[c]) and not self._is_categorical(data[c])]
        categorical_cols = [c for c in cols if c not in numeric_cols]

        total_plots = len(cols)
        # layout: try square-ish grid
        ncols = int(math.ceil(math.sqrt(total_plots)))
        nrows = int(math.ceil(total_plots / ncols))

        # We'll build a single Matplotlib figure and let existing visualisers draw into axes
        # Use the factory to obtain histogram and bar_chart visualisers
        hist_viz = VisualisationFactory.get_visualisation("histogram", title=None, ylabel="Count", **kwargs)
        bar_viz = VisualisationFactory.get_visualisation("bar_chart", title=None, ylabel="Count", xticks_rotation=45, **kwargs)

        # Create a combined figure via the visualisers' plotting functions. We assume their
        # plot methods accept an `ax` kwarg to draw into an existing axis; if not, we fall back
        # to letting them return a fig and then embedding — to be defensive we'll handle both cases.
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3.5 * nrows))
        # axes could be a single Axes or array
        if total_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, col in enumerate(cols):
            ax = axes[idx]
            try:
                if col in numeric_cols:
                    # Pass series or DataFrame depending on viz implementation
                    result = hist_viz.plot(data=data[col].dropna(), ax=ax, title=col)
                else:
                    counts = data[col].value_counts(dropna=False).to_dict()
                    result = bar_viz.plot(data=counts, ax=ax, title=col)

                # If the visualiser returned a (fig, ax), we assume it created its own fig;
                # in that case we try to move artists into our combined ax (best-effort)
                if isinstance(result, tuple) and len(result) >= 2:
                    # visualiser created its own axes; try to copy artists
                    created_ax = result[1]
                    for artist in created_ax.get_children():
                        try:
                            artist.remove()
                            ax.add_artist(artist)
                        except Exception:
                            # ignore non-transferable artists
                            pass
                # otherwise we assume it drew into provided ax
            except Exception as e:
                self.logger.warning(f"Failed to plot column '{col}': {e}")
                ax.text(0.5, 0.5, f"Error plotting {col}", ha="center")

        # hide any unused axes
        for j in range(total_plots, len(axes)):
            try:
                axes[j].set_visible(False)
            except Exception:
                pass

        fig.tight_layout()
        # Save the combined figure
        try:
            fig.savefig(filepath, bbox_inches='tight')
        except Exception as e:
            self.logger.error(f"Failed to save class balance figure: {e}")
            raise

        self.logger.info(f"Saved class balance figure to {filepath}")
        plt.close(fig)
        return filepath

