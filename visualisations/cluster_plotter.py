# visualisations/cluster_plotter.py
import matplotlib.pyplot as plt
from .base import Visualisation
from logs.logger import get_logger
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
import numpy as np

class ClusterPlotter(Visualisation):
    """
    Cluster scatter plot visualisation.
    """

    def __init__(self, name: str = "cluster_plot",  title="Cluster Visualisation",
                 output_dir = ".", xlabel=None, ylabel=None, zlabel=None, figsize=(10, 6), **params):
        super().__init__(title=title, figsize=figsize)
        self.logger = get_logger(self.__class__.__name__)
        self.name = name
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel
        self.figsize = figsize
        self.params = params  # optional style parameters
        self.output_dir = output_dir
        self.logger.info(
            f"Initialized ClusterPlotter with title={title}, xlabel={xlabel}, ylabel={ylabel}, figsize={figsize}, params={params}"
        )

    def build(self):
        self.logger.info(f"Built ClusterPlotter with params: {self.params}")
        return self

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    def plot(self, X_reduced, labels, **kwargs):
        """
        Plot 2D or 3D cluster scatter plot.
        Automatically detects dimensionality.

        Parameters:
            X_reduced: np.ndarray of shape (n_samples, 2 or 3)
            labels: cluster labels for each sample
            kwargs: optional style overrides (e.g. cmap, alpha)
        """
        n_dims = X_reduced.shape[1]
        self.logger.info(f"Creating cluster plot with {X_reduced.shape[0]} points and {n_dims}D embedding")

        # --- 3D PLOT ---------------------------------------------------------
        if n_dims == 3:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection="3d")

            scatter = ax.scatter(
                X_reduced[:, 0],
                X_reduced[:, 1],
                X_reduced[:, 2],
                c=labels,
                **kwargs
            )

            ax.set_title(self.title)
            if self.xlabel:
                ax.set_xlabel(self.xlabel)
            if self.ylabel:
                ax.set_ylabel(self.ylabel)
            zlabel = self.params.get("zlabel")
            if zlabel:
                ax.set_zlabel(zlabel)

            self.logger.info("3D cluster plot created")
            return fig, ax, scatter

        # --- 2D PLOT ---------------------------------------------------------
        elif n_dims == 2:
            fig, ax = plt.subplots(figsize=self.figsize)
            scatter = ax.scatter(
                X_reduced[:, 0],
                X_reduced[:, 1],
                c=labels,
                **kwargs
            )

            ax.set_title(self.title)

            if self.xlabel:
                ax.set_xlabel(self.xlabel)
            if self.ylabel:
                ax.set_ylabel(self.ylabel)

            # Optional: xticks rotation
            rotation = self.params.get("xticks_rotation")
            if rotation is not None:
                ax.tick_params(axis='x', labelrotation=rotation)

            # Optional: label large clusters
            min_size = self.params.get("label_min_cluster_size", 100)
            self._label_large_clusters(ax, X_reduced, labels, min_cluster_size=min_size)

            self.logger.info("2D cluster plot created with cluster labels (if enabled)")
            return fig, ax, scatter

        else:
            raise ValueError(f"Can only plot 2D or 3D, got {n_dims} dimensions")


    def plot_deprecated(self, X_reduced, labels, **kwargs):
        """
        Plot 2D cluster scatter plot.

        Parameters:
            X_reduced: np.ndarray of shape (n_samples, 2)
            labels: cluster labels for each sample
            kwargs: optional style overrides (e.g., cmap, alpha)
        """
        self.logger.info(f"Creating cluster plot with {X_reduced.shape[0]} points")

        fig, ax = plt.subplots(figsize=self.figsize)
        scatter = ax.scatter(
            X_reduced[:, 0], X_reduced[:, 1], c=labels, **kwargs
        )

        ax.set_title(self.title)
        if self.xlabel:
            ax.set_xlabel(self.xlabel)
        if self.ylabel:
            ax.set_ylabel(self.ylabel)

        # Optional: xticks / yticks rotation from params
        rotation = self.params.get("xticks_rotation")
        if rotation is not None:
            ax.tick_params(axis='x', labelrotation=rotation)

        self.logger.info("Cluster plot created")
        return fig, ax, scatter

    def save_embeddings(self, X_embedded, labels, df_original, prefix="embedding"):
        """
        Save reduced coordinates + cluster labels + original metadata.
        """
        self.logger.info(f"Saving embeddings with prefix '{prefix}'")

        # Convert array → dataframe
        if X_embedded.shape[1] == 2:
            reduced_df = pd.DataFrame(X_embedded, columns=["x", "y"])
        elif X_embedded.shape[1] == 3:
            reduced_df = pd.DataFrame(X_embedded, columns=["x", "y", "z"])
        else:
            raise ValueError("X_embedded must be 2D or 3D for plotting.")

        # Add cluster labels
        reduced_df["cluster"] = labels

        # Add metadata from the original df (optional)
        # meta_cols = ["gameID", "Name", "Genre"]
        # for col in meta_cols:
        #     if col in df_original.columns:
        #         reduced_df[col] = df_original[col].values

        # Save to CSV
        # create output directory if it doesn't exist
        self.logger.info(f"Creating output directory at {self.output_dir} if it doesn't exist")
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, f"{prefix}.csv")
        reduced_df.to_csv(out_path, index=False)

        self.logger.info(f"Saved embedding CSV to {self.output_dir}")
        return out_path

    def _label_large_clusters(self, ax, X_reduced, labels, min_cluster_size=100):
        """
        Annotate only the largest clusters on the scatter plot.

        Parameters:
            ax : matplotlib axes object
            X_reduced : np.ndarray (n_samples, 2)
            labels : np.ndarray of cluster labels
            min_cluster_size : int
               Minimum samples required for a cluster to be labeled
        """
        counts = Counter(labels)
        large_clusters = {c: n for c, n in counts.items() if c != -1 and n >= min_cluster_size}

        self.logger.info(f"Labeling {len(large_clusters)} clusters (min size={min_cluster_size})")

        for c in large_clusters:
            # centroid in UMAP/embedding space
            centroid = np.mean(X_reduced[labels == c], axis=0)

            ax.text(
                centroid[0],
                centroid[1],
                f"Cluster {c}",
                fontsize=10,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="black", pad=2)
            )