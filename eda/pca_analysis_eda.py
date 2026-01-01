#eda/pca_analysis_eda.py
"""
EDA step: PCA analysis using existing PCA_Reducer.
- Fits PCA on numeric columns of the provided DataFrame (does not alter the DataFrame)
- Exports explained variance (per-PC and cumulative) to Excel
- Produces a scree / cumulative variance scatter plot via the visualisation factory
- Exports top-K loadings per principal component to Excel (one column per PC)

YAML params (passed directly to the EDA run call):
  n_components: int (default: 10)
  top_k: int (default: 10)
  explained_variance_filename: str (default: "pca_explained_variance.xlsx")
  loadings_filename: str (default: "pca_top_loadings.xlsx")
  scree_plot_filename: str (default: "pca_scree_plot.png")

This file follows the project's EDA conventions and reuses reducers.PCA_Reducer.
"""
from .base import EDAComponent
from logs.logger import get_logger
from visualisations.factory import VisualisationFactory
from reducers.pca_reducer import PCA_Reducer
import os
import pandas as pd
import numpy as np


class PCAAnalysisEDA(EDAComponent):
    def __init__(self, **kwargs):
        self.logger = get_logger("PCAAnalysisEDA")
        self.logger.info("Initialized PCAAnalysisEDA")

    def run(self, data, target=None, text_field=None, save_path=None, **kwargs):
        """Run PCA analysis.

        data: pandas DataFrame (post-preprocessing: encoded/scaled)
        save_path: directory where outputs will be written
        kwargs: supports n_components, top_k, explained_variance_filename,
                loadings_filename, scree_plot_filename
        """
        self.logger.info("Running PCAAnalysisEDA")

        if save_path is None:
            raise ValueError("save_path must be provided to write EDA outputs")

        os.makedirs(save_path, exist_ok=True)

        # Read params from kwargs with sensible defaults
        # Read params from kwargs with sensible defaults
        if "n_components" in kwargs:
            n_components = kwargs["n_components"]
            if isinstance(n_components, str) and n_components.lower() in {"none", "null"}:
                n_components = None
        else:
            n_components = 10

        if n_components is not None:
            try:
                n_components = int(n_components)
            except Exception:
                self.logger.warning("Invalid n_components; defaulting to 10")
                n_components = 10

        top_k = kwargs.get("top_k", 10)
        try:
            top_k = int(top_k)
        except Exception:
            top_k = 10

        explained_variance_filename = kwargs.get("explained_variance_filename", "pca_explained_variance.xlsx")
        loadings_filename = kwargs.get("loadings_filename", "pca_top_loadings.xlsx")
        scree_plot_filename = kwargs.get("scree_plot_filename", "pca_scree_plot.png")

        # Select numeric columns only for PCA
        try:
            numeric_df = data.select_dtypes(include=[np.number])
        except Exception:
            self.logger.error("Failed to select numeric columns for PCA")
            raise

        if numeric_df.shape[1] == 0:
            self.logger.error("No numeric columns available for PCA")
            raise ValueError("No numeric columns available for PCA")

        # Ensure n_components does not exceed number of features
        n_features = numeric_df.shape[1]
        if n_components is not None and n_components > n_features:
            self.logger.info(
                f"Requested n_components={n_components} > n_features={n_features}; reducing to {n_features}"
            )
            n_components = n_features

        # Prepare data matrix. PCA cannot accept NaNs: fill with column means when present
        X = numeric_df.copy()
        if X.isna().any().any():
            self.logger.info("Missing values detected in numeric data; filling with column means for PCA fit")
            X = X.fillna(X.mean())

        X_mat = X.values

        # Build and fit PCA_Reducer (reuse existing reducer)
        reducer = PCA_Reducer(name="pca", n_components=n_components)
        reducer.build()
        reducer.fit(X_mat)
        model = reducer.model

        if not hasattr(model, 'explained_variance_ratio_'):
            self.logger.error("PCA model does not expose explained_variance_ratio_")
            raise RuntimeError("PCA model not available after fit")

        evr = np.asarray(model.explained_variance_ratio_)
        cumulative = np.cumsum(evr)

        # Export explained variance to Excel
        pcs = [f"PC{i+1}" for i in range(len(evr))]
        df_evr = pd.DataFrame({
            "PC": pcs,
            "explained_variance_ratio": evr,
            "cumulative_explained_variance": cumulative
        })

        evr_path = os.path.join(save_path, explained_variance_filename)
        try:
            df_evr.to_excel(evr_path, index=False)
            self.logger.info(f"Wrote explained variance to {evr_path}")
        except Exception as e:
            self.logger.error(f"Failed to write explained variance Excel: {e}")
            raise

        # Scree / cumulative explained variance plot
        try:
            df_plot = pd.DataFrame({
                "pc_index": list(range(1, len(cumulative) + 1)),
                "cumulative_explained_variance": cumulative
            })

            viz = VisualisationFactory.get_visualisation(
                "pair_scatter",
                title="PCA cumulative explained variance",
                xlabel="Principal Component",
                ylabel="Cumulative Explained Variance",
            )

            if viz is not None:
                fig, axes = viz.plot(data=df_plot, pairs=[("pc_index", "cumulative_explained_variance")], ncols=1)
                scree_path = os.path.join(save_path, scree_plot_filename)
                viz.save(fig, scree_path)
                self.logger.info(f"Saved scree plot to {scree_path}")
            else:
                # Fallback: plot directly with matplotlib
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(df_plot['pc_index'], df_plot['cumulative_explained_variance'], marker='o')
                ax.set_xlabel('Principal Component')
                ax.set_ylabel('Cumulative Explained Variance')
                ax.set_title('PCA cumulative explained variance')
                scree_path = os.path.join(save_path, scree_plot_filename)
                fig.savefig(scree_path, bbox_inches='tight')
                self.logger.info(f"Saved scree plot to {scree_path} (matplotlib fallback)")
        except Exception as e:
            self.logger.error(f"Failed to create scree plot: {e}")

        # PCA loadings export: map components_ back to original feature names
        try:
            comps = model.components_  # shape (n_components, n_features)
            feature_names = [str(c) for c in numeric_df.columns]

            # For each component, pick top_k features by absolute loading magnitude
            top_k = min(top_k, n_features)
            loadings_dict = {}
            for i in range(comps.shape[0]):
                comp = comps[i]
                abs_idx = np.argsort(np.abs(comp))[::-1]  # descending by abs
                top_idx = abs_idx[:top_k]
                rows = []
                for idx in top_idx:
                    fname = feature_names[idx]
                    loading = float(comp[idx])
                    rows.append(f"{fname} ({loading:.6f})")
                loadings_dict[f"PC{i+1}"] = rows

            df_loadings = pd.DataFrame(loadings_dict)
            loadings_path = os.path.join(save_path, loadings_filename)
            df_loadings.to_excel(loadings_path, index=False)
            self.logger.info(f"Wrote PCA top-{top_k} loadings to {loadings_path}")
        except Exception as e:
            self.logger.error(f"Failed to compute or write PCA loadings: {e}")

        return {
            "explained_variance_path": evr_path,
            "scree_plot_path": os.path.join(save_path, scree_plot_filename),
            "loadings_path": os.path.join(save_path, loadings_filename)
        }

