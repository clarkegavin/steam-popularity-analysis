#pipelines/clustering_pipeline.py
import numpy as np
import os
from visualisations import VisualisationFactory
from .base import Pipeline
from logs.logger import get_logger
from collections import Counter

#from vectorizers.tfidf_vectorizer import TfidfTextVectorizer
#from clusterers.hdbscan_clusterer import HDBSCANClusterer
#from reducers.umap_reducer import UMAPReducer
#from visualisations.cluster_plotter import ClusterPlotter

from vectorizers import VectorizerFactory
from models import ModelFactory
from reducers import ReducerFactory

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

class ClusteringPipeline(Pipeline):
    def __init__(self, **params):
        self.logger = get_logger("ClusteringPipeline")

        self.name = params.get("name", "clustering_pipeline")
        self.logger.info(f"Initializing ClusteringPipeline with name: {self.name}")
        # params dict contains all YAML keys
        self.text_field = params.get("text_field")
        self.genre_field = params.get("genre_field")
        self.filter_genre = params.get("filter_genre")

        #Vectorizer
        vectorizer_cfg = params.get("vectorizer", {})
        vectorizer_name = vectorizer_cfg.get("vectorizer_name")
        vectorizer_field = vectorizer_cfg.get("vectorizer_field")
        vectorizer_params = vectorizer_cfg.get("vectorizer_params", {})
        self.logger.info(f"Setting up vectorizer '{vectorizer_name}' for field '{vectorizer_field}' with params {vectorizer_params}")
        vectorizer_params['column'] = vectorizer_field
        self.vectorizer = VectorizerFactory.get_vectorizer(vectorizer_name, **vectorizer_params)

        #Clusterer
        clusterer_cfg = params.get("clusterer", {})
        clusterer_name = clusterer_cfg.get("name")
        clusterer_params = clusterer_cfg.get("params", {})
        self.logger.info(f"Setting up clusterer '{clusterer_name}' with params {clusterer_params}")
        self.clusterer = ModelFactory.get_model(clusterer_name, **clusterer_params)

        #Reducer
        reducer_cfg = params.get("reducer", [])
        self.reducers = ReducerFactory.get_reducers(reducer_cfg)

        visualisations_cfg = params.get("visualisations", {})
        visualisations_name = visualisations_cfg.get("name")
        visualisations_params = visualisations_cfg.get("params", {})
        self.dimensions = visualisations_params.get("dimensions", 2)
        self.plotter = VisualisationFactory.get_visualisation(visualisations_name, **visualisations_params)

    def execute(self, df):
        self.logger.info("Starting clustering pipeline")

        # Filter genre
        # df_filtered = df[df[self.genre_field] == self.filter_genre].copy()
        # self.logger.info(f"Filtering records where {self.genre_field} == {self.filter_genre}")
        # self.logger.info(f"df_filtered type: {type(df_filtered)}")
        #
        # texts = df_filtered[self.text_field].fillna("").tolist()
        # self.logger.info(f"Filtered records: {len(texts)}")


        # Vectorize
        if self.vectorizer is not None:
            self.logger.info(f"Vectorizing texts using {self.vectorizer.name}")
            df = self.vectorizer.fit_transform(df)
            self.logger.info(f"Vectorized shape: {df.shape}")

        # Reduce
        df.columns = df.columns.astype(str) # ensure columns are str for reducers
        for reducer in self.reducers:
            self.logger.info(f"Reducing dimensions using {reducer.name}")
            df = reducer.fit_transform(df)
            self.logger.info(f"Shape after {reducer.name}: {df.shape}")

        # Cluster
        self.logger.info(f"Clustering using {self.clusterer.name}")
        labels = self.clusterer.fit_predict(df)
        self.logger.info(type(self.clusterer))
        probabilities = self.clusterer.probabilities_ if hasattr(self.clusterer, "probabilities_") else None
        self.logger.info(f"Cluster labels assigned: {set(labels)}")
        if probabilities is not None:
            self.logger.info(f"Cluster probabilities shape: {probabilities.shape}")
        else:
            self.logger.info("Cluster probabilities not available for this clusterer")


        # Reduce (for visualisation)
        viz_reducer = ReducerFactory.create_reducer(name='umap', n_components=self.dimensions)
        self.logger.info(f"Reducing dimensions using {viz_reducer}")
        # dimensions = self.dimensions
        # self.reducer.set_components(dimensions)
        X_reduced = viz_reducer.fit_transform(df)
        self.logger.info(f"Reduced shape: {X_reduced.shape}")

        # Plot
        self.logger.info(f"Plotting clusters using {self.plotter.name}")
        fig, ax, scatter = self.plotter.plot(X_reduced, labels)
        self.logger.info("Cluster plot generated")
        plot_path = os.path.join(self.plotter.output_dir, f"{self.name}_cluster_plot.png")
        self.plotter.save(fig, plot_path)
        self.plotter.save_embeddings(X_reduced, labels, df, prefix=f"{self.name}_clustering_pipeline")
        self.logger.info(f"Cluster plot saved as '{self.name}_cluster_plot.png'")
        self.plotter.save_interactive_plot(X_reduced, labels, prefix=f"{self.name}_cluster_plot")
        if probabilities is not None:
            self.plotter.save_interactive_plot_by_probability(
                X_reduced,
                labels,
                probabilities,
                prefix=f"{self.name}_cluster_plot_by_probability"
            )



        # Attach cluster labels
        df["cluster"] = labels

        if self.vectorizer is not None:
        # Optional: extract cluster keywords
            cluster_keywords = self._extract_cluster_keywords(df, labels)
            # Log cluster keywords
            label_counts = Counter(labels)
            for cluster_id, keywords in cluster_keywords.items():
                size = label_counts.get(cluster_id, 0)
                self.logger.info(
                    f"Cluster {cluster_id} (size={size}) keywords: {keywords}"
                )

            # save cluster keywords to file
            keyword_path = os.path.join(self.plotter.output_dir, f"{self.name}_cluster_keywords.txt")
            self._save_cluster_keywords(cluster_keywords, keyword_path)
        #self._save_cluster_keywords(cluster_keywords, "output/clustering_pipeline_cluster_keywords.txt")


        return df #, cluster_keywords

    def _save_cluster_keywords(self, cluster_keywords, filepath):
        try:
            self.logger.info(f"Saving cluster keywords to {filepath}")
            with open(filepath, "w", encoding = "utf-8", errors="replace") as f:
                for cluster_id, keywords in cluster_keywords.items():
                    f.write(f"Cluster {cluster_id} keywords: {', '.join(keywords)}\n")
            self.logger.info(f"Cluster keywords saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving cluster keywords to {filepath}: {e}")

    def _extract_cluster_keywords(self, X, labels, top_n=10):
        terms = self.vectorizer.get_feature_names()
        result = {}

        for cluster_id in set(labels):
            if cluster_id == -1:
                continue

            idx = np.where(labels == cluster_id)[0]
            centroid = X[idx].mean(axis=0).A1
            top_idx = centroid.argsort()[-top_n:]

            result[cluster_id] = [terms[i] for i in top_idx]

        return result
