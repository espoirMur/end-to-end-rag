from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import linkage
import numpy as np
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster
from typing import Dict
from src.shared.logger import setup_logger

DEFAULT_TRANSFORMER_KWARGS = {"trust_remote_code": True,
                              "device": "cpu",
                              "config_kwargs": {"use_memory_efficient_attention": False,
                                                "unpad_inputs": False}}
logger = setup_logger("cluster_modeler")


class HierachicalClusterModeler:
    """
    This class is responsible for clustering the documents
    """

    def __init__(self, documents: pd.DataFrame, embedding_model_id: str) -> None:
        self.documents = documents
        self.embedding_model_id = embedding_model_id
        current_directory = Path.cwd()
        self.current_directory = current_directory
        self.sentence_transformer_model = self.init_sentence_transformer()

    def init_sentence_transformer(self, transformer_kwargs: Dict = DEFAULT_TRANSFORMER_KWARGS) -> SentenceTransformer:
        """ Initialize the sentence transformer model """

        embedding_model_path = self.current_directory.joinpath(
            "models", self.embedding_model_id)
        model_path = self.current_directory.joinpath(self.embedding_model_id)
        transformer_kwargs["cache_folder"] = model_path
        transformer_kwargs["model_name_or_path"] = embedding_model_path.__str__()
        sentence_transformer_model = SentenceTransformer(
            **transformer_kwargs)
        return sentence_transformer_model

    def embed_documents(self):
        """ Embed the documents using the sentence transformer model """
        today_news_embeddings = self.sentence_transformer_model.encode(
            self.documents.content, show_progress_bar=True)
        return today_news_embeddings

    def compute_linkage(self, today_news_embeddings: np.array, method: str = "complete", metric: str = "cosine") -> np.array:
        """ Compute the sklearn linkage"""
        mergings = linkage(today_news_embeddings,
                           method=method, metric=metric)
        return mergings

    def select_best_distance(self, X: np.array, merging: np.array) -> Tuple[np.array, float]:
        """ start with the document embedding x, and the hierachical clustering, find the k that maximize the shilouette score"""
        max_shilouette = float("-inf")
        return_labels = np.zeros(X.shape[0])
        best_k = 0
        for k in np.arange(0.2, 0.7, 0.01):
            labels = fcluster(merging, k, criterion="distance")
            score = silhouette_score(
                X, labels
            )
            if score > max_shilouette:
                max_shilouette = score
                return_labels = labels
                best_k = k
        return return_labels, best_k

    def analyse_embeddings(self, embeddings: np.array, index: int, label_column: str = "labels") -> np.array:
        """ take a matrix of embeddings and the labels.
        for each label compute the cosine similarity of the document with that label.
        """
        document_in_index = self.documents.query(f"{label_column} == {index}")
        with pd.option_context('display.max_colwidth', None):
            print(document_in_index.title)
        document_index = document_in_index.index
        vectors = embeddings[document_index]
        return self.sentence_transformer_model.similarity(vectors,  vectors)

    def select_top_clusters(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """ select the clusters with the more than two documents """
        cluster_counts = news_df["labels"].value_counts()
        labels_with_more_than_one = cluster_counts[cluster_counts > 1].index
        important_news_df = news_df.loc[news_df.labels.isin(
            labels_with_more_than_one)]
        return important_news_df

    def run(self) -> pd.DataFrame:
        today_news_embeddings = self.embed_documents()
        mergings = self.compute_linkage(today_news_embeddings)
        return_labels, best_k = self.select_best_distance(
            today_news_embeddings, mergings)
        logger.info(
            f" finished clustering with best_k = {best_k} with and number_of_clusters = {np.unique(return_labels).shape[0]}")
        self.documents["labels"] = return_labels
        important_news_df = self.select_top_clusters(self.documents)
        logger.info(
            f"the important news data is of shape: {important_news_df.shape[0]}")
        logger.info(
            f"the number of labels are {np.unique(important_news_df.labels).shape[0]}")
        return important_news_df