from typing import List

from openparse.schemas import ParsedDocument
from sentence_transformers import SentenceTransformer

from src.shared.logger import setup_logger

logger = setup_logger("computing embeddings")


class EmbeddingComputer:
	def __init__(self, model_name: str) -> None:
		self.model_name = model_name
		self.init_model()

	def init_model(self):
		if getattr(self, "model", None):
			logger.info("model is already initialized")
		else:
			logger.info("initializing model")
			model = SentenceTransformer(self.model_name)
			self.model = model

	def compute_embeddings(
		self, documents: List[ParsedDocument], batch_size=4
	) -> List[ParsedDocument]:
		"""
		Given a list of the parsed documents, compute the embeddings for each node in the document.
		and assign the embeddings to the nodes.
		Returns the list of documents with the embeddings assigned to the nodes.
		"""
		all_nodes = self.collect_node_text(documents)
		all_embeddings = self.process_batches(all_nodes, batch_size)
		documents = self.assign_embeddings(documents, all_embeddings)
		return documents

	def collect_node_text(self, documents: List[ParsedDocument]) -> List[str]:
		"""Collect all nodes with text from the documents."""
		all_nodes = []
		for doc in documents:
			all_nodes.extend(doc.nodes)
		all_nodes = [f"passage: {node.text}" for node in all_nodes]

		return all_nodes

	def process_batches(self, all_nodes: List[str], batch_size: int) -> List:
		"""Process nodes in batches and compute embeddings."""
		all_embeddings = []
		for i in range(0, len(all_nodes), batch_size):
			batch_nodes = all_nodes[i : i + batch_size]
			batch_embedding = self.model.encode(
				batch_nodes,
				convert_to_tensor=False,
				show_progress_bar=True,
				normalize_embeddings=True,
			)
			all_embeddings.extend(batch_embedding)
		return all_embeddings

	def assign_embeddings(
		self, documents: List[ParsedDocument], all_embeddings: List
	) -> List[ParsedDocument]:
		"""Assign embeddings to nodes in the documents."""
		embedding_index = 0
		for doc in documents:
			for node in doc.nodes:
				node.embedding = all_embeddings[embedding_index].tolist()
				embedding_index += 1
		return documents
